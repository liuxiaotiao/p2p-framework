# 迁移到 vLLM：可行性、方案与开发目标

> 状态：**规划中**（2026-10-07 立项，尚未动工）。本文是后续开发的目标与验收标准。
> 与 `TODO.md` 同一写法：每条写清楚**要动哪些文件**、**会破坏什么既有性质**、**怎么验收**。
> 涉及 vLLM 内部接口的地方都标了「⚠ 需在目标版本核对」—— vLLM 迭代很快，
> 内部 API 名称与签名以动工时装的版本为准，不以本文为准。

---

## 0. 结论先行

| 方案 | 一句话 | 保留动态配对 / 前段共享 | 工作量 | 建议 |
|---|---|---|---|---|
| **C** 框架内借用 vLLM 的算子 | 只换节点上的计算（`torch_model.py`），通信与调度不动 | ✓ | 小–中 | **第一阶段做**：直接打现在的瓶颈 |
| **B** 每条通道一个 vLLM 流水线 | 5 条通道各起一个 vLLM 流水线并行实例 | ✗（≈ 静态模式） | 中 | **第二阶段做**：论文的对照基线 |
| **A** vLLM 作每段的计算引擎、保留本框架调度 | 用 vLLM 的模型层 + KV 管理，绕过它的调度器 | ✓ | 大 | **暂缓**：收益与 C 接近、与 vLLM 内部耦合深 |

理由：实测每 token ~140ms，其中**通信只占 4–7%**，主体是每层 ~2.9ms 的 Python 派发与
小 kernel 开销（A30 与 T4 每层耗时几乎一样 —— 不是算力瓶颈）。整体迁移到 vLLM 的收益
几乎全部来自它的 kernel 与 CUDA Graph，与「通信方案」本身无关。所以先在框架内拿到这部分
收益（C），再用 vLLM 原生流水线做一个公平的对照（B）。

---

## 1. 现状基线（迁移前要对齐的数字）

部署：Qwen3-30B-A3B（48 层 × 128 专家，top-8），15 台（A30 / RTX6000 / T4，24/16GB），
方案 `qwen3-30b/plan_es90_fill.json`：5 个前段（层 1–8，全专家）+ 5 条后段通道
（gsm8k×2、mbpp×2、no_robots×1，层 9–48，各 task 的 90% 高频专家并集），
`miss_policy=local_topk`，LR 选择器（前 8 层 body token 逐层计数）。

| 指标 | 值 | 出处 |
|---|---|---|
| TPOT（单请求，无排队） | 129–145ms/token，随通道 | `results/sel_static4.json` |
| 每层每 token decode | ~2.9ms（快路径后；之前 4–5ms） | 节点算力占比折算 |
| 跨 4.x 子网的配对 | 每 token 多 20–25ms | `results/sel_dyn2.json` |
| 时延构成 | 计算 ~75–82%，网络+开销 4–7% | 同上 |
| 选择器 | 150/150（5 类，argmax） | 连续测试 |
| 连续测试（150 条、在途 5、max 2048） | 见 `results/run2.json`（关换绑后的干净版本） | `deploy_15.sh continuous` |

**迁移后必须用同一套测试复测**：`TestSamples/samples.txt` → `examples/build_cases.py`
→ `deploy_15.sh continuous`，同样 150 条、同样在途数、同样 `max_tokens=2048`、
`enable_thinking=False`、贪心解码。指标口径见 `p2pmoe/deploy/continuous.py`。

---

## 2. 本框架与 vLLM 的结构性差异

| 本框架 | vLLM | 影响 |
|---|---|---|
| **请求级动态路由**：前段算完 L₀=8 层、LR 判 task 后才决定走哪条后段，前后段任意组合（`coordinator._bind`） | 一个引擎 = 一条启动时固定的流水线（TP×PP），所有请求走同一条 | vLLM 原生 PP 表达不了「一个前段接多条后段」 |
| **前段共享、后段按 task 分池**（`free_fronts` / `free_backs`） | 流水线各级一一对应 | 方案 B 只能退化成静态配对 |
| **每段只驻留部分专家**（逐层不同的专家 id 列表，`NodeConfig.layer_experts`） | 默认加载全部专家；EP 是把全部专家切开分摊，不是「只装子集」 | 要自定义模型 + 权重加载 + 全局→本地专家号映射 |
| **段间传 hidden state**（`wire.py`，float32 numpy，TCP_NODELAY） | 对外只有 token in / token out；PP 内部用 `IntermediateTensors` 走 NCCL ⚠ 需核对 | 方案 A 需要把中间张量暴露出来 |
| **逐层路由统计旁路**（LR 特征、miss 率、通道二） | 无 | 需在自定义模型里加钩子 |
| **异构卡**（A30 sm80、RTX6000 sm75、T4 sm75） | 新版本对 Turing（sm75）的支持在收缩 ⚠ 需核对；FlashAttention 系后端要求 sm80+ | T4/RTX6000 可能要退到 SDPA/xformers 类后端，或干脆不能跑 |
| **一个前段同时只服务一条请求** | 连续批处理，同一引擎同时跑很多请求 | 这是 vLLM 最大的潜在收益之一（见 C4） |

---

## 3. 第一阶段：方案 C —— 框架内加速（不迁移调度与通信）

目标：**decode 每层 ≤ 0.5ms**（现 ~2.9ms），TPOT 从 ~140ms 降到 ≤ 40ms；
输出与现实现**逐位一致**（或在明确声明的数值容差内，见 C3）。

### C1 · 专家权重堆叠 + 融合 MoE kernel

* **做什么**：每层把驻留专家的 `gate/up/down` 堆成连续张量 `W13[n_res, 2I, d]`、
  `W2[n_res, d, I]`；建全局→本地专家号查表 `g2l[E]`（非驻留为 −1）。decode 与 prefill
  的专家计算都改为一次融合 kernel 调用（vLLM 的 fused MoE ⚠ 需核对函数名与签名，
  候选 `vllm.model_executor.layers.fused_moe` 下的 `fused_experts`/`fused_moe`），
  传入本地专家号与权重。
* **要动**：`runtime/torch_model.py`（`TorchPartialExpertMoEBlock.__init__/_choose/forward/
  forward_decode`）、`runtime/weights.py`（加载时按层堆叠，**逐层堆、堆完立刻释放原张量**）。
* **会破坏**：
  - 显存峰值：堆叠期间一层的专家要有两份。前段一层 128 专家 ≈ 1.2GB，N05/N08/N13 只剩
    ~3GB 余量 —— 必须逐层做、做完 `del` + `empty_cache`；
  - `resident_bytes` 的计算口径（改成对堆叠张量求和）；
  - `test_moe_block_only_touches_selected_experts`（它监视 `_gate` 字典的访问）要改写成
    「只取了选中的本地行」；
  - 逐位一致：融合 kernel 的累加顺序与现在按专家号升序 `index_add_` 不同，**一般不再逐位
    一致**。验收改为与现实现的 hidden state 余弦 ≥ 0.9999、贪心解码前 256 token 一致率
    ≥ 99%（在 sel_cases 上）。
* **T4/RTX6000**：融合 MoE kernel 多为 Triton 实现，sm75 能否跑 ⚠ 需核对；不能跑就在这两类
  卡上退回「堆叠 + bmm/grouped GEMM」的纯 torch 实现（同样消掉逐专家的 Python 循环）。
* **验收**：`tests/test_fast_decode.py` 改为容差比较并全部通过；单层 decode 微基准
  （新增 `examples/bench_layer.py`）在 A30 上 ≤ 0.3ms。

### C2 · 静态 KV 缓存 + 注意力后端

* **做什么**：把 `_attend` 里每步 `torch.cat` 增长的 KV 换成按请求槽位预分配的
  `K/V[slot, max_ctx, n_kv, d]` + 长度张量；注意力用 `F.scaled_dot_product_attention`
  （`enable_gqa=True`，去掉两次 `repeat_interleave`）或 vLLM 的 paged attention ⚠ 需核对 sm75。
* **要动**：`torch_model.py`（`_attend`、`TorchSegmentModel._kv`、`drop_kv/kv_tokens/
  active_requests`），`node.py` 的 release/`_stale` 逻辑只依赖 `drop_kv`，不用改。
* **会破坏**：KV 显存从「按需增长」变成「预留」。`max_ctx = prompt + 2048` 时每请求每层
  2×4×128×2B×~2.6K ≈ 5.3MB；N08（34 层）一个槽位 ~180MB —— 每个节点按显存余量定槽位数，
  写进 `NodeConfig`，规划器的 `kv_gb` 要按槽位数重算（`planner/hf_config.py: kv_gb_at`）。
* **验收**：与现实现的 decode 输出余弦 ≥ 0.9999；KV 不再随步数产生新的分配
  （`torch.cuda.memory_stats` 的 `num_alloc_retries` 不增长）。

### C3 · decode 的 CUDA Graph

* **前提**：C1（专家选择在 GPU 上完成，**不再有 `.tolist()` 同步**）+ C2（静态形状）。
  统计（hist/miss）已经在 GPU 上累加（`forward_decode` 的 `acc`），保持不变。
* **做什么**：每个节点按槽位捕获「本段全部层的一次 decode」为一张图；输入 hidden state、
  位置、KV 长度写进静态缓冲区后 replay。
* **要动**：`torch_model.py` 新增 `GraphRunner`；`node.py` 的 `_forward(decode=True)`
  分支调用它。
* **会破坏**：
  - 首次捕获有冷启动（每槽位一次）—— 预热计划（`control._warmup_plan`）要覆盖每个槽位；
  - `P2P_FAST_DECODE=0` 的对照开关要扩展为三档（慢路径 / 快路径 / 图）。
* **验收**：与 C2 结果逐位一致（图只是重放同一串 kernel）；每层 decode ≤ 0.5ms（含 T4）。

### C4 · 一个节点同时服务多条请求（批 decode）

* **背景**：连续测试里，排队的请求一直占着前段（前段要留它的 KV 给后续 decode），
  在途上限 = 前段数 = 5，导致后段闲置、GPU 利用率均值只有 7.9%。
* **做什么**：
  1. 协调器：前段/后段从「一次一条」改为「容量 N 条」（`free_fronts`/`free_backs` 变成
     计数池，`_release_front/_release_back` 计数归还）；
  2. 节点：同一 decode 步到达的多条请求合批（按很短的时间窗攒批），一次 forward；
  3. 环：每条请求仍是独立的环，合批只在节点内部发生。
* **要动**：`runtime/coordinator.py`、`runtime/node.py`（`_on_loop/_on_seg_in/_on_hop`
  加攒批队列）、`torch_model.py`（批 decode，依赖 C2 的槽位）、`deploy/continuous.py`
  （默认在途数改为 Σ 后段容量）。
* **会破坏**：「一条请求独占一条通道」是静态模式与通道二的隐含前提 —— 换绑（`_rebind`）
  与静态配对（`front_pools`）都要按容量语义重写；逐请求的 `compute_ms` 埋点要改成按批分摊。
* **验收**：连续测试吞吐 ≥ 现在的 3 倍；TPOT p99 不劣于单请求的 1.5 倍。

### C5 · 顺带要修的已知问题（与迁移无关，但会影响所有对比实验）

| 问题 | 现象 | 修法 | 要动 |
|---|---|---|---|
| **换绑不补上下文** | 中途换绑后新后段只重做 prompt 的 prefill，已生成 token 不在新后段 KV 里、位置编码错位 → 输出退化成复读（连续测试 6 条 2048 token 垃圾） | 前段把每步 decode 的 L₀ 输出也追加进 `cached_l0`；换绑时把 prompt + 已生成部分整段交给新后段 prefill，再继续 decode | `node.py`（`_front_tail` 的 decode 分支、`_on_bind`）、`coordinator._rebind` |
| **通道二对 no_robots 过敏** | 选择器全对，仍有 10/11 次报警落在 no_robots | 用 `--no-rebind` 跑的报警计数重新标定 `alarm_factor`，或按 task 设阈值 | `coordinator.alarm_baseline` |
| **no_robots 只有 1 条通道** | 均匀到达下成为瓶颈（TTFT p99 278s 几乎全是后段排队） | 重新规划 `TASKS=gsm8k=1,mbpp=1,no_robots=2` | `qwen3-30b/plan_*.json` |
| **报表缺口** | 去排队的端到端时延、按请求的 GPU 占用、MFU 没有进汇总 | `continuous.run_metrics` + `deploy_15.sh report` | 同左 |

---

## 4. 第二阶段：方案 B —— vLLM 原生流水线作为对照基线

目标：在**同样的 15 台机器、同样的层切分、同样的专家子集**上跑 vLLM，得到与本框架
（动态模式）可直接对比的时延 / 吞吐 / 资源数据。

### B1 · 拓扑

* 每条后段通道 + 静态配对给它的那个前段 = 一个 vLLM 流水线并行实例：

  | 实例 | PP 各级（节点：层） |
  |---|---|
  | gsm8k-0 | N01:1–8 → N03:9–21 → N05:22–48 |
  | gsm8k-1 | N10:1–8 → N04:9–24 → N02:25–48 |
  | mbpp-0 | N11:1–8 → N06:9–22 → N07:23–48 |
  | mbpp-1 | N15:1–8 → N13:9–35 → N14:36–48 |
  | no_robots | N12:1–8 → N09:9–14 → N08:15–48 |

  （与 `sel_static4` 的静态配对一致，见 `control.py` 的「静态链路：按清单里存的组合矩阵贪心配对」。）
* 不均匀切层：`VLLM_PP_LAYER_PARTITION="8,13,27"` 这类环境变量 ⚠ 需核对目标版本是否支持
  及写法；不支持就在自定义模型里自己按 rank 取层段。
* 多机：每个实例一个 Ray 集群（或 vLLM 的多节点启动方式 ⚠ 需核对），三台机器组成一个 PP 组。
  NCCL 跨 192.168.x 子网：设 `NCCL_SOCKET_IFNAME`、放行端口；4.x 子网与其他子网之间的慢链路
  （~11.5ms）只影响 mbpp-1 以外的组合 —— B 方案是静态配对，不会走到它。

### B2 · 只装专家子集的自定义模型（vLLM 插件）

* 通过 vLLM 的插件机制注册一个 `Qwen3MoePartialExperts` 架构（`ModelRegistry.register_model`
  ⚠ 需核对），基于 vLLM 内置的 Qwen3-MoE 实现改：
  - 构造时读本框架的方案 JSON（路径用环境变量传入），取本 rank 负责的层与每层驻留专家；
  - 权重加载只读这些专家的张量（复用 `runtime/weights.py` 的按 key 选择性加载思路）；
  - 路由：在驻留集内取 top-k（与 `miss_policy=local_topk` 同口径），专家号经 `g2l` 映射到
    本地堆叠权重；
  - 统计钩子：前 8 层按请求记录 body token 的逐层专家计数（供 LR），后段记 miss 率 ——
    需要拿到每个 token 属于哪条请求 ⚠ 需核对 vLLM 的 forward context / attention metadata
    能否取到请求边界；做不到就只在 B 里做「静态 + 真实 task 标注」的对照，不跑选择器。
* **任务路由**：B 是静态配对，请求按 task 进对应实例。两种做法：
  1. **oracle**：按真实 task 标注分发（对应本框架的静态模式，最干净的对照）；
  2. **前置选择器**：网关先用一台机器跑前 8 层 + LR，再把请求转给对应实例 —— 前 8 层算了两遍，
     只用来说明「没有前后段解耦时要付出的代价」。

### B3 · 测量对齐

* 复用 `deploy/continuous.py` 的闭环驱动与指标：新增一个「vLLM 后端」的提交函数，
  用 OpenAI 兼容接口的**流式**返回测 TTFT/TPOT（首个 chunk 到达 = 首 token）；
* 同一份 `TestSamples/cases.jsonl`，prompt 按 `segments` 拼好后作为 chat 消息发送
  （`enable_thinking=False` 通过 chat template 参数传 ⚠ 需核对）；
* 采样：贪心（`temperature=0`），`max_tokens=2048`；
* 资源：节点上的 `gpu_stats.snapshot()` 与 vLLM 进程无关，照样能采（同一块卡）。

### B4 · 对比实验矩阵

| 编号 | 系统 | 路由 | 目的 |
|---|---|---|---|
| E1 | 本框架 · 静态 | oracle | 本框架下限 |
| E2 | 本框架 · 动态 | LR | 本框架主结果 |
| E3 | vLLM · 5 实例 PP | oracle | 主流框架、同等专家子集 |
| E4 | vLLM · 5 实例 PP | 前置 LR | 「不解耦」的代价 |
| E5 | E2 + C1–C4 | LR | 加速后的本框架 |

每组都跑完整 150 条连续测试，报：生成时延（含 / 去排队）mean/p50/p99、TTFT（含 / 去排队）、
TPOT、吞吐、各节点 GPU 利用率与显存、识别正确率。

---

## 5. 方案 A（暂缓）：为什么不做、要做的话做什么

* **做法**：每个节点起一个只含本段层的 vLLM 模型执行器（不用它的调度器），由本框架的
  节点 agent 直接喂 hidden state、取 hidden state；KV 用 vLLM 的块管理。
* **难点**：
  - vLLM 的执行入口围绕「调度器产出的一批请求」设计，单请求、单步、外部驱动地调用它需要
    自己构造调度输出与 attention 元数据 ⚠ 需核对，且这部分是内部接口，版本间变化大；
  - 中间 hidden state 的输入输出（PP 里叫 `IntermediateTensors`）不是公开 API；
  - 版本升级大概率要重写适配层。
* **收益**：与 C1–C4 之和接近（都是 kernel + 图 + 批处理），多出来的只有 paged KV 的显存利用率。
* **重新评估的条件**：C 做完后若每层仍 > 1ms，或需要上百条并发的 paged KV 时再考虑。

---

## 6. 动工前要在集群上确认的事

| # | 要确认 | 怎么确认 | 影响 |
|---|---|---|---|
| 1 | vLLM 版本与 torch 版本兼容性（现 `moe` 环境是 torch 2.9.0+cu128） | 新建独立 conda 环境装 vLLM，**不要装进 `moe` 环境** | 所有方案 |
| 2 | T4 / RTX6000（sm75）能否跑目标版本的 vLLM，用哪个注意力后端 | 在 N12（T4）、N10（RTX6000）上跑最小模型 | B 是否能用全部 15 台；C1 的融合 kernel |
| 3 | 不均匀切层的写法 | 2 台机器 PP=2、指定切分跑通 | B1 |
| 4 | 跨子网 NCCL | 4.x 与 1.x 之间跑 `nccl-tests` 或 vLLM 多机 PP | B1（mbpp-1 组内都在 4.x，其余组不跨 4.x） |
| 5 | 插件模型能否拿到按请求的 token 边界 | 读目标版本的 forward context 实现 | B2 的统计钩子 / 前置选择器 |
| 6 | 融合 MoE kernel 的函数名、签名、对「非连续专家号」的支持 | 读源码 + 单层对拍 | C1 |

---

## 7. 里程碑与验收

| 阶段 | 内容 | 验收 |
|---|---|---|
| M0 | 第 6 节全部确认，结论补进本文 | 本文每个「⚠ 需核对」都有答案 |
| M1 | C1 + C2 | 单层 decode ≤ 0.5ms（A30），对拍通过；连续测试 TPOT ≤ 60ms |
| M2 | C3 | 每层 ≤ 0.3ms（A30）/ 0.6ms（T4）；TPOT ≤ 40ms |
| M3 | C4 + C5 | 连续测试吞吐 ≥ 现状 3 倍；无 2048 token 复读；换绑后输出质量与不换绑一致 |
| M4 | B1–B3 | vLLM 5 实例跑通 150 条连续测试，指标与本框架同口径落盘 |
| M5 | B4 实验矩阵 | E1–E5 全部跑完，结果表进论文 |

每个里程碑结束：`python3 -m pytest -q` 全绿，`deploy_15.sh continuous` 在集群上复测一次，
结果文件按 `results/<里程碑>_run.json` 保存。
