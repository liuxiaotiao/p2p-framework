# 从零复现：Qwen3-30B-A3B + task 选择器 + 15 节点两段式部署

> 适用场景：控制机（wash1）上的 `/home/ubuntu/p2p-framework` 丢了或要换一台控制机。
> Mac 上的 `p2p-framework` 是完整、最新的一份，所有代码、方案、选择器、样本清单都在里面；
> 只有**下载的权重**和 **`results/`** 不在 Mac 上 —— 权重可以重下，结果只能重跑。

集群约定（与 `task/hosts.txt` 一致）：

| 角色 | 节点 |
|---|---|
| 控制机 | wash1 = **N10**（192.168.3.2；2026-11-05 重新编号之前是 N09 = 192.168.3.4） |
| 前段（层 1–8，全专家） | N01、N10、N11、N12、N15 |
| 后段（层 9–48，按 task 的专家子集） | gsm8k：N03→N05、N04→N02；mbpp：N06→N07、N13→N14；no_robots：N09→N08 |
| 代码目录（每台都一样） | `/home/ubuntu/p2p-framework` |
| 权重目录（每台都一样） | `/home/ubuntu/p2p-framework/weights-q3-30b` |
| Python（节点与控制机） | `conda activate moe`（`/home/ubuntu/miniforge3/envs/moe`（2026-11 之前是 anaconda3），Python 3.10） |

---

## 1. 把代码恢复到 wash1

**在 Mac 上**（`wash1` 换成你 ssh 登录控制机用的地址/别名）：

```bash
cd "/Users/qingzheguo/Desktop/cc workspace/p2p-framework"
rsync -av --exclude '__pycache__' --exclude '.pytest_cache' --exclude 'weights*/' \
  ./ ubuntu@wash1:/home/ubuntu/p2p-framework/
```

> ⚠ **在第 1 步完成之前，绝对不要在 wash1 上跑 `deploy_15.sh sync`。**
> `sync` 是以 wash1 的目录为准、带 `--delete` 推到另外 14 台的：wash1 上是空目录或残缺目录时，
> 它会把其余节点上的代码一起删掉（权重目录被排除，不受影响）。

**在 wash1 上**核对版本（与 Mac 上一致即可；下面是 2026-10-13 时的值）：

```bash
cd /home/ubuntu/p2p-framework
md5sum p2pmoe/runtime/node.py p2pmoe/runtime/coordinator.py p2pmoe/runtime/torch_model.py \
       p2pmoe/deploy/control.py p2pmoe/deploy/continuous.py deploy_15.sh | cut -c1-12
```

## 2. 环境（每开一个新终端都要做）

```bash
cd /home/ubuntu/p2p-framework
conda activate moe
source qwen3-30b/env.sh            # WORKDIR / NODE_PY / REPO / WEIGHTS / PLAN / PROFILE / LR_MODEL / PROBE …
export ADVERTISE=192.168.3.2       # 控制机对节点可见的 IP —— 不设的话 measure/continuous 直接退出
mkdir -p results                   # tee 写日志前目录必须存在
```

**按清单设环境（推荐）**：上面四行加上 `PLAN` / `HOSTS` 一次设好，hosts 自动按清单用到的节点生成，
并核对 WORKDIR / NODE_PY / REPO / WEIGHTS：

```bash
cd /home/ubuntu/p2p-framework && conda activate moe
source qwen3-30b/use_plan.sh vllm        # 或 full_2ch / doc_v2 / doc_v2_5ch / full / doc_l8 / es90_fill / 清单路径
HOSTS=$HOSTS_ALL bash ./deploy_15.sh stop   # 停 agent 用全部 15 台
```

没 source 就跑 `deploy_15.sh` 时，脚本会退回旧默认值（`/home/ubuntu/P2P_MoE`、系统 Python 3.8、Qwen3-Next）——
症状是 fetch 预检报 `Addr = tuple[str, int]` / `'type' object is not subscriptable`。

一次性（`moe` 环境没被删就不用重装，装过会直接跳过）：

```bash
pip install -r requirements-control.txt      # 含 pyarrow（build_cases 读 parquet 用）、scikit-learn、joblib
tar xzf Delta_30B/delta_30b.tgz -C Delta_30B  # 选择器复核用的 probe 数据 → $PROBE
```

想以后不用每次手敲 `ADVERTISE`：把 `export ADVERTISE=192.168.3.2` 追加到 wash1 上
`qwen3-30b/env.sh` 的末尾（只在控制机上这么改）。

## 3. 模型文件

```bash
# 3a. 控制机的模型元数据（config.json + tokenizer，约 10MB，不含权重）
bash ./deploy_15.sh meta

# 3b. tokenizer 与选择器训练口径一致（5 个 task 都应 N/N ✓）
python3 -m p2pmoe.deploy.lr_tool tokens --model-dir "$WEIGHTS" --probe-dir "$PROBE"

# 3c. wash1 自己（N09）的后段权重 —— 跟着目录一起没了，单独重拉（约 4GB）
bash ./deploy_15.sh refetch N09

# 3d. 15 台全部核对（只读文件头，几十秒）
bash ./deploy_15.sh verify
```

`verify` 里哪台不是 ✓ 就 `bash ./deploy_15.sh refetch <节点>`。全部重下用 `bash ./deploy_15.sh fetch`
（约 200GB，前段每台 ~10.6GB、后段 4–21.5GB；HF 走镜像就先 `export HF_ENDPOINT=…`）。

## 4. 同步代码、起 agent

```bash
bash ./deploy_15.sh sync           # 代码推到其余 14 台（wash1 自己跳过；权重目录不动）
bash ./deploy_15.sh stop
bash ./deploy_15.sh start          # 应该立刻返回 15/15
bash ./deploy_15.sh status         # 15/15 在线；不应有「⚠ 持有 KV 的请求」
```

N09 上原来那个 agent 是从被删的目录里跑起来的，**必须 stop/start 一次**。

## 5. 实验

所有实验的共同设置：动态/静态模式由命令决定；`miss_policy=local_topk`（后段只在驻留专家里取
top-8）；选择器 τ_hi=0.9、只数 body token、`enable_thinking=False`；贪心解码。
**每个实验换一个 `RESULTS` 与 `DUMP_FRONT` 文件名**，否则会互相覆盖。

### 5.1 静态模式（20 条，选择器只旁路）

```bash
export DUMP_FRONT=results/sel_front_static.jsonl
RESULTS=results/sel_static.json bash ./deploy_15.sh measure --static 2>&1 | tee results/sel_static.log
bash ./deploy_15.sh report results/sel_static.json
python3 -m p2pmoe.deploy.lr_tool check --model $LR_MODEL --dump $DUMP_FRONT --probe-dir $PROBE
```

### 5.2 动态模式（20 条，选择器决定路由）

```bash
export DUMP_FRONT=results/sel_front_dyn.jsonl
LR_CLASSES="math=gsm8k,humaneval=mbpp" RESULTS=results/sel_dyn.json \
  bash ./deploy_15.sh measure 2>&1 | tee results/sel_dyn.log
bash ./deploy_15.sh report results/sel_dyn.json
```

`LR_CLASSES` 把没部署后段的 math 并进 gsm8k 池、humaneval 并进 mbpp 池。

### 5.3 连续性测试（150 条闭环，主实验）

```bash
# 样本清单 → prompt（wash1 能连 HF；一次即可，产物在 TestSamples/cases.jsonl）
python3 examples/build_cases.py --samples TestSamples/samples.txt --out TestSamples/cases.jsonl

export DUMP_FRONT=results/cont_front.jsonl
RESULTS=results/run.json bash ./deploy_15.sh continuous 2>&1 | tee results/continuous.log
```

* 三个 task 按 gsm8k → mbpp → no_robots 轮转进同一个池，5 个前段共享；
  同时在途 = 后段通道数（5），每完成一条立刻放出下一条；最大生成 2048 token；
* **默认关闭换绑**（日志开头应有「通道二：只报警、不换绑」）；
* 约 40 分钟；每完成 5 条打一行 `[k/150] 池深 …`，最后打印汇总（时延 / TTFT / 去排队 TTFT /
  TPOT / 吞吐 / 识别 / 资源）。

看结果：

```bash
bash ./deploy_15.sh report results/run.json | sed -n '/连续性测试/,$p'   # 汇总表 + 分 task + 各节点资源
python3 examples/analyze_run.py results/run.json --csv results/run_latency.csv
python3 -m p2pmoe.deploy.lr_tool check --model $LR_MODEL --dump $DUMP_FRONT --probe-dir $PROBE
bash ./deploy_15.sh status                                                # 不应有残留 KV
```

`analyze_run.py` 的各节（可只看某几节，如 `... results/run.json latency ttft`）：

| 节 | 内容 |
|---|---|
| `latency` | 端到端时延：含排队 vs 去排队，总体 + 分 task |
| `ttft` | TTFT：原始 / 去排队 / 排队本身 |
| `throughput` | 吞吐：整体、稳态（在途保持满）、收尾、每 5 分钟一段 |
| `slowest` | TTFT 最高的 5 条：链路、TTFT 拆分、排队时占着后段的是谁 |
| `accuracy` | 选择器按 task 的正确率与混淆 |
| `rebind` | 换绑 / 报警计数、「后段排队 > TTFT」异常、写满 2048 的请求 |

### 5.4 常用变体

| 想做什么 | 怎么改 |
|---|---|
| 打开换绑（复现「换绑后输出退化」） | 命令前加 `REBIND=1` |
| 换缺专家的补救方式 | `MISS_POLICY=drop` 或 `drop_noscale`（默认 `local_topk`） |
| 改在途数 | `continuous` 后面加 `--inflight 10` |
| 改最大生成长度 | `MAX_TOKENS=512` |
| 只跑前 N 条 | `continuous` 后面加 `--requests N` |
| 按文件顺序而不是 task 轮转 | `continuous` 后面加 `--order file` |
| 原样重出现在用的方案 | 规划命令加 `--beam-legacy --avail 0.95 --theta 0.8 --rent-policy smallest_sufficient`（自 2026-10-20 起可用率 r_v 默认 1.0、不计稳定性罚，θ 默认 1.0、不打折，影子租金默认不预留（前段压到每台都放得下）；之前的方案都按 r_v=0.95、θ=0.8、租金「够装前段的最小 N 台」规划；2026-10-14 修了 beam 之后，默认参数重出的方案在 gsm8k-0、mbpp-0 的切层点上会不同；`qwen3-30b/plan_es90_fill.json` 本身没变，直接用它就不受影响） |
| 每台固定留显存给 KV | 规划命令加 `--measured-mem --kv-reserve-gb 2`：每台整块留 2GB 给 KV cache，剩下全给权重，KV 不再按 `--ctx` 逐层计入 |
| 不裁剪专家（完整模型） | 规划命令加 `--back-full`（配 `--front-full`）：后段每层也装全部专家，例见 5.10 |
| 手写分配（不经过规划器） | 写布局 JSON → `python3 examples/manual_plan.py 布局.json --out 清单.json`，例见 5.11 |
| 按轮次交错建段 | 规划命令加 `--build-interleave`：各池先各建第 1 条、再各建第 2 条（池的先后仍按 `--build-order`），例见 5.9d |
| 补位不向前看 | 规划命令加 `--fill-greedy`：补位配前段总取最快的（2026-10-27 之前的行为），可能少补一条、剩下的去分摊 |
| 放宽公共带 | 规划命令加 `--w-cap <ms>`（直接给上沿）或 `--rho-w <倍数>`（上沿 = 倍数 × 典型跳 p50，默认 1.5）；配合 `--eta` 放宽终审均匀性，例见 5.8 |
| 建段顺序 | 默认 `--build-order bottleneck`：Step 3 按公平比从低到高建后段（瓶颈池先挑节点）；2026-10-27 之前是 `constrained`（形态最少的池先建）。5.5–5.10 的清单都是 `constrained` 规划的，命令里已带上 |
| 调链长上限 | 规划命令加 `--hop-slack N`：每条段的跳数 ≤ 整数下界 hops_min（按段驻留量 ÷ 最大单机可用算）+ N，默认 1；`--hop-slack -1` 不限。求解器搜索时就剪掉超长前缀，防止 beam 交出吃光节点的长链（`--beam-legacy` 下不生效）。用现在的部署参数重出方案，开不开它结果逐字节相同 |
| 每个 task 均分 + 原始模型路径 | 规划命令加 `--quota-equal --raw-fill`：Step 2 每个 task ⌊N_work / task 数⌋ 条、照原算法建段并盲绑任意组合；剩下的节点（含备胎）建全专家的原始模型路径，专属后段全忙时溢出过去，例见 5.12 |
| 前段限定单节点 | 规划命令加 `--front-max-nodes 1`：前段（第 1–L₀ 层）只能放在一台上（doc I.2.2「单节点成段是首选」），补位与原始模型路径的前段同样适用；默认不限，只受 `--hop-slack` 约束。例见 5.13 |
| 单前段 | 规划命令加 `--single-front`（配 `--l0 2`）：只建一条前段、放一台，所有请求共用；后段照旧建并收紧，不做公共带 / 超建 / 回环裁剪，例见 5.15 |
| 按轮次分组收紧 | 规划命令加 `--balance-by-round`：所有 task 的第 1 条后段一起收紧、第 2 条一起收紧……（`--single-front` 时自动打开），例见 5.15 |
| 收紧用精确放置 | 规划命令加 `--tighten-exact`：两轮筛重建段时用 `planner/exact.py` 代替 beam（`--single-front` 时自动打开），例见 5.15 |
| 先定前段 | 规划命令加 `--front-first`（配 `--l0 2`，隐含 `--single-front`）：先按计算定前段，围绕它按组选后段首节点、按组从首节点建链，再按组收紧，例见 5.16；建链默认用精确放置，`--chain-beam` 改回 beam |
| 理论计算耗时 | 规划命令加 `--theory-times`：ms/层用拓扑 compute_capacity 的反比（A30 0.35、RTX6000 0.443、T4 0.889），`--layer-times` 的表只用来取 `--measured-mem` 的显存，日志里把实测值列在旁边对照。例见 5.17 |
| 给 no_robots 两条通道 | 重新规划（`examples/task_deploy.py … --tasks gsm8k=1,mbpp=1,no_robots=2`），换 `PLAN` 后从第 3 步的 `fetch`/`verify` 重来 |

### 5.5 对照组：严格按文档规划的方案（`qwen3-30b/plan_doc_l8.json`）

规划命令（Mac 或 wash1 上跑都行，结果逐字节相同，`tests/test_hop_cap.py` 有校验）：

```bash
python3 examples/task_deploy.py --data task3 --config qwen3-30b-a3b --phase prefill \
  --tasks gsm8k=1,mbpp=1,no_robots=1 \
  --expert-sets qwen3-30b/expert_sets_90/expert_sets_90_union.json \
  --l0 8 --front-full --reserve-gb 2.5 --hop-slack 0 \
  --rent-policy largest --endpoint-w 0 --avail 0.95 --theta 0.8 --build-order constrained \
  --save-plan qwen3-30b/plan_doc_l8.json
```

文档六步流水线 + 文档的租金规则（R_F 取内存最大者）+ 文档目标函数（无端点项）+ 新 beam；
L₀=8、前段装全部专家、预留 2.5GB、链长 = 内存所需最少台数（`--hop-slack 0`）。不开 `--fill-idle`。

| 角色 | 节点 |
|---|---|
| 前段（层 1–8） | N03、N04、N01、N02；N12 备胎（不进池） |
| gsm8k | N07(9–24)→N08(25–48)、N05(9–24)→N06(25–48) |
| mbpp | N10(9–23)→N11(24–48) |
| no_robots | N09(9–14)→N13(15–48) |
| 不用 | N14、N15 |

4 条通道（gsm8k 2、mbpp 1、no_robots 1）；静态配对组合 p50 19–49ms（no_robots 那条跨 4.x 子网）。
驻留集与现方案相同，`PROFILE` 不用换。

**权重就地替换**（磁盘放不下新旧两份）：`REPLACE=1 bash ./deploy_15.sh fetch` 让每台先看
`$WEIGHTS` 里已有的是不是新清单要的那份张量 —— 是就跳过，不是就**先删旧权重**
（`*.safetensors` / 索引；config 与 tokenizer 保留）再下新的。从现方案换到对照组：

| 节点 | 现方案 → 对照组 | 动作 |
|---|---|---|
| N01、N12 | 前段 → 前段（N12 改为备胎，张量相同） | 跳过 |
| N09 | no_robots 9–14 → no_robots 9–14 | 跳过 |
| N02、N03、N04 | gsm8k 后段 → 前段 | 删旧下新 |
| N05、N06、N07、N08 | 各自 → gsm8k 9–24 / 25–48 | 删旧下新 |
| N10、N11 | 前段 → mbpp 9–23 / 24–48 | 删旧下新 |
| N13 | mbpp 9–35 → no_robots 15–48 | 删旧下新 |
| N14、N15 | 对照组不用 | 不动（hosts 里去掉了） |

合计下载约 144GB。删掉的现方案权重，切回去时要重新下（同一条命令，见下）。

```bash
# 0) Mac → wash1 推代码（同第 1 步的 rsync，带上 plan_doc_l8.json）

# 1) 环境
cd /home/ubuntu/p2p-framework && conda activate moe && source qwen3-30b/env.sh
export ADVERTISE=192.168.3.4 && mkdir -p results

# 2) 先用原来的 15 台清单把现在的 agent 全停掉（N14/N15 上的也停，否则一直占显存）
bash ./deploy_15.sh stop

# 3) 切到对照组：清单 + 只含 13 台的 hosts（WEIGHTS 不变，就地替换）
export PLAN=qwen3-30b/plan_doc_l8.json
awk '$1!="N14" && $1!="N15"' task/hosts.txt > task/hosts_doc.txt
export HOSTS=task/hosts_doc.txt

# 4) 推代码与清单 → 各节点就地换权重 → 核对
bash ./deploy_15.sh sync
REPLACE=1 bash ./deploy_15.sh fetch   # 会问 [y/N]；当初走 HF 镜像就先 export HF_ENDPOINT=…
bash ./deploy_15.sh verify            # 13 台都 ✓；某台失败就再跑一次上一条（已完成的会跳过）

# 5) 起 agent
bash ./deploy_15.sh start
bash ./deploy_15.sh status            # 13/13，无残留 KV

# 6) 连续性测试：与 5.3 完全同样的设置（150 条、三 task 轮转、共享池、闭环、2048）
export DUMP_FRONT=results/cont_front_doc.jsonl
RESULTS=results/run_doc.json bash ./deploy_15.sh continuous 2>&1 | tee results/continuous_doc.log

# 7) 看结果
bash ./deploy_15.sh report results/run_doc.json | sed -n '/连续性测试/,$p'
python3 examples/analyze_run.py results/run_doc.json --csv results/run_doc_latency.csv
```

* 同时在途 = 4（后段通道数）。三个 task 各 50 条、mbpp 与 no_robots 各只有 1 条通道，开头会提示
  「每条通道分到的请求数不均」—— 这是该方案的容量结构，不是故障；预计约 1 小时；
* 资源表的平均 GPU 利用率含备胎 N12（一直空闲），逐节点表里能单独看；
* `REPLACE=1` 下某台中途下载失败时，那台此时没有可用权重 —— 重跑同一条命令即可；
* **切回现方案**（同样就地替换，N01、N09、N12、N14、N15 会跳过）：

```bash
bash ./deploy_15.sh stop                       # 此时 HOSTS 还是 13 台
# 新开终端：PLAN 回到 plan_es90_fill.json，HOSTS 回到默认 15 台
cd /home/ubuntu/p2p-framework && conda activate moe && source qwen3-30b/env.sh
export ADVERTISE=192.168.3.4
REPLACE=1 bash ./deploy_15.sh fetch
bash ./deploy_15.sh verify
bash ./deploy_15.sh start
```

### 5.6 逐层计时：用实测的每层耗时规划

拓扑文件里只有抽象的 `compute_capacity`，规划器按它缩放成「最快 0.35ms/层」的相对值，
比实测（TPOT 130–145ms / 48 层 ≈ 2–3ms/层）低了 7 倍左右。`layerbench` 让每台用
与服务完全相同的代码路径（同 dtype、同 miss 策略、同 decode 快路径）量 decode（逐 token）
与 prefill 的每层耗时。

**在控制机上一条命令，经 agent 量完全部节点并汇总**（不用 ssh 到各台、不用停 agent）：

```bash
bash ./deploy_15.sh status           # agent 都在线
bash ./deploy_15.sh layerbench       # 各台并行；逐台 → results/layer_bench/<节点>.json
                                     #          汇总 → results/layer_bench.json（逐节点 + 按 GPU 型号 mean ± std）
```

* agent **装着权重**（跑过 measure / continuous 之后）：直接量它在清单里分到的那几层、那些专家
  —— 节点上的真实模型参数，不另占显存（只多计时用的激活与 KV）；
* agent **没装**（刚 start）：按 `$WEIGHTS/config.json` 的形状建随机权重量，需要约 5GB 空闲显存；
* `LB_MODE=loaded` / `LB_MODE=synthetic` 强制其一；有在途请求的节点会拒绝（等测试跑完再量）；
* 默认把 prompt / 上下文长度从 128 扫到 1024（步长 128），再加 2048（规划按 ctx_max=2048 留显存，要它的激活峰值）；
  长度写法：逗号分隔，区间写 `起:止:步长`，如 `LB_PREFILL=128:1024:64` 或 `LB_CTX=256,512,1024`；
* 其他可调：`LB_LAYERS=4 LB_STEPS=32 LB_TIMEOUT=900 LB_KEY=decode@1024`；
* `LB_VIA=ssh` 退回 ssh 到每台跑的老办法（agent 必须先停）。

规划时用实测值（缺测的节点退到同型号均值，再退到按算力比例外推）：

```bash
python3 examples/task_deploy.py ...原参数... \
    --layer-times results/layer_bench.json --layer-time-key decode@1024
```

* `--layer-time-key`：`decode@<上下文长度>` 是生成阶段每 token 每层的耗时（默认 1024），
  `prefill@<token 数>` 是一次喂 T 个 token 时整层的耗时；
* 规划日志开头会打印每台「拓扑推算值 → 实测值」和来源；
* 不加 `--layer-times` 时规划结果与以前逐字节相同。

**显存也用实测**（同一份 `layer_bench.json`，`layerbench` 同时量了显存）：

```bash
python3 examples/task_deploy.py ...原参数... \
    --layer-times results/layer_bench.json --measured-mem --mem-margin-gb 0.5
```

| 量 | 怎么量 | 规划怎么用 |
|---|---|---|
| 整卡容量 | `torch.cuda.mem_get_info` | 代替拓扑里的标称值（N13 这类卡标 24GB，实际 23.59GB） |
| CUDA context | 整卡已用 − PyTorch 占用 | 从容量里扣掉 |
| prefill 激活峰值 | 一次 prefill 时分配器峰值 − 权重 − 这条请求的 KV（取量过的最长 T） | 从容量里扣掉 |
| 分配器开销比 | 权重占用（含碎片、取整）÷ 张量本身大小 | 可用量 ÷ 它，相当于把每层权重按实测放大 |
| 每层权重、每 token 每层 KV | 张量字节数（已加载模式逐层给出） | 与规划器公式对账（日志里打印） |
| 词嵌入 / 输出头 | vocab × d_model × dtype | 计入前段第一台 / 后段最后一台（以前漏算的 0.62GB） |

每台可用量 = (容量 − context − 激活峰值 − 余量) ÷ 开销比；`--reserve-gb` 此时不用。
按实际请求长度留显存：`--ctx` 是 KV 按多长的上下文（prompt + 生成）算，`--mem-prefill-tokens` 是激活峰值按多长的
prompt 取（实测里不小于它的最短一档）。例如 prompt 512、生成 512：`--ctx 1024 --mem-prefill-tokens 512`。缺测的量退到同型号均值、
再退到全体中位数；容量没测且没有同型号可借时保留标称值。

### 5.7 文档六步 + 实测输入的方案（`qwen3-30b/plan_doc_v2.json`）

规划命令（Mac 或 wash1 上跑都行，结果逐字节相同，`tests/test_hop_cap.py::test_doc_v2_plan_file_is_reproducible` 有校验）：

```bash
python3 examples/task_deploy.py --data task3 --config qwen3-30b-a3b --phase prefill \
  --tasks gsm8k=1,mbpp=1,no_robots=1 \
  --expert-sets qwen3-30b/expert_sets_90/expert_sets_90_union.json \
  --l0 8 --front-full \
  --layer-times qwen3-30b/layer_bench_wash1.json --measured-mem \
  --mem-prefill-tokens 512 --mem-margin-gb 0 --kv-reserve-gb 2 --endpoint-w 0 --build-order constrained \
  --save-plan qwen3-30b/plan_doc_v2.json
```

* 文档六步流水线，只用文档里有的东西：不开 `--exhaustive-short`、无端点项、不开 `--fill-idle`；
  超参全取默认（w_cap = 1.5 × 典型跳 ≈ 4.6ms、η = 15%、β = 1.3、κ_over = 0.3）；
  r_v = 1、θ = 1、不留影子租金；链长上限 `--hop-slack 1`（默认，这份方案里不起作用）；
* 每层耗时与显存用 `layerbench` 实测（`qwen3-30b/layer_bench_wash1.json` = wash1 上的 `results/layer_bench.json`）；
  激活峰值按 512 token prompt 取；**每台固定留 2GB 给 KV cache，剩下的全给权重**（`--kv-reserve-gb 2`，
  KV 不再按上下文逐层算进段驻留；`--mem-margin-gb 0` 不另留余量）。
  权重可用量 = 容量 − context − 激活峰值 − 2GB：A30 22.93GB（N01 22.63）、RTX6000 21.28GB、T4 13.42GB；
  KV 每 token 每层 2048B，2GB 在放 30 层的节点上约容 3.2 万 token，单条请求（≤ 512 + 2048）只用约 0.16GB；
* 规划日志：`qwen3-30b/plan_doc_v2.log`。

| 角色 | 节点 |
|---|---|
| 前段（层 1–8，全专家） | N07、N08、N09、N10 |
| gsm8k | N04(9–36)→N01(37–48) |
| mbpp | N02(9–37)→N03(38–48)、N05(9–36)→N06(37–48) |
| no_robots | N12(9–29)→N11(30–48) |
| 不用 | N13、N14、N15（到其他组 > 11ms，进不了 4.6ms 的公共带） |

4 条通道（gsm8k 1、mbpp 2、no_robots 1），静态组合 p50 114–129ms，终审最坏相对极差 9.1%。
权重最满的是 N02（mbpp 9–37 层 22.70GB / 可用 22.93GB）；与余量 0.5GB、KV 按 2560 计的版本相比，只有 N12 少放两层给 N11。

权重就地替换：从 `plan_doc_l8.json` 换过来 12 台全部删旧下新（约 159GB）；从 `plan_es90_fill.json`
换过来只有 N10 跳过（约 149GB）。

```bash
# 0) Mac → wash1 推代码（同第 1 步的 rsync，带上 plan_doc_v2.json）

# 1) 环境（新终端）
cd /home/ubuntu/p2p-framework && conda activate moe && source qwen3-30b/env.sh
export ADVERTISE=192.168.3.4 && mkdir -p results

# 2) 用完整的 15 台清单把现在所有 agent 停掉（不论之前跑的是哪份方案）
HOSTS=task/hosts.txt bash ./deploy_15.sh stop

# 3) 切到新方案：清单 + 只含 12 台的 hosts
export PLAN=qwen3-30b/plan_doc_v2.json
awk '$1!="N13" && $1!="N14" && $1!="N15"' task/hosts.txt > task/hosts_v2.txt
export HOSTS=task/hosts_v2.txt

# 4) 推代码与清单 → 各节点就地换权重 → 核对
bash ./deploy_15.sh sync
REPLACE=1 bash ./deploy_15.sh fetch   # 会问 [y/N]；走 HF 镜像就先 export HF_ENDPOINT=…
bash ./deploy_15.sh verify            # 12 台都 ✓；失败的再跑一次上一条（已完成的会跳过）

# 5) 起 agent
bash ./deploy_15.sh start
bash ./deploy_15.sh status            # 12/12，无残留 KV

# 6) 连续性测试：与 5.3 同样的设置（150 条、三 task 轮转、前段共享池、闭环、最大生成 2048）
export DUMP_FRONT=results/cont_front_v2.jsonl
RESULTS=results/run_v2.json bash ./deploy_15.sh continuous 2>&1 | tee results/continuous_v2.log

# 7) 看结果
bash ./deploy_15.sh report results/run_v2.json | sed -n '/连续性测试/,$p'
python3 examples/analyze_run.py results/run_v2.json --csv results/run_v2_latency.csv
python3 -m p2pmoe.deploy.lr_tool check --model $LR_MODEL --dump $DUMP_FRONT --probe-dir $PROBE
bash ./deploy_15.sh status            # 不应有残留 KV
```

* 同时在途 = 4（后段通道数）；gsm8k、no_robots 各 1 条通道、各 50 条请求，开头会提示「每条通道分到的
  请求数不均」—— 是方案的容量结构，不是故障；
* 每台权重之外留着 2GB 给 KV（N12 是 13.15GB 权重 / 15.66GB 整卡）；仍有节点 OOM 就把 `--kv-reserve-gb` 或 `--mem-margin-gb` 调大重出，再从第 4 步来；
* 切回其他方案：stop → 换 `PLAN`/`HOSTS` → `REPLACE=1 bash ./deploy_15.sh fetch` → verify → start。

### 5.7b 瓶颈池优先建段的 4 通道方案（`qwen3-30b/plan_doc_v2_bn.json`）

与 5.7 参数完全相同，只是建段顺序用新的默认 `bottleneck`（公平比最低的池先建：no_robots → gsm8k → mbpp）。
`tests/test_build_order.py::test_bn_plan_file_is_reproducible` 校验逐字节重出：

```bash
python3 examples/task_deploy.py --data task3 --config qwen3-30b-a3b --phase prefill \
  --tasks gsm8k=1,mbpp=1,no_robots=1 \
  --expert-sets qwen3-30b/expert_sets_90/expert_sets_90_union.json \
  --l0 8 --front-full \
  --layer-times qwen3-30b/layer_bench_wash1.json --measured-mem \
  --mem-prefill-tokens 512 --mem-margin-gb 0 --kv-reserve-gb 2 --endpoint-w 0 \
  --save-plan qwen3-30b/plan_doc_v2_bn.json
```

| 角色 | 5.7（constrained） | 5.7b（bottleneck） |
|---|---|---|
| 前段 | N07、N08、N09、N10 | N02、N03、N09、N10 |
| no_robots | N12(T4)+N11 | **N07+N08**（A30） |
| gsm8k | N04+N01 | N04+N01 |
| mbpp | N02+N03、N05+N06 | N05+N06、**N12(T4)+N11** |
| 各 task 组合 p50 区间 | no_robots 128–131ms | no_robots 111.5–118.9、gsm8k 113.6–123.0、mbpp 118.5–132.3ms |
| 终审最坏相对极差 | 9.1% | 11.4%（mbpp） |

慢的那组节点（T4 N12 + RTX N11）从 no_robots 挪到了 mbpp 的第 2 条：no_robots 只有 1 条通道，
它变快对排队最有利；mbpp 有 2 条，慢的那条只承担一半请求。部署：`source qwen3-30b/use_plan.sh doc_v2_bn`，
其余同 5.7（从 5.7 的权重换过来动 6 台，约 76GB）。

### 5.7c 5.7b 的 4 条主通道 + 剩余节点补位（`qwen3-30b/plan_doc_v2_bn_fill.json`）

5.7b 的命令加 `--fill-idle`：主流程（瓶颈池优先建段）照旧出 4 条、12 台原样不动，剩下的 N13、N14、N15
交给补位再建一条通道（`tests/test_build_order.py::test_bn_fill_plan_file_is_reproducible` 校验）：

```bash
python3 examples/task_deploy.py --data task3 --config qwen3-30b-a3b --phase prefill \
  --tasks gsm8k=1,mbpp=1,no_robots=1 \
  --expert-sets qwen3-30b/expert_sets_90/expert_sets_90_union.json \
  --l0 8 --front-full \
  --layer-times qwen3-30b/layer_bench_wash1.json --measured-mem \
  --mem-prefill-tokens 512 --mem-margin-gb 0 --kv-reserve-gb 2 --endpoint-w 0 \
  --fill-idle --save-plan qwen3-30b/plan_doc_v2_bn_fill.json
```

补位按公平比给最堵的池加一条：no_robots[N15(9–29)+N14(30–48)]（两台 T4），前段 F[N13]（RTX）。
配额变成 {gsm8k 1, mbpp 2, no_robots 2}，5 条通道、15 台全上岗。

补位段不过公共带：C 组到 A/B 组单程 11.5–13.5ms，盲绑下 N13 前段接 A 组后段、或 A/B 组前段接
N15+N14 后段，每 token 都要多付约 23ms。全网格组合 p50：

| task | 范围 | 相对极差 |
|---|---|---|
| no_robots | 111.5–156.5ms | 34.9% |
| mbpp | 118.5–155.8ms | 29.0% |
| gsm8k | 113.6–139.2ms | 19.9% |

核心 4×4 网格仍是 11.4%（通过）；全网格 34.9% 是补位的已知代价（日志里 16 条公共带警告）。
部署：`source qwen3-30b/use_plan.sh doc_v2_bn_fill`（15 台），其余同 5.7。

### 5.8 五条通道的方案（`qwen3-30b/plan_doc_v2_5ch.json`）

5.7 的参数，只放宽两个阈值，让五条后段都能留住（`tests/test_hop_cap.py::test_doc_v2_5ch_plan_file_is_reproducible` 校验逐字节重出）：

```bash
python3 examples/task_deploy.py --data task3 --config qwen3-30b-a3b --phase prefill \
  --tasks gsm8k=1,mbpp=1,no_robots=1 \
  --expert-sets qwen3-30b/expert_sets_90/expert_sets_90_union.json \
  --l0 8 --front-full \
  --layer-times qwen3-30b/layer_bench_wash1.json --measured-mem \
  --mem-prefill-tokens 512 --mem-margin-gb 0 --kv-reserve-gb 2 --endpoint-w 0 --build-order constrained \
  --w-cap 14 --eta 0.21 \
  --save-plan qwen3-30b/plan_doc_v2_5ch.json
```

| 参数 | 5.7 | 5.8 | 为什么 |
|---|---|---|---|
| w_cap（公共带上沿） | 1.5 × 典型跳 ≈ 4.6ms | **14ms**（≈ 4.5 × 典型跳） | C 组（N13–N15）到后段入口要 11.5–13.5ms，最远 N15→N12 13.52ms；上沿 ≥ 13.55 才能进带 |
| η（均匀性阈值） | 15% | **21%** | 带进 C 组前段后，终审最坏相对极差 20.0%（no_robots：128.0–154.8ms ÷ 网格中位 134.3ms） |

只调 η 没用：上沿被 w_cap 卡着，η 再大公共带也只有 [0, 4.6]。规划日志：`qwen3-30b/plan_doc_v2_5ch.log`。

| 角色 | 节点 |
|---|---|
| 前段（层 1–8，全专家） | N09、N10、N13、N14、N15 |
| gsm8k | N07(9–37)→N08(38–48)、N04(9–36)→N01(37–48) |
| mbpp | N02(9–37)→N03(38–48)、N05(9–36)→N06(37–48) |
| no_robots | N12(9–29)→N11(30–48) |

5 条通道、15 台全上岗；静态配对：

| 通道 | 前段 → 后段 | 组合 p50 |
|---|---|---|
| gsm8k-0 | N10 → N07+N08 | 117.7ms |
| gsm8k-1 | N09 → N04+N01 | 123.0ms |
| mbpp-0 | N15 → N02+N03 | 139.8ms |
| mbpp-1 | N14 → N05+N06 | 142.9ms |
| no_robots-0 | N13 → N12+N11 | 155.1ms |

代价：C 组前段到任何后段入口、以及回环都要 ≈ 11.5–13.5ms，挂在它们后面的通道每 token 多 ≈ 23ms；
N14、N15 是 T4，prefill 比 A30 慢（TTFT 更长）。

权重就地替换：从 5.7 的 `plan_doc_v2.json` 换过来只动 N07、N08（前段 → gsm8k 后段）和 N13–N15（→ 前段），约 61GB；
从 `plan_es90_fill.json` 约 180GB（N10、N15 跳过）；从 `plan_doc_l8.json` 15 台全换，约 200GB。

部署与实验同 5.7，只改清单、hosts 用完整的 15 台、结果文件换名：

```bash
HOSTS=task/hosts.txt bash ./deploy_15.sh stop
export PLAN=qwen3-30b/plan_doc_v2_5ch.json
export HOSTS=task/hosts.txt
bash ./deploy_15.sh sync
REPLACE=1 bash ./deploy_15.sh fetch
bash ./deploy_15.sh verify            # 15 台都 ✓
bash ./deploy_15.sh start && bash ./deploy_15.sh status   # 15/15

export DUMP_FRONT=results/cont_front_v2_5ch.jsonl
RESULTS=results/run_v2_5ch.json bash ./deploy_15.sh continuous 2>&1 | tee results/continuous_v2_5ch.log
bash ./deploy_15.sh report results/run_v2_5ch.json | sed -n '/连续性测试/,$p'
python3 examples/analyze_run.py results/run_v2_5ch.json --csv results/run_v2_5ch_latency.csv
```

同时在途 = 5；gsm8k、mbpp 各 2 条通道，no_robots 1 条（仍会提示请求数不均）。

### 5.9 按 vLLM 实测规划的方案（`qwen3-30b/plan_vllm.json`）

输入是 `vllm/summary_by_gpu.json`（按 GPU 型号汇总的 vLLM 逐层实测）。只取 batch 1、全部 128 个专家，
按 512 prompt + 512 生成换算成规划器认的表，再规划：

```bash
# 1) vLLM 汇总 → layer_bench 格式（逐节点；结果已在仓库里）
python3 -m p2pmoe.deploy.vllm_summary vllm/summary_by_gpu.json \
  --prompt 512 --gen 512 --batch 1 --out qwen3-30b/layer_bench_vllm.json

# 2) 规划（tests/test_hop_cap.py::test_vllm_plan_file_is_reproducible 校验逐字节重出）
python3 examples/task_deploy.py --data task3 --config qwen3-30b-a3b --phase prefill \
  --tasks gsm8k=1,mbpp=1,no_robots=1 \
  --expert-sets qwen3-30b/expert_sets_90/expert_sets_90_union.json \
  --l0 8 --front-full \
  --layer-times qwen3-30b/layer_bench_vllm.json --layer-time-key req@p512g512 \
  --measured-mem --mem-margin-gb 0 --kv-reserve-gb 2 --endpoint-w 0 --build-order constrained \
  --theta 0.8 --eta 0.4 \
  --save-plan qwen3-30b/plan_vllm.json
```

**每层耗时**（`req@p512g512` = (prefill 512 + 512 × decode 平均) ÷ 512，一条请求摊到每个生成 token；
decode 平均取上下文 512→1023 的插值；逐节点按 `ms_per_layer_nodes` 与型号均值之比换算）：

| GPU | decode（ms/层） | prefill 512（ms/层） | req（ms/层） | 每步固定开销（ms/token，未进规划） |
|---|---|---|---|---|
| A30 | 0.19–0.22 | 2.6–3.0 | 0.19–0.22 | 1.40 |
| RTX6000 | 1.69–1.71 | 56.9–57.6 | 1.80–1.82 | 1.95 |
| T4 | 3.01–3.06 | 95.9–97.3 | 3.20–3.25 | 3.43 |

**显存**：整卡 − vLLM 非权重固定开销（非 KV 固定部分减去词嵌入与输出头：A30 1.69、RTX 1.56、T4 1.49GB）
− 2GB KV，÷ 每层实占/权重比（1.004–1.015）→ 权重可用 A30 21.53、RTX6000 19.96、T4 11.99GB。

**为什么 θ=0.8、η=0.4**：vLLM 下 A30 比 Turing 卡快约 10 倍，A30 上整段后段只要 8–9ms，
网络一跳（A30 组 ↔ RTX 组 3.1ms）与前段落在哪种卡上（A30 1.7 / RTX 14.5 / T4 26ms）成了延迟的主体；
默认的 η=15%（≈ 2.4ms）在这份拓扑上凑不出公共带，规划失败。θ=0.8 建 4 条后段（全放 A30），
前段落到 B 组，终审最坏相对极差 38.4%，η 放到 40% 通过。

| 角色 | 节点 |
|---|---|
| 前段（层 1–8，全专家） | N09、N10、N11（RTX6000）、N12（T4） |
| gsm8k | N07(9–32)→N06(33–48)、N05(9–32)→N03(33–48) |
| mbpp | N01(9–33)→N04(34–48) |
| no_robots | N08(9–34)→N02(35–48) |
| 不用 | N13、N14、N15 |

每 token 组合 p50：前段是 RTX 时约 28.5–29.3ms，是 T4（N12）时约 42ms（盲绑下任何前段都可能接任何后段）。
以上不含 vLLM 每步的固定开销（每台约 1.4–3.4ms/token，一条通道 3 台约 +5–7ms）。

用 vLLM 起这份方案时，`gpu_memory_utilization` 至少要到（权重 + 首尾表 + 固定开销 + 2GB KV）÷ 整卡：
N01 0.92、N12 0.91、N05/N07 0.89、N08 0.79，其余 ≤ 0.66；统一设 0.93 即可（默认 0.9 时 N01、N12 的 KV 不到 2GB）。

### 5.9b vLLM 实测，θ=1 选完再补位（`qwen3-30b/plan_vllm_bn_fill.json`）

5.9 的输入，但 θ 回到 1、超参全取默认（η=15%、w_cap≈4.6ms）、瓶颈池优先建段；主流程选完后
用剩下的节点补位（`--fill-idle`，补位带前瞻）。`tests/test_hop_cap.py::test_vllm_bn_fill_plan_file_is_reproducible`
校验逐字节重出：

```bash
python3 examples/task_deploy.py --data task3 --config qwen3-30b-a3b --phase prefill \
  --tasks gsm8k=1,mbpp=1,no_robots=1 \
  --expert-sets qwen3-30b/expert_sets_90/expert_sets_90_union.json \
  --l0 8 --front-full \
  --layer-times qwen3-30b/layer_bench_vllm.json --layer-time-key req@p512g512 \
  --measured-mem --mem-margin-gb 0 --kv-reserve-gb 2 --endpoint-w 0 \
  --fill-idle --save-plan qwen3-30b/plan_vllm_bn_fill.json
```

**主流程**（θ=1 → 配额 {gsm8k 2, mbpp 2, no_robots 1}；建段 no_robots → mbpp → gsm8k；公共带凑不够，
Step 4 先后降掉 mbpp、gsm8k 各一条）：3 条通道，前段全是 A30（N06、N07、N08），终审 1.4%。

| 通道 | 前段 → 后段 | 组合 p50 |
|---|---|---|
| no_robots | N07 → N01(9–37)+N04 | 11.9ms |
| mbpp | N06 → N05(9–33)+N03 | 12.7ms |
| gsm8k | N08 → N02(9–32)+N11(RTX) | 43.7ms |

**补位**（空闲 N09、N10、N12、N13、N14、N15，一前一后成对地加）：

| 通道 | 前段 → 后段 | 组合 p50 |
|---|---|---|
| gsm8k-1 | N12(T4) → N09(9–33)+N10（RTX×2） | 98.1ms |
| no_robots-1 | N14(T4) → N13(9–39,RTX)+N15(40–48,T4) | 110.7ms |

补位配前段时**向前看一步**（`planner/fill.py` 的 `pick_front`，2026-10-27 起默认）：最快的前段 N13
会让剩下的三台 T4 凑不出下一条后段，于是改用 N12（T4），把 N13 留给第二条后段 —— 补出 2 条、
15 台全上岗、没有分摊。不向前看（`--fill-greedy`，旧行为）只补出 1 条，剩下 3 台挂到那条后段上
分摊，它变成 5 台 4 跳、每 token 约 169ms。

盲绑全网格组合 p50（vLLM 下核心段只要 12ms，补位段与 T4 前段的差距被放大）：

| task | 核心 3 条 | 含补位 |
|---|---|---|
| no_robots | 11.9–12.0ms | 11.9–137.3ms |
| mbpp | 12.6–12.7ms | 12.6–57.5ms |
| gsm8k | 43.5–43.7ms | 43.5–125.4ms |

核心网格 1.4%（通过）；全网格 218% 是补位的已知代价（日志里 21 条公共带警告）：接到 T4 前段
（N12/N14）或补位后段（N09+N10、N13+N15）的组合每 token 要 40–137ms。

用真 vLLM 跑时 `gpu_memory_utilization` 至少：N13 0.98、N09 0.98、N05 0.92、N12/N14 0.91（按每台留 2GB KV；
单条请求实际只要 0.1–0.2GB，统一设 0.95 时 N13、N09 仍各有约 1.3GB KV）。

部署：`source qwen3-30b/use_plan.sh vllm_bn_fill`（15 台），其余同 5.9。与 `plan_vllm.json` 只有 N12
一台相同，其余 14 台都要换，约 183GB。

### 5.9c 在 `plan_vllm.json` 上再补一条（`qwen3-30b/plan_vllm_fill.json`）

5.9 的命令原样加 `--fill-idle`（建段顺序钉在 `constrained`，与 5.9 一致）：`plan_vllm.json` 的 12 台、4 条通道
一点不变，只把空闲的 N13–N15 补成第 5 条（`tests/test_hop_cap.py::test_vllm_fill_plan_keeps_plan_vllm` 校验）：

```bash
python3 examples/task_deploy.py --data task3 --config qwen3-30b-a3b --phase prefill \
  --tasks gsm8k=1,mbpp=1,no_robots=1 \
  --expert-sets qwen3-30b/expert_sets_90/expert_sets_90_union.json \
  --l0 8 --front-full \
  --layer-times qwen3-30b/layer_bench_vllm.json --layer-time-key req@p512g512 \
  --measured-mem --mem-margin-gb 0 --kv-reserve-gb 2 --endpoint-w 0 --build-order constrained \
  --theta 0.8 --eta 0.4 --fill-idle \
  --save-plan qwen3-30b/plan_vllm_fill.json
```

| 通道 | 前段 → 后段 | 每 token |
|---|---|---|
| ch0 gsm8k | N10 → N07+N06 | 29.3ms |
| ch1 gsm8k | N11 → N05+N03 | 29.2ms |
| ch2 mbpp | N09 → N01+N04 | 28.5ms |
| ch3 no_robots | N12(T4) → N08+N02 | 42.1ms |
| **ch4 no_robots（补位）** | **N14(T4) → N13(9–39，RTX)+N15(40–48，T4)** | **110.7ms** |

补位给最堵的 no_robots（公平比 0.75）：三台只够一条（N13+N14 可装 gsm8k/no_robots，装不下 mbpp）。
盲绑下 N09–N12 前段接 N13+N15 后段要 126–137ms，前段 N14 接 A30 后段要 57–59ms；全网格各 task 范围
gsm8k 29.1–57.6、mbpp 28.5–56.9、no_robots 30.9–137.3ms（全网格相对极差 268%，核心 4×4 仍是 38.4%）。

部署：`source qwen3-30b/use_plan.sh vllm_fill`（15 台）。节点上已经是 `plan_vllm.json` 的权重时，
`REPLACE=1 fetch` 只动 N13、N14、N15（约 35GB），其余 12 台跳过。用真 vLLM 跑时 N13 的
`gpu_memory_utilization` 要 0.98（留 2GB KV）；设 0.95 也还有约 1.2GB KV。

### 5.9d 5.9c 改成按轮次交错建段（`qwen3-30b/plan_vllm_rr_fill.json`）

5.9c 的命令加 `--build-interleave`：池的先后不变（mbpp → gsm8k → no_robots），但每个池先各建第 1 条、
再建第 2 条，于是顺序是 mbpp0 → gsm8k0 → **no_robots0 → gsm8k1**（原来 gsm8k1 先于 no_robots0）。
`tests/test_hop_cap.py::test_vllm_rr_fill_plan_file_is_reproducible` 校验：

```bash
python3 examples/task_deploy.py --data task3 --config qwen3-30b-a3b --phase prefill \
  --tasks gsm8k=1,mbpp=1,no_robots=1 \
  --expert-sets qwen3-30b/expert_sets_90/expert_sets_90_union.json \
  --l0 8 --front-full \
  --layer-times qwen3-30b/layer_bench_vllm.json --layer-time-key req@p512g512 \
  --measured-mem --mem-margin-gb 0 --kv-reserve-gb 2 --endpoint-w 0 --build-order constrained \
  --build-interleave --theta 0.8 --eta 0.45 --fill-idle \
  --save-plan qwen3-30b/plan_vllm_rr_fill.json
```

| 后段 | 5.9c（plan_vllm_fill） | 5.9d（plan_vllm_rr_fill） |
|---|---|---|
| gsm8k0 | N07+N06 | N07+N06 |
| gsm8k1 | N05(9–32)+N03（A1 内，跳 0.04ms） | **N08(9–34)+N02（A2→A1，跳 1.23ms）** |
| mbpp0 | N01+N04 | N01+N04 |
| no_robots0 | N08+N02 | **N05(9–25)+N03（A1 内）** |
| no_robots1（补位） | N13+N15 | N13+N15 |

前段不变（N09–N12、补位 N14）。核心 4×4 终审 43.9%，超过 5.9 的 η=0.40，所以 η 放到 0.45（分配与 0.40 下完全相同）。
全网格各 task：gsm8k 29.2–59.3、mbpp 28.5–56.9、no_robots 29.1–137.3ms。

部署：`source qwen3-30b/use_plan.sh vllm_rr_fill`（15 台）。从 5.9c 的权重换过来只动 N02、N03、N05、N08（约 56GB）。

### 5.10 不裁剪专家的对照组：完整模型（`qwen3-30b/plan_full.json`）

前段、后段每层都装全部 128 个专家（`--front-full --back-full`），其余与 5.7 相同；后段不再按 task 裁剪，
三个 task 的后段内容一模一样，用来和裁剪方案对比（缺专家的影响、显存/节点的代价）。
规划命令（`tests/test_hop_cap.py::test_full_model_plan_file_is_reproducible` 校验逐字节重出）：

```bash
python3 examples/task_deploy.py --data task3 --config qwen3-30b-a3b --phase prefill \
  --tasks gsm8k=1,mbpp=1,no_robots=1 \
  --expert-sets qwen3-30b/expert_sets_90/expert_sets_90_union.json \
  --l0 8 --front-full --back-full \
  --layer-times qwen3-30b/layer_bench_wash1.json --measured-mem \
  --mem-prefill-tokens 512 --mem-margin-gb 0 --kv-reserve-gb 2 --endpoint-w 0 --build-order constrained \
  --hop-slack 0 \
  --save-plan qwen3-30b/plan_full.json
```

* 超参全取默认（w_cap ≈ 4.6ms、η = 15%）；
* 一条完整后段 40 层 × 1.246GB + 输出头 ≈ 50.5GB，单台最多 22.93GB → 至少 3 台；`--hop-slack 0` 让每条后段
  正好 3 台（默认 1 时 beam 会多拉一台只放 1 层，白多一跳）；
* 每条通道 = 1 台前段 + 3 台后段，15 台最多 3 条 → 三个 task 各 1 条；
* 规划日志：`qwen3-30b/plan_full.log`。

| 角色 | 节点 |
|---|---|
| 前段（层 1–8） | N09、N10（RTX6000）、N12（T4） |
| gsm8k | N07(9–26)→N06(27–31)→N08(32–48) |
| mbpp | N04(9–25)→N02(26–31)→N01(32–48) |
| no_robots | N03(9–23)→N05(24–34)→N11(35–48) |
| 不用 | N13、N14、N15 |

3 条通道，静态组合 p50 118.7 / 123.0 / 129.3ms，终审最坏相对极差 1.5%（η = 15% 通过）。
权重共 179GB：从 `plan_doc_l8.json` 换过来只有 N12 跳过（约 169GB），从 `plan_doc_v2.json` 只有 N09、N10 跳过（约 160GB）。

部署与实验同 5.7（hosts 也是去掉 N13–N15 的 12 台），只换清单和结果文件名：

```bash
HOSTS=task/hosts.txt bash ./deploy_15.sh stop
export PLAN=qwen3-30b/plan_full.json
awk '$1!="N13" && $1!="N14" && $1!="N15"' task/hosts.txt > task/hosts_v2.txt
export HOSTS=task/hosts_v2.txt
bash ./deploy_15.sh sync
REPLACE=1 bash ./deploy_15.sh fetch
bash ./deploy_15.sh verify            # 12 台都 ✓
bash ./deploy_15.sh start && bash ./deploy_15.sh status

export DUMP_FRONT=results/cont_front_full.jsonl
RESULTS=results/run_full.json bash ./deploy_15.sh continuous 2>&1 | tee results/continuous_full.log
bash ./deploy_15.sh report results/run_full.json | sed -n '/连续性测试/,$p'
python3 examples/analyze_run.py results/run_full.json --csv results/run_full_latency.csv
```

* 同时在途 = 3（每个 task 1 条通道），150 条请求预计比 4 条通道的方案更久；
* 后段不缺专家，`MISS_POLICY` 不起作用；选择器照常识别并按 task 派发（三条后段内容相同，识别错了也不影响输出）；
* `PROFILE` 不用换（只给前段的识别器用）。

### 5.11 手写分配：完整模型两条链（`qwen3-30b/plan_full_2ch.json`）

不经过规划器，按给定的分配直接出清单。分配（原表层号是 0-based 切片，`0:8` = 第 1–8 层）：

| 层 | 链 A | 链 B |
|---|---|---|
| 1–8（前段） | N01 | N11 |
| 9–16 | N05 | N08 |
| 17–24 | N03 | N09 |
| 25–32 | N07 | N13 |
| 33–40 | N02 | N04 |
| 41–48 | N06 | N10 |

每台 8 层、全部 128 个专家（完整模型）。布局写在 `qwen3-30b/layout_full_2ch.json`，转成清单：

```bash
python3 examples/manual_plan.py qwen3-30b/layout_full_2ch.json --out qwen3-30b/plan_full_2ch.json \
  --layer-times qwen3-30b/layer_bench_wash1.json --kv-reserve-gb 2 --topo task3/topo_fabric_mbps.json
```

* 前段 = 第 1–8 层那台（L₀ = 8，选择器照常在前段尾识别），后段 = 第 9–48 层那 5 台；
* 两条后段都标成 `back:any`（**共享后段池**）：三个 task 都没有专属后段，识别完一律派到这个池，
  哪条空闲用哪条（`p2pmoe/runtime/coordinator.py` 的 `SHARED_POOL`）。结果里每条请求的 `task` 记为
  `any`，选择器准确率仍按它自己判的类统计；
* 每台需要 ≤ 10.6GB（8 层 9.97GB + 首尾表 0.62GB），留 2GB KV 后可用 21.3–22.9GB，都装得下；
* 按实测每层耗时与拓扑估的每 token 时延（盲绑下前段与后段任意组合）：

| 前段 × 后段 | 正向 | 回环 | 合计 |
|---|---|---|---|
| N01 × 链 A 后段 | 0.04 | 1.21 | 118.7ms |
| N01 × 链 B 后段 | 1.21 | 3.10 | 158.5ms |
| N11 × 链 A 后段 | 3.08 | 3.25 | 126.6ms |
| N11 × 链 B 后段 | 3.24 | 0.04 | 160.3ms |

  链 B 后段的段内跳合计 31.3ms：N09→N13（13.5ms）和 N13→N04（11.5ms）两跳跨到 4.x 子网；
* 权重共 120GB：从 `plan_doc_l8.json` 换过来只有 N01 跳过（约 110GB）。

```bash
cd /home/ubuntu/p2p-framework && conda activate moe && source qwen3-30b/env.sh
export ADVERTISE=192.168.3.4 && mkdir -p results

HOSTS=task/hosts.txt bash ./deploy_15.sh stop
export PLAN=qwen3-30b/plan_full_2ch.json
awk '$1!="N12" && $1!="N14" && $1!="N15"' task/hosts.txt > task/hosts_full2.txt   # 用到的 12 台
export HOSTS=task/hosts_full2.txt

bash ./deploy_15.sh sync
REPLACE=1 bash ./deploy_15.sh fetch
bash ./deploy_15.sh verify            # 12 台都 ✓
bash ./deploy_15.sh start && bash ./deploy_15.sh status

export DUMP_FRONT=results/cont_front_full2.jsonl
RESULTS=results/run_full2.json bash ./deploy_15.sh continuous 2>&1 | tee results/continuous_full2.log
bash ./deploy_15.sh report results/run_full2.json | sed -n '/连续性测试/,$p'
python3 examples/analyze_run.py results/run_full2.json --csv results/run_full2_latency.csv
```

* 同时在途 = 2（两条后段）；开头日志应有「gsm8k/mbpp/no_robots 没有专属后段 → 都走共享池 any（2 条）」；
* 控制器仍在 wash1（N09，链 B 的第 17–24 层）上跑；原分配里 client 在 N01，要一致就在 N01 上执行上面的
  命令并把 `ADVERTISE` 换成 N01 的地址（192.168.1.2）；
* 别的手写分配照这个格式写一个布局 JSON，`task` 写某个 task 名就是那个 task 的专属后段，写 `any` 进共享池。

### 5.12 每个 task 均分 + 原始模型路径（`qwen3-30b/plan_vllm_eq_raw.json`、`plan_doc_v2_eq_raw.json`）

六步流水线只改两处（`--quota-equal --raw-fill`）：

1. **Step 2 按 task 均分**：N_work 照旧来自 Step 1 的显存估算（精确位配平 × θ），但不按 λ 做最大余额分配，
   而是每个 task ⌊N_work / task 数⌋ 条。这里 N_work = 5、3 个 task → **每个 task 1 条**，余下 2 条的显存
   不够每个 task 再各分一条；
2. Step 3–6 **原样**：建后段、公共带、建前段、回环裁剪、终审 —— 前段仍是 task 无关的，**盲绑、任意组合**；
3. **原始模型路径**（`p2pmoe/planner/raw.py`）：主流程之后剩下的节点（含备胎前段 —— 协调器从不用备胎）
   一前一后成对地建：后段第 9–48 层装**全部 128 个专家**，进共享池 `any`；前段与主流程的前段相同，并入前段池
   一起盲绑。建到放不下为止，仍空着的节点记在清单的 `idle` 里。全专家后段 50.5GB 要 3 台以上，用精确放置
   （`p2pmoe/planner/exact.py`：枚举节点链 + 逐链 DP 切层，目标函数同 `deploy_path`）—— beam 在这种段上不可靠。

在线（`runtime/coordinator.py`）：识别成 u 的请求先取 u 的专属后段；**u 全忙就溢出到原始模型路径**；两边都忙才在
u 池排队。原始模型路径空出来时接**所有池里等得最久**的那条，专属后段只接本池。结果里的 `task` 仍是识别结果
（准确率照常算），`back` 是 `Bany0` 的就是走了原始模型路径。

`tests/test_raw_fill.py::test_eq_raw_plan_file_is_reproducible` 校验两份清单逐字节重出：

```bash
# vLLM 实测（5.9 的输入）
python3 examples/task_deploy.py --data task3 --config qwen3-30b-a3b --phase prefill \
  --tasks gsm8k=1,mbpp=1,no_robots=1 \
  --expert-sets qwen3-30b/expert_sets_90/expert_sets_90_union.json \
  --l0 8 --front-full \
  --layer-times qwen3-30b/layer_bench_vllm.json --layer-time-key req@p512g512 \
  --measured-mem --mem-margin-gb 0 --kv-reserve-gb 2 --endpoint-w 0 \
  --quota-equal --raw-fill --save-plan qwen3-30b/plan_vllm_eq_raw.json

# PyTorch layerbench 实测（5.7 的输入）
python3 examples/task_deploy.py --data task3 --config qwen3-30b-a3b --phase prefill \
  --tasks gsm8k=1,mbpp=1,no_robots=1 \
  --expert-sets qwen3-30b/expert_sets_90/expert_sets_90_union.json \
  --l0 8 --front-full \
  --layer-times qwen3-30b/layer_bench_wash1.json --measured-mem \
  --mem-prefill-tokens 512 --mem-margin-gb 0 --kv-reserve-gb 2 --endpoint-w 0 \
  --quota-equal --raw-fill --save-plan qwen3-30b/plan_doc_v2_eq_raw.json
```

**`plan_vllm_eq_raw.json`**（15 台全上岗）：

| 段 | 节点（层） | 段时延 |
|---|---|---|
| 前段 F0 | N02(1–7)+N12(8) | 7.9ms |
| 前段 F1 | N08(1–7)+N10(8) | 6.6ms |
| 前段 F2 | N09 | 14.4ms |
| 前段 F3（原始模型路径） | N14 | 26.0ms |
| gsm8k | N07(9–32)+N06(33–48) | 8.3ms |
| mbpp | N01(9–33)+N04(34–48) | 7.9ms |
| no_robots | N05(9–25)+N03(26–48) | 8.5ms |
| any（原始模型路径） | N11(9–24)+N13(25–40)+N15(41–48) | 97.0ms |

盲绑组合 p50：专属后段 18.2–29.2ms（接 F3 时 56.9–57.6ms），原始模型后段 115.3–136.5ms。
核心 3×3 终审 56.4%（vLLM 输入下前段 6.6–14.4ms 与后段 8ms 同量级，3ms 的跨子网接口就占了组合时延的一大截）。
N11 是 Step 6 裁下来的备胎，拆回来进了原始模型路径。

**`plan_doc_v2_eq_raw.json`**（13 台，空 N13、N14）：

| 段 | 节点（层） | 段时延 |
|---|---|---|
| 前段 F0 / F1 / F2 | N05 / N06 / N11 | 19.3 / 19.7 / 21.9ms |
| 前段 F3（原始模型路径） | N15 | 21.1ms |
| gsm8k | N07(9–37)+N08(38–48) | 89.3ms |
| mbpp | N04(9–36)+N01(37–48) | 94.4ms |
| no_robots | N02(9–44)+N03(45–48) | 95.7ms |
| any（原始模型路径） | N09(9–21)+N10(22–38)+N12(39–48) | 109.1ms |

盲绑组合 p50：专属后段 109.1–123.7ms（接 F3 时 133.8–139.8ms），原始模型后段 131.1–157.3ms。核心 3×3 终审 7.5%（通过）；
N13、N14 合计 34.7GB，放不下第二条全专家后段（50.5GB）。

原始模型路径不在公共带内：清单里涉及它的组合的公共带 / w_cap 检查记为警告（vLLM 11 条、PyTorch 8 条），其余校验照旧严格。

```bash
cd /home/ubuntu/p2p-framework && conda activate moe
HOSTS=task/hosts.txt bash ./deploy_15.sh stop
source qwen3-30b/use_plan.sh vllm_eq_raw               # 或 doc_v2_eq_raw；hosts 只含清单用到的节点
bash ./deploy_15.sh sync
REPLACE=1 bash ./deploy_15.sh fetch
bash ./deploy_15.sh verify
bash ./deploy_15.sh start && bash ./deploy_15.sh status

RESULTS=results/run_vllm_eq_raw.json bash ./deploy_15.sh continuous 2>&1 | tee results/continuous_vllm_eq_raw.log
python3 examples/analyze_run.py results/run_vllm_eq_raw.json --csv results/run_vllm_eq_raw_latency.csv
python3 examples/analyze_run.py results/run_vllm_eq_raw.json channel   # 每条通道接了多少条、溢出到原始模型路径的有几条
```

* 同时在途 = 4（3 条专属后段 + 1 条原始模型后段）；日志里「…池 1 条全忙 → 溢出到原始模型路径」就是溢出；
* 动态模式照常（不加 `--static`），选择器照常在前段尾识别。

### 5.13 2026-11-05 新拓扑重新规划（`qwen3-30b/plan_1105_*.json`）

节点重新编号、换了机器（`task/hosts.txt`、`task/topo_fabric_mbps.json`）：9 台 A30（RUTG N01–N05、PRIN N06–N09）、
3 台 RTX6000（WASH N10–N12，wash1 = N10）、3 台 T4（N13 = wash4，N14/N15 = MICH）。站点间单向时延：
RUTG–PRIN 1.2ms，RUTG/PRIN–WASH 3.1–3.2ms，MICH 到其他站 14–16ms。

**1. 数据目录**：`task4/` = `task3/` 的激活统计 CSV + 新拓扑（`task3/` 不动，旧方案的复现测试读它）：

```bash
mkdir -p task4 && cp task3/*_expert_activation.csv task4/ && cp task/topo_fabric_mbps.json task4/
```

**2. 逐层实测表改挂到新编号**：表里的键是节点编号、数据是按机器测的，编号重排后要按主机名对回去
（`examples/remap_layer_bench.py`，主机名取 `hosts.txt` 每行注释里的括号）。旧表里没有的新机器
（prin4 = N09、mich1/mich2 = N14/N15）**借同型号卡里耗时居中那台**的实测 —— 只是估计，
尽快在新机器上跑一次 `bash ./deploy_15.sh layerbench`（vLLM 的也要重测）：

```bash
python3 examples/remap_layer_bench.py qwen3-30b/layer_bench_wash1.json \
  --hosts task/hosts.txt --topo task4/topo_fabric_mbps.json --out qwen3-30b/layer_bench_wash1_1105.json
python3 examples/remap_layer_bench.py qwen3-30b/layer_bench_vllm.json --ref qwen3-30b/layer_bench_wash1.json \
  --hosts task/hosts.txt --topo task4/topo_fabric_mbps.json --out qwen3-30b/layer_bench_vllm_1105.json
```

**3. 规划**（与 5.12 相同的方案：每个 task 均分 + 原始模型路径，只换 `--data` 与实测表）：

```bash
python3 examples/task_deploy.py --data task4 --config qwen3-30b-a3b --phase prefill \
  --tasks gsm8k=1,mbpp=1,no_robots=1 \
  --expert-sets qwen3-30b/expert_sets_90/expert_sets_90_union.json --l0 8 --front-full \
  --layer-times qwen3-30b/layer_bench_vllm_1105.json --layer-time-key req@p512g512 \
  --measured-mem --mem-margin-gb 0 --kv-reserve-gb 2 --endpoint-w 0 \
  --quota-equal --raw-fill --save-plan qwen3-30b/plan_1105_vllm_eq_raw.json

python3 examples/task_deploy.py --data task4 --config qwen3-30b-a3b --phase prefill \
  --tasks gsm8k=1,mbpp=1,no_robots=1 \
  --expert-sets qwen3-30b/expert_sets_90/expert_sets_90_union.json --l0 8 --front-full \
  --layer-times qwen3-30b/layer_bench_wash1_1105.json --measured-mem --mem-prefill-tokens 512 \
  --mem-margin-gb 0 --kv-reserve-gb 2 --endpoint-w 0 \
  --quota-equal --raw-fill --save-plan qwen3-30b/plan_1105_doc_eq_raw.json
```

换别的方案就照 5.7–5.12 的命令，把 `--data task3` 换成 `--data task4`、实测表换成 `*_1105.json`。

| | vLLM（`plan_1105_vllm_eq_raw.json`） | PyTorch（`plan_1105_doc_eq_raw.json`） |
|---|---|---|
| 前段 | N02+N12、N08+N11、N06+N10（A30 放 1–7 层、RTX 放第 8 层） | N05、N09、N06；原始模型路径 N14 |
| gsm8k | N07+N09 | N07+N08 |
| mbpp | N01+N04 | N04+N01 |
| no_robots | N05+N03 | N02+N03 |
| 原始模型路径 | 建不出（剩 3 台 T4 共 36GB < 50.5GB） | N12+N13+N11 |
| 组合 p50 | 17.5–19.4ms，终审 7.0% | 专属 108.6–144.9ms、原始模型 133.5–161.2ms；核心终审 2.4% |
| 空闲 | N14、N15（备胎 N13） | N10、N15 |

**前段为什么会拆到两台、怎么限定单节点**：vLLM 下组合时延中位只有 16ms，公共带宽上限 η × 16 = 2.4ms。
后段入口在 RUTG/PRIN，空闲的 A30（N02/N06/N08）到三个入口 0.04–1.2ms（跨度 1.2ms），WASH（N10–N13）
3.07–3.24ms（跨度 0.16ms）—— 两组进不了同一条 2.4ms 的带，按人口取了 4 台的 WASH 组。前段出口必须在 WASH，
最快的放法是「A30 放 1–7 层 + RTX 放第 8 层」。加 `--front-max-nodes 1` 禁止这种拆法：

| 方案（都在 vLLM 实测上） | 前段 | 组合 p50 | 原始模型路径 |
|---|---|---|---|
| `plan_1105_vllm_eq_raw.json`（默认） | N02+N12、N08+N11、N06+N10（两台） | 17.5–19.4ms | 建不出 |
| `plan_1105_vllm_eq_raw_f1.json`（`--front-max-nodes 1`） | N10、N11、N12（单台 RTX，出口仍在 WASH）+ N13 | 专属 28.5–40.4ms，原始模型 30.8–41.9ms；核心终审 0.5% | N06+N08+N02（A30，10.0ms） |
| `plan_1105_vllm_eq_raw_f1_rho1.json`（再加 `--rho-w 1`） | N02、N08、N06（单台 A30）+ N13 | 专属 9.7–40.4ms（接 A30 前段 9.7–12.7ms），原始模型 80.4–98.2ms；核心终审 20.2%（未过 15%） | N12+N10+N11（RTX，72.5ms） |

`--rho-w 1`（doc 给的范围 1–1.5 的下端）让 w_cap = 3.1ms，WASH 的出口（≥ 3.07–3.24ms）出局，前段只能用 A30。
三份都是 13 台、空 N14/N15；接 N13（T4）前段的组合都在 40ms 左右。

**按 task 分配（不建原始模型路径）—— 现在用的方案**：六步流水线照原样（Step 2 按 λ 最大余额分配额），
前段单节点（`--front-max-nodes 1`），主流程用不上的节点**按 task 补位**（`--fill-idle`：一前一后成对地加，
后段给公平比最低的 task）。`tests/test_plan_1105.py` 校验逐字节重出：

```bash
# vLLM（加 --rho-w 1：w_cap = 3.1ms，前段出口落在 A30 上，不去 WASH）
python3 examples/task_deploy.py --data task4 --config qwen3-30b-a3b --phase prefill \
  --tasks gsm8k=1,mbpp=1,no_robots=1 \
  --expert-sets qwen3-30b/expert_sets_90/expert_sets_90_union.json --l0 8 --front-full \
  --layer-times qwen3-30b/layer_bench_vllm_1105.json --layer-time-key req@p512g512 --rho-w 1 \
  --measured-mem --mem-margin-gb 0 --kv-reserve-gb 2 --endpoint-w 0 \
  --front-max-nodes 1 --fill-idle --save-plan qwen3-30b/plan_1105_vllm_fill.json

# PyTorch
python3 examples/task_deploy.py --data task4 --config qwen3-30b-a3b --phase prefill \
  --tasks gsm8k=1,mbpp=1,no_robots=1 \
  --expert-sets qwen3-30b/expert_sets_90/expert_sets_90_union.json --l0 8 --front-full \
  --layer-times qwen3-30b/layer_bench_wash1_1105.json --mem-prefill-tokens 512 \
  --measured-mem --mem-margin-gb 0 --kv-reserve-gb 2 --endpoint-w 0 \
  --front-max-nodes 1 --fill-idle --save-plan qwen3-30b/plan_1105_doc_fill.json
```

两份都是 15 台全上岗、5 条前段 × 5 条后段，补位段不在公共带内（清单里记为警告）。

`plan_1105_vllm_fill.json`：配额 {gsm8k 2, mbpp 2, no_robots 1}，公共带凑不够、Step 4 降掉 mbpp、gsm8k 各一条 →
主流程每个 task 1 条；补位再给 gsm8k、no_robots 各加一条。

| 段 | 节点（层） | 段时延 | 组合 p50（接 5 条前段） |
|---|---|---|---|
| 前段 | N05、N03、N06（A30）；补位 N13、N14（T4） | 1.7ms / 25.6ms | |
| gsm8k | N08(9–34)+N02(35–48) | 10.1ms | 13.1–64.0ms |
| gsm8k（补位） | N10(9–33)+N12(34–48)（RTX） | 72.3ms | 80.2–130.1ms |
| mbpp | N07(9–33)+N09(34–48) | 8.3ms | 10.1–62.3ms |
| no_robots | N01(9–37)+N04(38–48) | 7.8ms | 9.6–61.6ms |
| no_robots（补位） | N11(9–39)+N15(40–48)（RTX+T4） | 101.4ms | 120.2–143.1ms |

核心 3×3（A30 前段 × 主流程后段）9.6–13.1ms，终审 19.1%（未过 15%：组合只有 10ms 上下，3ms 的绝对差就超了）。

`plan_1105_doc_fill.json`：配额 {gsm8k 2, mbpp 2, no_robots 1}，Step 4 降掉 gsm8k 一条 → 主流程 gsm8k 1、mbpp 2、
no_robots 1；补位给 no_robots 加一条。

| 段 | 节点（层） | 段时延 | 组合 p50 |
|---|---|---|---|
| 前段 | N02、N03（A30）、N11、N12（RTX）；补位 N10（RTX） | 19.1–22.5ms | |
| gsm8k | N04(9–36)+N01(37–48) | 94.4ms | 113.6–123.1ms |
| mbpp | N09(9–37)+N06(38–48)、N05(9–36)+N13(37–48) | 96.4 / 101.5ms | 118.0–127.1ms |
| no_robots | N07(9–33)+N08(34–48) | 89.9ms | 111.5–118.9ms |
| no_robots（补位） | N14(9–28)+N15(29–48)（T4×2） | 105.7ms | 152.9–160.3ms |

核心 4×4 终审 7.4%（通过）；全网格 39.4%（补位段的已知代价）。

上面「原始模型路径」那几份（`plan_1105_*_eq_raw*.json`）留作对照，不再用。

**只跑主流程的 3 条、不跑补位**：用 `qwen3-30b/plan_1105_vllm.json`（同一条 vLLM 命令去掉 `--fill-idle`）。
它的 9 台节点、层区间、专家与 `plan_1105_vllm_fill.json` 的核心部分逐项相同（`tests/test_plan_1105.py` 校验），
所以**不用重新拉权重**，换清单即可；补位的 6 台（N10–N15）不起 agent：

```bash
HOSTS=task/hosts.txt bash ./deploy_15.sh stop                     # 先把 15 台都停掉
source qwen3-30b/use_plan.sh qwen3-30b/plan_1105_vllm.json        # HOSTS 只剩 9 台
bash ./deploy_15.sh sync && bash ./deploy_15.sh verify            # verify 应 9 台都 ✓（权重是现成的）
bash ./deploy_15.sh start && bash ./deploy_15.sh status
```

**只按 decode 规划（对照）**：上面的 vLLM 方案用 `req@p512g512`（一次 512 token prefill 摊到 512 个生成 token
上，加进每层耗时）。换成 `--layer-time-key decode@p512g512` 就只按 decode（上下文 512→1023 的平均）规划：

```bash
python3 examples/task_deploy.py --data task4 --config qwen3-30b-a3b --phase prefill \
  --tasks gsm8k=1,mbpp=1,no_robots=1 \
  --expert-sets qwen3-30b/expert_sets_90/expert_sets_90_union.json --l0 8 --front-full \
  --layer-times qwen3-30b/layer_bench_vllm_1105.json --layer-time-key decode@p512g512 --rho-w 1 \
  --measured-mem --mem-margin-gb 0 --kv-reserve-gb 2 --endpoint-w 0 \
  --front-max-nodes 1 --save-plan qwen3-30b/plan_1105_vllm_dec.json        # 加 --fill-idle → plan_1105_vllm_dec_fill.json
```

节点与通道和 req 版完全相同，只有两条后段的切层点不同（两台都是 A30、每层耗时只差几个百分点，切在哪层
段时延只差零点几毫秒；换了计时口径，求解器在几乎打平的切法里选了另一个）：

| 段 | req@p512g512（`plan_1105_vllm*.json`） | decode@p512g512（`plan_1105_vllm_dec*.json`） |
|---|---|---|
| no_robots | N01(9–37)+N04(38–48) | N01(9–25)+N04(26–48) |
| gsm8k | N08(9–34)+N02(35–48) | N08(9–35)+N02(36–48) |
| 其余（前段 N05/N03/N06、mbpp N07+N09、补位段） | 同 | 同 |

两份之间切换时 `REPLACE=1 bash ./deploy_15.sh fetch` 只会重拉 N01、N02、N04、N08 四台，其余跳过。

**4. 部署**：`ADVERTISE` 现在是 **192.168.3.2**（wash1 = N10）。`~/.bashrc` 里写过 `export ADVERTISE=192.168.3.4`
的要改掉 —— 那个地址现在是 wash3（N12）。

```bash
HOSTS=task/hosts.txt bash ./deploy_15.sh stop
source qwen3-30b/use_plan.sh qwen3-30b/plan_1105_vllm_fill.json     # 或 plan_1105_doc_fill.json
bash ./deploy_15.sh sync && REPLACE=1 bash ./deploy_15.sh fetch && bash ./deploy_15.sh verify
bash ./deploy_15.sh start && bash ./deploy_15.sh status
```

### 5.14 Petals 的分配直接拿来部署（`qwen3-30b/plan_petals_cap8.json`）

Petals 给的分配（`qwen3-30b/petals/plan_cap8_petals.json`）：两条链，每台 8 层、完整模型，层号是 0-based 切片
（`0:8` = 第 1–8 层）。转成本框架的清单（与 5.11 同一条路：手写布局 → `manual_plan.py`）：

```bash
python3 examples/petals_to_layout.py qwen3-30b/petals/plan_cap8_petals.json --out qwen3-30b/layout_petals_cap8.json
python3 examples/manual_plan.py qwen3-30b/layout_petals_cap8.json --out qwen3-30b/plan_petals_cap8.json \
  --layer-times qwen3-30b/layer_bench_wash1_1105.json --kv-reserve-gb 2 --topo task4/topo_fabric_mbps.json
```

| 层 | 链 A | 链 B |
|---|---|---|
| 1–8（前段） | N06 | N02 |
| 9–16 | N01 | N03 |
| 17–24 | N07 | N04 |
| 25–32 | N08 | N05 |
| 33–40 | N09 | N11 |
| 41–48 | N10 | N12 |

* 前段 = 第 1–8 层那台（L₀ = 8），后段 = 其余 5 台，标 `back:any` 进共享池（同 5.11）；专家全装；
* 每台要 ≤ 10.6GB（8 层 9.97GB + 首尾表 0.62GB），留 2GB KV 后可用 21.3–22.9GB，都装得下；
* 按 PyTorch 实测估的组合 p50：123.7–128.3ms（盲绑，前段 × 后段任意组合）；
* Petals 的 `client`（N06 / N02）不进清单，控制器仍在 wash1（N10）；N13–N15 不用。

```bash
HOSTS=task/hosts.txt bash ./deploy_15.sh stop
source qwen3-30b/use_plan.sh qwen3-30b/plan_petals_cap8.json     # HOSTS 只含这 12 台
bash ./deploy_15.sh sync
REPLACE=1 bash ./deploy_15.sh fetch                              # 各节点删掉旧方案的层、拉自己这 8 层（全部专家）
bash ./deploy_15.sh verify && bash ./deploy_15.sh start && bash ./deploy_15.sh status
RESULTS=results/run_petals_cap8.json bash ./deploy_15.sh continuous
```

以后 Petals 出了新的分配，换掉第一条命令的输入文件、给新的 `--out` 名字即可。

### 5.15 单前段：一台节点放第 1–2 层，所有请求共用（`qwen3-30b/plan_1105_*_1f.json`）

算法改动（`--single-front`，`PlannerConfig.single_front`）：

* **后段不变**：Step 1 分档估上限（前段只算一次，不再每条通道各算一条）、Step 2 按 λ 最大余额配额、
  Step 3 建后段（按轮次交错，见下）+ 间隙检测 + 字典序收紧 —— 后段之间照旧要均衡；
* **前段只有一条、只放一台**：不再有「多条前段 × 各条后段组合时延一致」的要求，所以 Step 4（公共中值域）、
  Step 5（超建前段）、Step 6（回环裁剪）都不做。后段建完之后，从剩下的节点里按同一个目标函数（计算 +
  稳定性 + 抖动 + 端点项）挑一台放第 1..L₀ 层，端点项取它到全体后段的正向 / 回环的平均实测代价；
* **在线**：清单里这条前段记 `slots = 后段条数`，协调器在空闲队列里放这么多个它的名额 —— 同时在途的请求数
  = 后段条数，每条请求在前段上的 KV 分开存；节点上的前向串行执行（`node.py` 的 `_fwd_lock`）。

```bash
# vLLM
python3 examples/task_deploy.py --data task4 --config qwen3-30b-a3b --phase prefill \
  --tasks gsm8k=1,mbpp=1,no_robots=1 \
  --expert-sets qwen3-30b/expert_sets_90/expert_sets_90_union.json --l0 2 --front-full \
  --layer-times qwen3-30b/layer_bench_vllm_1105.json --layer-time-key req@p512g512 \
  --measured-mem --mem-margin-gb 0 --kv-reserve-gb 2 --endpoint-w 0 \
  --single-front --save-plan qwen3-30b/plan_1105_vllm_1f.json
# PyTorch：把 --layer-times 换成 qwen3-30b/layer_bench_wash1_1105.json --mem-prefill-tokens 512（去掉 --layer-time-key），
#          --save-plan qwen3-30b/plan_1105_doc_1f.json
```

**后段建段顺序**：单前段一律**按轮次交错**——每个 task 先各建一条，再各建第二条…；每一轮里按
公平比从低到高（瓶颈池先）。λ 相同时多出的那 1 条配额给了 gsm8k，它公平比最高，所以**每一轮都排最后**：
`mbpp0 → no_robots0 → gsm8k0 → mbpp1 → no_robots1 → gsm8k1 → gsm8k2`。另外单前段自动给前段预留一台
（影子租金 smallest_sufficient），不然后段可能把节点吃光（PyTorch 实测 6 条后段用掉全部 15 台）。

**收紧用精确放置**：单前段时两轮筛的重建自动改用 `planner/exact.py`（`--tighten-exact` 可在其他模式下打开）。
beam 在收紧的小池子上会交出「一台 RTX 只放 1 层却占着两台 RTX」的链，下一条段无处可放、整轮回退 ——
vLLM 下 gsm8k 两条 9.8 / 83.5ms 原样没动；改用精确放置后拆成「A30 + RTX」各一条，43.2 / 43.4ms。

**按轮次分组收紧**（单前段时自动打开；其他模式 `--balance-by-round`）：两轮筛不再按 task 分组，而是按轮次 ——
所有 task 的第 1 条后段（第 1 轮建的）一组收紧，第 2 条一组收紧……；组内是不同 task 的段，各用各的驻留集重建，
目标区间取组内中位 ± β。同一 task 的第 1 条与第 2 条不要求一样快。日志的 `[终审·按组]` 报每组的组合极差。

Step 1 给出 7 条（配额 gsm8k 3、mbpp 2、no_robots 2），gsm8k 第 3 条放不下 → 每个 task 2 条，共 6 条后段。
组合 p50 = 前段 + 正向 + 后段 + 回环（每 token）：

| | vLLM（`plan_1105_vllm_1f.json`，13 台） | PyTorch（`plan_1105_doc_1f.json`，14 台） |
|---|---|---|
| 前段 | N13（T4，第 1–2 层，6.4ms） | N13（T4，5.2ms） |
| 第 1 组 | mbpp N01+N04 21.6、no_robots N07+N09 22.4、gsm8k N05+N03 22.4ms | mbpp N07+N08 115.0、no_robots N04+N01 119.9、gsm8k N02+N03 121.4ms |
| 第 2 组 | mbpp N06+N12 53.1、no_robots N02+N10 48.2、gsm8k N08+N11 53.5ms | mbpp N09+N06 122.8、no_robots N05+N12 126.0、gsm8k N14+N15+N11 161.3ms |
| 组内相对极差 | 第 1 组 3.4%、第 2 组 9.9% | 第 1 组 5.3%、第 2 组 30.6% |
| 空闲 | N14、N15 | N10 |

* vLLM 第 2 组收紧前是 mbpp 两台 A30（10.0ms）、no_robots A30+RTX（38.7ms）、gsm8k 两台 RTX（83.5ms）：
  拆掉偏快的 mbpp 和偏慢的 gsm8k，重配成 A30+RTX 各一条 —— 三条都落在 38–44ms；
* PyTorch 第 2 组的 gsm8k（两台 MICH 的 T4 + RTX，140ms）仍在组内中位 ±30%（β = 1.3）以内，收紧不触发；
  要按 15% 拉得把 β 调到约 1.15。

**识别**：主用 `selector/selector_body_L2_cmw_params.npz`（2026-11-16 在 `verify/` 里用同一批 20% 参考样本训练，
`verify/train_selector_L2.py --layers 2 --feature counts+meanw`；报告见 `selector/selector_body_L2_cmw_report.txt`）：
前 2 层，每层拼「选中次数份额 + 平均门控权重（wsum/count）」，512 维。样本外平衡准确率 98.4%（只用份额的 L2 是 96.9%，
原来 8 层的是 98.0%），最大概率 ≥ 0.9 的 95.8% 样本里准确率 99.0%。特征口径与 `verify/task_selector.py` 逐位一致
（`tests/test_selector_cmw.py`）。`env.sh` 默认已改成它；它只用前 2 层，L₀ ≥ 2 的方案都能用：

```bash
HOSTS=task/hosts.txt bash ./deploy_15.sh stop
source qwen3-30b/use_plan.sh qwen3-30b/plan_1105_vllm_1f.json    # 或 plan_1105_doc_1f.json
bash ./deploy_15.sh sync && REPLACE=1 bash ./deploy_15.sh fetch && bash ./deploy_15.sh verify
bash ./deploy_15.sh start && bash ./deploy_15.sh status
RESULTS=results/run_1105_vllm_1f.json bash ./deploy_15.sh continuous
```

### 5.16 先定前段：按计算定前段，再围绕它按组找后段（`qwen3-30b/plan_1105_*_ff*.json`）

单前段（5.15）是先建后段、再从剩下的节点里挑前段 —— A30 都被后段占了，前段落到 T4 上，而这台要同时服务
全部请求。`--front-first` 把顺序倒过来：

1. **先定前段（按计算）**：能单台装下第 1–2 层的节点里，第 1–2 层计算时延最小的那台（并列时取到全体节点单向
   时延均值最小的）；
2. **按组选后段首节点**：第 1 组每个 task 各选一个，取前段 → 首节点的正向时延最小的空闲节点（按 0.1ms 取整比，
   同站点内算并列，并列时内存大、每层快的优先）；
3. **按组从首节点建链**：首节点固定，目标函数照旧（计算 + 段内跳 + 稳定性 + 抖动），端点项只计链尾 → 前段的回环；
   默认精确放置（beam 在这里会漏解：vLLM 第 2 组 gsm8k 从 N12 起 beam 建不出链），`--chain-beam` 改回 beam；
   第 1 组建完再选第 2 组的首节点、建第 2 组的链 ……；
4. 之后照原来：按组收紧（精确放置重建）、终审。

```bash
python3 examples/task_deploy.py --data task4 --config qwen3-30b-a3b --phase prefill \
  --tasks gsm8k=1,mbpp=1,no_robots=1 \
  --expert-sets qwen3-30b/expert_sets_90/expert_sets_90_union.json --l0 2 --front-full \
  --layer-times qwen3-30b/layer_bench_vllm_1105.json --layer-time-key req@p512g512 \
  --measured-mem --mem-margin-gb 0 --kv-reserve-gb 2 --endpoint-w 0 \
  --front-first --save-plan qwen3-30b/plan_1105_vllm_ff.json
# 加 --avail 0.95 → plan_1105_vllm_ff_a95.json；PyTorch 换 --layer-times qwen3-30b/layer_bench_wash1_1105.json
#   --mem-prefill-tokens 512（去掉 --layer-time-key）→ plan_1105_doc_ff.json / plan_1105_doc_ff_a95.json
```

**`--avail 0.95`**：可用率 r_v < 1 时目标函数里的稳定性项 μ·Σ(−log r_v) 每多用一台节点罚约 2ms。r_v = 1（默认）
时这一项为 0，求解器会为省 0.3ms 多占一台 A30（例如第 1 组 gsm8k 成了 N02 只放 1 层 + N09 + N06），第 2 组就缺节点：
PyTorch 版 gsm8k 第 2 条建不出来。

| | 前段 | 第 1 组（组合 p50） | 第 2 组 | 台数 |
|---|---|---|---|---|
| `plan_1105_vllm_ff.json` | N01（A30，0.39ms） | mbpp N04+N05 9.9、no_robots N03+N07 12.3、gsm8k N02+N09+N06 12.7ms | mbpp N11+N08 50.1、no_robots N10+N13 110.5、gsm8k N12+N14+N15 146.3ms | 15 |
| `plan_1105_vllm_ff_a95.json` | N01 | 9.9 / 12.3 / gsm8k N02+N09 12.9ms | mbpp N06+N12 48.1、no_robots N08+N11 37.3、gsm8k N10+N13+N14 146.0ms | 14 |
| `plan_1105_doc_ff.json` | N07（A30，4.4ms） | 114.7 / 116.5 / 117.0ms | mbpp 125.9、no_robots N12+N14+N15 159.8ms；gsm8k 第 2 条建不出 | 14 |
| `plan_1105_doc_ff_a95.json` | N07 | 114.7 / 116.5 / 117.9ms | mbpp 125.4、no_robots N05+N10 126.7、gsm8k N12+N14+N15 160.5ms | 15 |

第 1 组都贴着前段（同站点 0.05ms），组内相对极差 vLLM 约 23%、PyTorch 约 2–3%；第 2 组每轮排最后的 task 只剩
RTX / MICH 的 T4，最慢。前段现在是 A30，但它同时服务全部请求时的真实代价（排队、prefill 插队）规划里还没算 ——
要另测前段在 c 条并发下的单步耗时与 2 层 prefill 耗时，才能放进前段的计算项。

### 5.17 单前段 + 理论计算耗时（`qwen3-30b/plan_1105_*_1f_theory.json`）

与 5.15 同一个算法（先建后段、按组收紧，最后挑一台放第 1–2 层），区别只在计算项：每台的 ms/层不用 layerbench 的实测，
而用拓扑 `compute_capacity` 的反比（A30 165000 → 0.35，RTX6000 130500 → 0.443，T4 65000 → 0.889 ms/层）。
显存仍用实测（`--measured-mem`，KV 留 2GB），所以 vLLM 版与 PyTorch 版只差在每台能装多少层。

```bash
# vLLM 显存
python3 examples/task_deploy.py --data task4 --config qwen3-30b-a3b --phase prefill \
  --tasks gsm8k=1,mbpp=1,no_robots=1 \
  --expert-sets qwen3-30b/expert_sets_90/expert_sets_90_union.json --l0 2 --front-full \
  --layer-times qwen3-30b/layer_bench_vllm_1105.json --layer-time-key req@p512g512 --theory-times \
  --measured-mem --mem-margin-gb 0 --kv-reserve-gb 2 --endpoint-w 0 \
  --single-front --save-plan qwen3-30b/plan_1105_vllm_1f_theory.json
# PyTorch 显存：换 --layer-times qwen3-30b/layer_bench_wash1_1105.json --mem-prefill-tokens 512（去掉 --layer-time-key）
#   → plan_1105_doc_1f_theory.json
```

**前段硬保留**：理论值下 T4 只有 0.89ms/层（实测 2.6–3.2），后段代价偏低，影子租金（软）被出价超过 ——
PyTorch 显存下第 7 条 gsm8k 把三台 T4 全占，前段无处可放。此时规划器自动重来一次，Step 3 期间把租金那台（N13）
**硬保留**给前段（日志：`[单前段] 第一次规划后段把节点用光 … 硬保留 ['N13']`）；第一次就成功的方案不受影响，
5.15 的两份 1f 方案原样复现。

两份方案都是 6 条后段（gsm8k 第 3 条放不下）、13 台，前段 N13（T4），N14 / N15 空闲；后段全在 A30 + RTX 上：

| 第几组 | vLLM 显存（`plan_1105_vllm_1f_theory.json`） | PyTorch 显存（`plan_1105_doc_1f_theory.json`） |
|---|---|---|
| 第 1 组 | mbpp N02+N03、no_robots N01+N04、gsm8k N06+N09 | mbpp N02+N03、no_robots N04+N05、gsm8k N06+N09 |
| 第 2 组 | mbpp N07+N08、no_robots N05+N11、gsm8k N10+N12 | mbpp N07+N08、no_robots N01+N11、gsm8k N10+N12 |

把同一份方案的组合 p50 分别按理论值、按实测值算（前段 2 层 + 正向 + 后段 + 回环，不含稳定性项）：

| 方案 | 按理论 | 按实测 |
|---|---|---|
| `plan_1105_vllm_1f_theory`（实测用 req@p512g512） | 22.3–25.6ms | 21.7–89.7ms（no_robots N05+N11 49.7、gsm8k N10+N12 89.7） |
| `plan_1105_vllm_1f`（5.15，按实测规划） | — | 21.6–53.5ms |
| `plan_1105_doc_1f_theory`（实测用 decode@1024） | 22.3–25.6ms | 116.7–132.5ms（相对极差 14%） |
| `plan_1105_doc_1f`（5.15） | — | 115.0–161.3ms（40%，gsm8k 第 2 条走 N14+N15+N11） |

理论值里 RTX6000 只比 A30 慢 1.26 倍，vLLM 实测约 8.6 倍（1.81 vs 0.21ms/层），所以 vLLM 版第 2 组把 2 台 RTX
串成一条 gsm8k，实际最慢；PyTorch 实测 RTX 只比 A30 慢约 1.17 倍（2.76 vs 2.36），与理论接近，理论方案反而比 5.15
更均衡（没让 T4 做后段）。

### 5.18 GLM-4.7-Flash：Petals 的分配 → 清单 → 拉权重 → 逐层计时（`glm47flash/`）

GLM-4.7-Flash（`zai-org/GLM-4.7-Flash`，`glm4_moe_lite`）与 Qwen3-30B-A3B 结构不同，执行层另写了一份
`p2pmoe/runtime/glm4_moe_lite.py`（节点、fetch、layerbench 按 config 的 `model_type` 自动分派）：

* 47 层：第 1 层（checkpoint 第 0 层）是 dense MLP（10240），第 2–47 层是 MoE；checkpoint 里的
  `model.layers.47.*` 是 MTP 层，不部署、不下载；
* 64 个专家选 4 + 1 个共享专家（不参与裁剪）；路由 sigmoid + 偏置选专家，权重归一后 ×1.8，路由器两项 fp32；
* MLA 注意力（q / kv 低秩，rope 64 维交错），KV 按展开后缓存：每 token 每层 20KiB（与 Petals 方案一致）；
* 节点只要 torch + safetensors（不要 transformers）。数值与 transformers 5.x 的官方实现逐层对过
  （`tests/test_glm4_moe_lite.py`，同形状的小模型，prefill + decode 的 logits 差 < 1e-4）。

**清单**（Petals 的 `glm47flash/petals/plan_glm47flash_16k_cap8.json` → 本框架的清单）：

```bash
python3 examples/petals_to_layout.py glm47flash/petals/plan_glm47flash_16k_cap8.json \
  --out glm47flash/layout_petals_cap8.json
python3 examples/manual_plan.py glm47flash/layout_petals_cap8.json --config glm47flash/config.json \
  --ctx 16384 --out glm47flash/plan_petals_cap8.json
```

L₀ = 8，2 条链、12 台，专家全装，后段进共享池（`task: any`）；N13–N15 空闲：

| 链 | 前段（第 1–8 层） | 后段 |
|---|---|---|
| A | N06 | N05[9–16] → N07[17–24] → N08[25–32] → N09[33–40] → N10[41–47]（RTX） |
| B | N01 | N02[9–16] → N03[17–24] → N04[25–32] → N11[33–40]（RTX） → N12[41–47]（RTX） |

每台约 9.5–10.2GB（全装一层 1.27GB；前段多一张词嵌入 0.63GB、第 1 层 dense 只有 0.17GB；末台多输出头 0.63GB），
两条链合计约 120GB。清单里第 1 层也按 64 个专家列着（手写布局默认全装），`weight_gb` 因此对前段偏大约 1.1GB；
拉权重与装载时 dense 层的专家号一律忽略。

**在 wash1 上**（先把 Mac 上的代码 rsync 过去，同第 1 步）：

```bash
cd /home/ubuntu/p2p-framework
HOSTS=task/hosts.txt bash ./deploy_15.sh stop          # 先停掉现在跑着的方案（15 台）
source glm47flash/use_plan.sh petals_cap8              # env + PLAN + 只含这 12 台的 HOSTS
bash ./deploy_15.sh sync                               # 代码发到 12 台
bash ./deploy_15.sh meta                               # 控制机取 config.json / tokenizer 到 $WEIGHTS（约 10MB）
# 硬盘只装得下一个模型：下载前先删这 12 台上 Qwen3-30B-A3B 的权重目录（源站试通、回 y 之后才删）
PURGE_OLD=$WORKDIR/weights-q3-30b bash ./deploy_15.sh fetch   # 12 台各拉自己那份到 weights-glm47
bash ./deploy_15.sh verify                             # 12 台都应 ✓
# 不在这份清单里的 N13–N15 也清掉（先列出、问一次再删；CONFIRM=1 不问）
HOSTS=task/hosts.txt bash ./deploy_15.sh purge-weights $WORKDIR/weights-q3-30b

# 逐层计时：15 台都量（N13–N15 不在这份清单里，但以后给 GLM 规划要用）
HOSTS=task/hosts.txt bash ./deploy_15.sh sync
HOSTS=task/hosts.txt bash ./deploy_15.sh start
HOSTS=task/hosts.txt LB_MODE=synthetic bash ./deploy_15.sh layerbench
cp results/layer_bench.json glm47flash/layer_bench_wash1_glm47.json
```

`purge-weights` 删的是**整个目录**（权重、config、tokenizer）。护栏：必须是绝对路径、目录名以 `weights`
开头（`FORCE=1` 才放行别的）、不能是当前的 `$WEIGHTS`；有进程开着里面的文件（agent 还装着 Qwen）就整批不删，
先 `HOSTS=task/hosts.txt bash ./deploy_15.sh stop`。删之前逐台列出大小与 config（如 `qwen3_moe，48 层 × 128 专家`），
删之后报剩余空间。以后切回 Qwen 要重新 `fetch`（反过来同理：`PURGE_OLD=$WORKDIR/weights-glm47`）。

layerbench 用随机权重建 4 个 MoE 层（第 2–5 层；第 1 层 dense 不代表其余 46 层，跳过）、64 个专家全装，
约 5.1GB 显存，T4 也装得下；结果的 `model.arch` 应为 `glm4_moe_lite`。量出来的表与 Qwen 的同格式，
可直接给 `task_deploy.py --layer-times` / `manual_plan.py --layer-times`。

还没做的：GLM 的 task 选择器（`LR_MODEL` 置空）、规划器对 GLM 的适配（dense 层、层号对齐），以及
对话模板 / 停止符的端到端核对 —— 拉权重和逐层计时用不到这些。

### 5.19 GLM-4.7-Flash：按理论算力规划（单前段，`glm47flash/plan_glm47_1f_theory.json`）

GLM 第 1 层是 dense，激活统计（`task_glm47/*.csv`）与驻留集（`glm47flash/glm47_expert_sets_90_union.json`）都只有
46 个 MoE 层。规划器按 config 的 `first_k_dense_replace` 认出 dense 层：画像在第 1 层补均匀分布（各 task 一样，
不参与识别），驻留集补全集；第 1 层的权重按 dense 记（注意力 + SwiGLU 0.169GB，`ModelSpec.layer_gb`），
与专家数无关。Qwen 的方案不受影响（没有 dense 层，`tests/test_single_front.py` 原样复现）。

```bash
python3 examples/task_deploy.py --data task_glm47 --config glm47flash/config.json --phase prefill \
  --tasks gsm8k=1,mbpp=1,no_robots=1 --expert-sets glm47flash/glm47_expert_sets_90_union.json \
  --l0 3 --front-full --single-front --endpoint-w 0 --save-plan glm47flash/plan_glm47_1f_theory.json
```

* 计算按拓扑 `compute_capacity` 的反比（A30 0.35、RTX6000 0.443、T4 0.889 ms/层），显存按标称（`memory_capacity`
  − `--reserve-gb 1`，KV 按 `--ctx 2048`）—— GLM 的 layerbench 还没有，不加 `--layer-times` / `--measured-mem`；
  标称口径下前段的词嵌入、后段末台的输出头（各约 0.63GB）不计，量完 layerbench 换 `--measured-mem` 重出；
* L₀ = 3：dense 第 1 层 + 2 个 MoE 层，与 Qwen 的 L₀ = 2（2 个 MoE 层，L2 选择器）同口径。直方图的识别率下界
  只有 52%（Chernoff 界偏保守，Qwen 在 L₀ = 2 时也只有 53.5%，实际选择器 98%），GLM 的选择器还没训；
* 结果：前段 N13（T4，第 1–3 层，2.7GB），6 条后段每条 2 台、全在 A30 + RTX 上（第 4–23/24–47 层上下），
  13 台，N14 / N15 空闲；组合时延（理论）第 1 组 24.3–24.6ms、第 2 组 22.3–26.3ms，终审通过。

### 5.20 换方案时节点互传（`deploy_15.sh fetch` 默认开，`PEERS=0` 关）

同一个模型换方案（如 GLM 的 Petals → `plan_glm47_vllm_1f_theory`），新方案要的张量大多已经在集群里：
`fetch` 先在 `$HOSTS_ALL`（默认 `task/hosts.txt`，15 台）的各台上用 `serve_weights` 把**现有**权重目录发出来
（端口 `PEER_PORT`，默认 9400），每台按「本机旧权重 → 别的节点（按往返由近到远）→ 上游」取张量：

* 对等节点只认**同一个模型**（`config.json` 一致）—— Qwen 的权重不会被当成 GLM 的；
* 新文件先写成 `model.safetensors.next`，旧文件不动（别的节点可能还在读它）；**全部节点拉完**后控制机统一
  `fetch --commit-staged` 换上。某台失败就不换，旧权重还在，重跑即可；
* 某个对等节点中途掉线，它没给完的张量自动改从上游拿，结果不变（`tests/test_fetch_peers.py` 逐位核对）；
* 要能同时放下新旧两份；放不下且 `REPLACE=1` 时那台先删旧的（它的张量不再复用、也不再给别人）。

Petals → `plan_glm47_vllm_1f_theory`：13 台要的张量**全部**在 Petals 那 12 台上，上游只取 config / tokenizer。

```bash
source glm47flash/use_plan.sh glm47flash/plan_glm47_vllm_1f_theory.json
HOSTS=task/hosts.txt bash ./deploy_15.sh stop
bash ./deploy_15.sh sync && HOSTS=task/hosts.txt bash ./deploy_15.sh sync
REPLACE=1 bash ./deploy_15.sh fetch
bash ./deploy_15.sh verify
```

## 6. 参考数值（用来判断复现是否正常）

| 实验 | 指标 | 历史值 |
|---|---|---|
| 静态 20 条 | 每 token（TPOT） | 129–145ms，随通道（mbpp-1 带 T4 最慢） |
| 静态 20 条 | 选择器（按自己判的类） | 20/20，置信 1.00 |
| 动态 20 条（带 LR_CLASSES） | 跨 4.x 子网的配对 | 每 token 多 20–25ms |
| 动态 20 条 | 时延构成 | 计算 ~75%，后段排队 ~17%，网络+开销 ~7% |
| 连续 150 条 | TPOT | 均值 ~143ms，p99 ~178ms |
| 连续 150 条 | 选择器 | 150/150 |
| 连续 150 条 | 去排队 TTFT | gsm8k ~1.5s，mbpp ~2.2s（T4 链路长尾 ~3.8s），no_robots ~1.3s |
| 连续 150 条 | 原始 TTFT | 长尾几乎全是 no_robots 的后段排队（只有 1 条通道） |

生成的文本是确定性的（贪心解码）：静态 req0 以 "Let's solve the problem step by step." 开头，
req12 在第 27 个 token 处停于 eos。对不上就说明代码版本或配置与当时不同。

## 7. 踩过的坑

| 症状 | 原因 | 处理 |
|---|---|---|
| `✗ 必须设 ADVERTISE` | 新终端没 export | 第 2 步 |
| fetch 预检 `TypeError: 'type' object is not subscriptable`，路径是 `/home/ubuntu/P2P_MoE` | 没 `source qwen3-30b/env.sh`：脚本退回旧默认（旧目录、系统 Python 3.8） | `source qwen3-30b/use_plan.sh <清单>` |
| verify/fetch 报「清单里没有节点 Nxx」、某台没拉 | hosts 与清单不配套 | `source qwen3-30b/use_plan.sh <清单>` 自动生成对应 hosts |
| `No module named 'tokenizers'` / 奇怪的导入错误 | 用了 base 的 Python 3.9 | `conda activate moe` |
| `tee: results/xxx.log: No such file or directory` | `results/` 不存在 | `mkdir -p results` |
| `mkdir: cannot create directory '/weights-q3-30b'` | 没 `source qwen3-30b/env.sh`，`$WORKDIR` 为空 | 第 2 步 |
| `report … \| sed …` 没有任何输出 | 结果文件不存在（被 sed 滤掉了提示） | 去掉 `\| sed …` 看原始输出 |
| 同一通道越跑越慢、最后 CUDA OOM | （已修）max_tokens 结束后环不停 | 用当前代码；`status` 里看「持有 KV 的请求」 |
| 换绑后输出变成复读 | 换绑不补已生成部分的上下文（未修，见 `VLLM_MIGRATION.md` C5） | 连续测试默认 `--no-rebind` |
| 识别显示 60% | 旧版按「路由去哪」算，没部署的类永远记错 | 用当前代码；看「LR 自己判的类」 |
| 每条通道的第一条请求首 token 多 ~11s | 冷启动 | 当前代码的预热已覆盖每条通道 |
| `start` 卡 60s 显示 0/15 | （已修）ssh 等后台进程 | 用当前 `launch.py`；agent 其实已经起来了，`status` 确认 |
| `sync` 后其他节点代码没了 | 在控制机目录不完整时跑了 `sync`（带 `--delete`） | 先完成第 1 步再 sync |
