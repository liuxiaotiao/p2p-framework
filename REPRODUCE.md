# 从零复现：Qwen3-30B-A3B + task 选择器 + 15 节点两段式部署

> 适用场景：控制机（wash1）上的 `/home/ubuntu/p2p-framework` 丢了或要换一台控制机。
> Mac 上的 `p2p-framework` 是完整、最新的一份，所有代码、方案、选择器、样本清单都在里面；
> 只有**下载的权重**和 **`results/`** 不在 Mac 上 —— 权重可以重下，结果只能重跑。

集群约定（与 `task/hosts.txt` 一致）：

| 角色 | 节点 |
|---|---|
| 控制机 | wash1 = **N09**（192.168.3.4）—— 它同时是 no_robots 后段的头（层 9–14） |
| 前段（层 1–8，全专家） | N01、N10、N11、N12、N15 |
| 后段（层 9–48，按 task 的专家子集） | gsm8k：N03→N05、N04→N02；mbpp：N06→N07、N13→N14；no_robots：N09→N08 |
| 代码目录（每台都一样） | `/home/ubuntu/p2p-framework` |
| 权重目录（每台都一样） | `/home/ubuntu/p2p-framework/weights-q3-30b` |
| Python（节点与控制机） | `conda activate moe`（`/home/ubuntu/anaconda3/envs/moe`，Python 3.10） |

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
export ADVERTISE=192.168.3.4       # 控制机对节点可见的 IP —— 不设的话 measure/continuous 直接退出
mkdir -p results                   # tee 写日志前目录必须存在
```

一次性（`moe` 环境没被删就不用重装，装过会直接跳过）：

```bash
pip install -r requirements-control.txt      # 含 pyarrow（build_cases 读 parquet 用）、scikit-learn、joblib
tar xzf Delta_30B/delta_30b.tgz -C Delta_30B  # 选择器复核用的 probe 数据 → $PROBE
```

想以后不用每次手敲 `ADVERTISE`：把 `export ADVERTISE=192.168.3.4` 追加到 wash1 上
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
| 调链长上限 | 规划命令加 `--hop-slack N`：每条段的跳数 ≤ 整数下界 hops_min（按段驻留量 ÷ 最大单机可用算）+ N，默认 1；`--hop-slack -1` 不限。求解器搜索时就剪掉超长前缀，防止 beam 交出吃光节点的长链（`--beam-legacy` 下不生效）。用现在的部署参数重出方案，开不开它结果逐字节相同 |
| 给 no_robots 两条通道 | 重新规划（`examples/task_deploy.py … --tasks gsm8k=1,mbpp=1,no_robots=2`），换 `PLAN` 后从第 3 步的 `fetch`/`verify` 重来 |

### 5.5 对照组：严格按文档规划的方案（`qwen3-30b/plan_doc_l8.json`）

规划命令（Mac 或 wash1 上跑都行，结果逐字节相同，`tests/test_hop_cap.py` 有校验）：

```bash
python3 examples/task_deploy.py --data task3 --config qwen3-30b-a3b --phase prefill \
  --tasks gsm8k=1,mbpp=1,no_robots=1 \
  --expert-sets qwen3-30b/expert_sets_90/expert_sets_90_union.json \
  --l0 8 --front-full --reserve-gb 2.5 --hop-slack 0 \
  --rent-policy largest --endpoint-w 0 --avail 0.95 --theta 0.8 \
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
  --mem-prefill-tokens 512 --mem-margin-gb 0 --kv-reserve-gb 2 --endpoint-w 0 \
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
