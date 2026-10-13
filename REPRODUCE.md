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
| 给 no_robots 两条通道 | 重新规划（`examples/task_deploy.py … --tasks gsm8k=1,mbpp=1,no_robots=2`），换 `PLAN` 后从第 3 步的 `fetch`/`verify` 重来 |

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
