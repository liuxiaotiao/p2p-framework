# 任务选择器（LR selector）使用说明

> 给后续代码 session 看的参考文档。目的：看一条 prompt 在 Qwen3-30B-A3B **前 8 层**选了哪些专家，
> 判断它属于哪个任务（gsm8k / math / mbpp / humaneval / no_robots），以便之后按任务选专家集合做裁剪。
>
> 最后更新：2026-10-06

---

## 1. 文件清单

所有代码以 Mac 上的 `verify/` 为准（`/Users/qingzheguo/Desktop/cc workspace/verify/`），
用到集群时拷到同一个目录（Delta 上是 `/u/qguo1/qwen3/`）。

| 文件 | 作用 | md5 |
|---|---|---|
| `task_selector.py` | **推理入口**：predict / eval / offline 三个子命令，也可以当模块 import | `db683f7c660ca66485800fc24f83e96f` |
| `analyze_30b.py` | **训练入口**：评估（交叉验证）并用 `--save-selector` 存出选择器 | `4e6a017db87768ed98f6e2252a741ac5` |
| `moe_prefill_probe.py` | 采集训练特征；prompt 构造（`prompt_segments`）、body 定位（`PromptEncoder`）都在这里 | `522929cf31edf5107a843529a2cc1e6c` |
| `moe_expert_activation.py` | `find_decoder_stack`、`cfg_get` | `0d80d545bcc90fdbc371fc7f33cd76bd` |
| `expert_mask.py` | `find_gates`、`_split_router_out`（识别 router 输出） | `7e930ef30249bbdc7e5e47c316cc8f14` |
| `nr_prune.py` | `RoutingCounter`（按样本、按段计数），No Robots 的单轮过滤 `load_single_turn` | `e1758834fbadef2968c2421b61f19d8c` |

数据和模型（Delta）：

| 内容 | 路径 |
|---|---|
| 模型 | `/work/nvme/bgfc/qguo1/models/Qwen3-30B-A3B` |
| 数据集（`save_to_disk`） | `/work/nvme/bgfc/qguo1/hf_data/{gsm8k,math,mbpp,humaneval,no_robots}` |
| **已训练的选择器** | `/work/nvme/bgfc/qguo1/probe_30b_ref20/selector_body_L8.joblib`（Mac 上副本：`verify/Delta_30B/probe_30b_ref20/selector_body_L8.joblib`） |
| **同一个选择器的纯参数版（推荐）** | Mac：`verify/Delta_30B/probe_30b_ref20/selector_body_L8_params.npz`（md5 `75bf1e32a7d40b9067e548b5f123e0fd`），用前拷到集群同目录 |
| 训练特征（5 个任务各 20%） | `/work/nvme/bgfc/qguo1/probe_30b_ref20/prefill_Qwen3-30B-A3B/{task}_prefill_persample.npz` |

环境：transformers 5.x、torch、scikit-learn、joblib、accelerate、datasets、tqdm。
Delta 计算节点没外网，先 `export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1`。

---

## 2. 选择器文件里有什么

同一个模型有两种格式，`task_selector.load_selector()` 两种都能读，结果完全一致
（1486 条样本上概率最大差 1e-8，预测类别全部相同）：

| 文件 | 依赖 | 说明 |
|---|---|---|
| `selector_body_L8_params.npz` | 只要 numpy | **推荐**。存的是标准化均值/方差、LR 系数和截距，用纯 numpy 算 softmax，没有版本兼容问题 |
| `selector_body_L8.joblib` | scikit-learn | 在 Delta 上用 **sklearn 1.7.2** 存的 pickle。换 sklearn 版本加载会出 `InconsistentVersionWarning`（1.8.0 下实测结果正确，但 sklearn 不保证跨版本兼容） |

`_params.npz` 的键：`mean`、`scale`（各 1024 维）、`coef`（5×1024）、`intercept`（5）、`classes`、
`n_layers`、`n_experts`。纯 numpy 预测就是：

```python
z = (x - mean) / scale                 # x: (N, 1024)，见第 3 节
logit = z @ coef.T + intercept
prob = softmax(logit, axis=1)          # 列顺序 = classes
```

`joblib.load(path)` 得到一个 dict：

| 键 | 值 |
|---|---|
| `pipeline` | sklearn `Pipeline(StandardScaler, LogisticRegression(max_iter=3000))` |
| `classes` | `["gsm8k", "math", "mbpp", "humaneval", "no_robots"]`，`predict` 返回的下标按这个顺序 |
| `feature` | `"prefill_counts_body"` |
| `n_layers` | `8`（只用第 0–7 层） |
| `n_experts` | `128` |
| `trained_on` | 每类训练样本数：gsm8k 264 / math 1000 / mbpp 100 / humaneval 33 / no_robots 89 |
| `how_to_use` | 一句话说明 |

---

## 3. 输入特征的精确定义（必须和训练时逐字一致）

对每一条 prompt：

1. **构造 prompt**：`PromptEncoder(tok, ("full","body"), (), enable_thinking=False)`，
   `text, spans = encoder.build(segs)`。`segs` 是 `[(文本, 是否模板), ...]`：
   任务指令模板（如 gsm8k 的 "Solve the following math problem..."）标 `True`，题目本身标 `False`。
   chat 模板（`<|im_start|>user` … `<think>\n\n</think>\n\n`）由 encoder 自动加上。非思考模式。
2. **分词**：tokenizer 用 `padding_side="left", use_fast=True`，`return_offsets_mapping=True`。
3. **body 掩码**：token 的字符区间和 `spans` 有重叠、且不是特殊 token / padding（offset 非 `(0,0)`）→ 算 body。
   模板 token、特殊 token、padding 都不算。
4. **前向**：`base = find_decoder_stack(model)`，
   `position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)`，
   `base(input_ids=..., attention_mask=..., position_ids=...)`。只做 prefill，不生成。
5. **计数**：在第 0–7 层每层的 router 上挂 hook，取每个 token 选中的 8 个专家下标
   （Qwen3MoeTopKRouter 返回 `(router_logits, router_scores, router_indices)`，用第三个）。
   `counts[l, e]` = 第 `l` 层有多少个 **body token** 选中了专家 `e`。
   每层总和 = body token 数 × 8。**只数次数，不用门控权重。**
6. **逐层归一化 + 展平**：
   ```python
   c = counts[:8].astype(np.float32)               # (8, 128)
   x = (c / c.sum(axis=1, keepdims=True)).reshape(-1)   # 1024 维，每层 128 个数和为 1
   ```
7. **预测**：`sel["pipeline"].predict_proba(x[None])` → 5 个类别的概率。

`task_selector.features()` 实现的就是第 6 步；第 1–5 步在 `task_selector.body_counts()` 里，
直接复用 `moe_prefill_probe.py` 的 `PromptEncoder.encode_batch` 和 `make_batches`。

**已验证**：在小模型上，`task_selector.py` 只加载前 4 层算出的特征，和 `moe_prefill_probe.py`
采集时存的 `counts_body[:, :4]` 逐位相同（191 条样本）。

---

## 4. 怎么训练（复现已有的选择器）

```bash
# 1) 采 5 个任务各 20% 的 prefill 路由（只 prefill，几分钟）
python moe_prefill_probe.py --model $M \
  --datasets gsm8k math mbpp humaneval no_robots \
  --data-dir /work/nvme/bgfc/qguo1/hf_data --variants full body --subset reference \
  --output-dir /work/nvme/bgfc/qguo1/probe_30b_ref20/prefill_Qwen3-30B-A3B

# 2) 交叉验证评估 + 训练并保存（用 body 特征、前 8 层）
python analyze_30b.py --prefill-dir /work/nvme/bgfc/qguo1/probe_30b_ref20/prefill_Qwen3-30B-A3B \
  --datasets gsm8k math mbpp humaneval no_robots --skip correctness --train-on cv \
  --save-selector /work/nvme/bgfc/qguo1/probe_30b_ref20/selector_body_L8.joblib --selector-layers 8
```

- 20% = `reference_indices(n, 0.2, 42)`，和裁剪实验拟合专家集合用的是同一批样本。
- `--datasets` 的顺序决定 `classes` 的顺序。
- 采集默认跳过已完成的数据集；换配置要换输出目录。

---

## 5. 怎么用

下面都假设：

```bash
M=/work/nvme/bgfc/qguo1/models/Qwen3-30B-A3B
SEL=/work/nvme/bgfc/qguo1/probe_30b_ref20/selector_body_L8_params.npz   # 或 .joblib，两者等价
D=/work/nvme/bgfc/qguo1/hf_data
```

默认只实例化模型前 8 层（显存约为完整模型的 1/6，约 10GB）。加载时打印的一大段
`UNEXPECTED` 权重是被跳过的后面那些层，正常。

### 5.1 predict：给新 prompt 预测

输入 jsonl，每行一条，三种写法：

```json
{"id": 1, "text": "Janet's ducks lay 16 eggs per day..."}
{"id": 2, "segments": [["Solve the following math problem. ...Problem: ", true], ["题目原文", false]]}
{"id": 3, "text": "Write a short poem about autumn.", "system": "Be brief."}
```

- `text`：整条 user 消息都算 body
- `segments`：和训练时一样，`true` = 模板（不计入特征），`false` = body
- `system`（可选）：当作模板段放在最前面，不计入 body

```bash
python task_selector.py predict --model $M --selector $SEL \
  --prompts in.jsonl --out pred.jsonl --min-prob 0.9
```

输出每行：`{"id": ..., "pred": "gsm8k", "prob": {"gsm8k": 0.99, "math": 0.01, ...}}`。
最大概率低于 `--min-prob` 时 `pred` 为 `"unknown"`（应该用完整模型、不裁剪）。

### 5.2 eval：在数据集上验证

```bash
# 样本外 80%，带任务指令模板（和训练一致）
python task_selector.py eval --model $M --selector $SEL --data-dir $D \
  --datasets gsm8k math mbpp humaneval no_robots --subset heldout

# 同上，但去掉任务指令模板、只给原题（模拟真实用户输入）
python task_selector.py eval --model $M --selector $SEL --data-dir $D \
  --datasets gsm8k math mbpp humaneval no_robots --subset heldout --raw
```

输出每类召回率、平衡准确率、混淆矩阵、最大概率分布。`--out x.npz` 可保存特征和预测。
`--subset reference` 是训练样本，数字必然偏高，只用于自检。

### 5.3 offline：对已采集的 npz 预测（不加载模型）

```bash
python task_selector.py offline --selector $SEL \
  --prefill-dir <moe_prefill_probe 的输出目录> \
  --datasets gsm8k math mbpp humaneval no_robots --subset heldout
```

npz 必须带 `counts_body`（采集时加 `--variants full body`）。

### 5.4 在自己的代码里调用

```python
import argparse, numpy as np
import task_selector as ts

sel = ts.load_selector(SEL)

# 只要特征已经有了（(N, ≥8, 128) 的 body 计数）：
pred, P = ts.predict(sel, counts, min_prob=0.9)

# 从原始 prompt 开始：
args = argparse.Namespace(model=M, dtype="bfloat16", device_map="auto", attn_impl="sdpa",
                          trust_remote_code=False, full_model=False)
cfg, tok, model = ts.load_first_layers(args, sel["n_layers"])
n_exp, top_k = ts._model_meta(cfg, sel)
seg_lists = [[("Janet's ducks lay 16 eggs per day...", False)]]   # 每条 = [(文本, 是否模板), ...]
enc = ts.make_encoder(tok, "".join(s for s, _ in seg_lists[0]))   # 必须先调用，body_counts 依赖它
recs = ts.build_recs(enc, seg_lists)
counts = ts.body_counts(model, tok, recs, sel["n_layers"], n_exp, top_k,
                        max_tokens=16384, max_seqs=32)             # (N, 8, 128)
pred, P = ts.predict(sel, counts, min_prob=0.9)
```

- `make_encoder` 会用第一条样本检测 chat 模板是否 trim 内容两端空白，并把 encoder 存在模块级缓存里，
  `body_counts` 从缓存取。**换 tokenizer 时要重新调用。**
- 如果代码里已经加载了完整模型（比如要接着生成），可以直接把完整模型传给 `body_counts`：
  它只取前 `n_layers` 个 router 计数，结果一样，只是前向会跑完全部 48 层。

---

## 6. 必须遵守的约束

| 约束 | 违反的后果 |
|---|---|
| 同一个模型（Qwen3-30B-A3B）、bf16、`enable_thinking=False`、同一个 chat 模板 | 路由不同，特征对不上 |
| **只数 body token**，不能把模板 token 算进去 | 训练时用的是 body；用 full 特征会严重偏离 |
| 只用前 `n_layers`（8）层，每层**单独**归一化后再展平 | 维度或尺度对不上，预测无意义 |
| 计数用"被选中次数"，不是门控权重 | 同上 |
| 左 padding + `position_ids = cumsum(mask) - 1` | 数值和训练不一致（影响一般很小，但别改） |
| 不同 batch 组成会带来 bf16 级的数值噪声 | 可以忽略，量级远小于类间差异 |

---

## 7. 当前验证结果（20% 内部 5 折交叉验证）

| 特征 | 方法 | 平衡准确率 |
|---|---|---|
| full（含模板） | 逻辑回归 | 0.997 |
| **body** | **逻辑回归，全部 48 层** | **0.987** |
| body | 逻辑回归，**前 8 层**（= 已保存的选择器） | **0.976** |
| body | 最近质心 | 0.952 |

- 每类召回率（body，48 层）：gsm8k 0.966 / math 0.981 / mbpp 1.000 / humaneval 1.000 / no_robots 0.989
- 29 个错误里 28 个是 **gsm8k ↔ math** 互相混淆
- 只用第 0 层就有 0.960：分类主要依赖表层词汇和格式（第 0 层路由基本由 token 身份决定）

---

## 8. 已知限制 / 还没做的

1. **没在 80% 样本外测过**：跑 5.2 的第一条命令即可（几分钟）。
2. **模板上下文风险**：训练时 gsm8k/math/mbpp/humaneval 的 prompt 都带任务指令模板。body token 虽不计入
   模板，但它们的路由是在"前面有模板"的上下文里算的（每层 MoE 之前都有注意力）。部署到没有模板的
   原始用户输入时，准确率可能下降。跑 5.2 的 `--raw` 验证；若明显下降，需要用无模板数据重新训练。
3. **没有"都不是"这一类**：只会在 5 类里选。用 `--min-prob` 设门槛。
4. **No Robots 里的编程题几乎没测到**：训练集 89 条 No Robots 里只有约 2 条 Coding，
   "No Robots 编程题 vs mbpp/humaneval" 这条边界基本没被检验。
5. **小类置信区间宽**：humaneval 训练 33 条，全对时 95% 召回率下限约 0.91。
6. **还没有纯词汇基线**（TF-IDF + LR 在同样的 body 文本上），所以不能断言路由比词本身多提供了信息。
7. **端到端流程还没写**：选择器 → 选专家集合 → 带逐样本 mask 生成。现有 `eval_benchmarks.py` /
   `nr_prune.py` 只支持整轮一个固定 mask。建议两遍实现：第一遍只跑前 8 层拿特征，第二遍带逐样本
   mask 生成；对比 oracle（每条用真实任务的集合）和不裁剪。

---

## 9. 和专家集合的配合

专家集合在 `verify/expert_sets_90/expert_sets_90_union.json`（90% 阈值、prefill ∪ decode、20% 拟合）：

```python
import json
S = json.load(open("expert_sets_90/expert_sets_90_union.json"))["tasks"]
allowed = S[pred]["union"]          # 48 个列表，第 l 层允许的专家编号
```

- 目前只有 **gsm8k、mbpp、no_robots**。humaneval 的集合在 OSC 上的 `sets3.json` 里；
  **math 没有集合**（它的 decode 数据有偏，被排除了）。
- 预测为 math、unknown、或没有集合的任务时，应回退到完整模型（不裁剪）。
- 集合来源不完全一致：gsm8k/mbpp 来自 OSC A100、decode 是模型自己生成；no_robots 来自
  Delta H200、decode 是对人写答案的 teacher forcing。
- 裁剪时前 8 层不屏蔽（`--mask-skip-first 8`），正好对应选择器只用前 8 层做判断。
- gsm8k 和 math 的集合在 70% 阈值下 Jaccard 只有 0.39，所以选择器把 gsm8k 判成 math 的代价不小。
