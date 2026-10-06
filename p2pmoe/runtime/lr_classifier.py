"""task 识别：前段出口的逻辑回归（LR）分类器。

与 `identify.HistogramClassifier` 并列的第二种识别器，插在同一个位置 ——
前段 tail 聚齐前 L₀ 层的统计之后、问协调器要后段之前（文档 II.5）。区别只在
输入与判别器：

    HistogramClassifier  各层直方图**求和**成 n_experts 维 → 与参考画像算余弦
    LRClassifier         **逐层**统计拼成 L×n_experts 维   → 已训练好的 LR 打分

LR 是外面训练好的，这里只负责三件事：

1. **装进来** —— `load_lr_params` 认 sklearn 的 pickle/joblib（含
   `Pipeline([Scaler, LogisticRegression])`）、npz、JSON 三种格式，在控制机上
   全部化成纯 numpy 的 (W, b, 仿射预处理)。节点**不需要 sklearn**。
2. **特征要和训练时一模一样** —— `FeatureSpec` 描述「哪几层、哪种统计、怎么
   归一、怎么排」。默认口径对齐 task/*.csv：层号 0-based，`count` 即
   `prefill_count`，`mean_weight` 即 `prefill_mean_weight`。描述不了的，用
   `feature_fn` 直接给一段 Python（源码随清单下发，节点 exec）。
3. **在线判别** —— `predict_stats(MoEStats) → Verdict`，与直方图识别器同一个
   置信三区，所以协调器、换绑、通道二一行都不用改。

特征维度对不上是最常见的错，所以在**控制机装载时**就核对，而不是等第一条
请求到 tail 才炸。对不上时会把「这个维度对应几层」算出来告诉你。
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .identify import Verdict
from .model import LayerStat, MoEStats

__all__ = ["FeatureSpec", "LRClassifier", "load_lr_params", "make_classifier",
           "parse_class_map", "layer_matrix"]

_STATS = ("count", "weight", "mean_weight", "freq", "weight_freq")
_NORMS = ("none", "per_token", "per_layer")
_VNORMS = ("none", "l1", "l2")


# --------------------------------------------------------------------------- #
@dataclass
class FeatureSpec:
    """LR 的输入怎么从逐层统计拼出来。**必须与训练时一致。**"""

    layers: list[int] | None = None
    """用哪几层，按 `layer_base` 的口径编号。None = 前段全部层（1..L₀）。"""
    layer_base: int = 0
    """层号从几开始。task/*.csv 与 HF 的 `model.layers.{i}` 都是 0；
    规划器内部是 1。默认 0 —— 「第 0 层」就是 checkpoint 的第一层。"""
    stats: list[str] = field(default_factory=lambda: ["count"])
    """每层取哪些统计，多个就拼起来：

    * `count`       —— 该专家进 top-k 的次数（= csv 的 prefill_count）
    * `weight`      —— 门控权重之和
    * `mean_weight` —— weight / count，没进过 top-k 的记 0（= csv 的 prefill_mean_weight）
    * `freq`        —— count / 该层 token 数（每 token 被选中的频率，与 prompt 长度无关）
    * `weight_freq` —— weight / 该层 token 数
    """
    normalize: str = "none"
    """逐层归一（只作用于 count/weight 这类外延量）：
    `none` | `per_token`（除以 token 数）| `per_layer`（除以本层总和，变成分布）。"""
    log1p: bool = False
    """归一之后、拼接之前取 log(1+x)。"""
    vector_norm: str = "none"
    """拼完之后整条向量再归一：`none` | `l1` | `l2`。"""
    tokens: str = "all"
    """数哪些 prompt token：`all`（整条 prompt，含对话模板）| `body`（只数用户内容本身，
    不含对话模板、任务指令模板、特殊 token —— task 选择器 selector_body_L8 就是这个
    口径）。`body` 要求协调器把 body 的 token 区间随 prefill 下发（`Coordinator.count_body`），
    节点只对这些位置计数。"""
    layout: str = "layer_major"
    """`layer_major`：[层0 的 stat1(E), stat2(E)] [层1 ...] ……
    `stat_major` ：[stat1 的 层0(E), 层1(E) …] [stat2 …]。
    单个 stat 时两者相同 —— 都是「层 0 的 E 个专家，接着层 1 的 E 个……」。"""

    def __post_init__(self) -> None:
        bad = [s for s in self.stats if s not in _STATS]
        if bad:
            raise ValueError(f"不认识的统计 {bad}；可选 {list(_STATS)}")
        if self.normalize not in _NORMS:
            raise ValueError(f"normalize={self.normalize!r}；可选 {list(_NORMS)}")
        if self.vector_norm not in _VNORMS:
            raise ValueError(f"vector_norm={self.vector_norm!r}；可选 {list(_VNORMS)}")
        if self.layout not in ("layer_major", "stat_major"):
            raise ValueError(f"layout={self.layout!r}；可选 layer_major / stat_major")
        if self.tokens not in ("all", "body"):
            raise ValueError(f"tokens={self.tokens!r}；可选 all / body")

    @classmethod
    def from_any(cls, x: "FeatureSpec | Mapping | str | Path | None") -> "FeatureSpec":
        """接受 FeatureSpec / dict / JSON 字符串 / JSON 文件路径 / None。"""
        if x is None:
            return cls()
        if isinstance(x, FeatureSpec):
            return x
        if isinstance(x, (str, Path)):
            sx = str(x).strip()
            d = json.loads(sx) if sx.startswith("{") else json.loads(
                Path(sx).read_text(encoding="utf-8"))
        else:
            d = dict(x)
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        extra = sorted(set(d) - set(known))
        if extra:
            raise ValueError(f"特征描述里有不认识的键 {extra}；"
                             f"可用的键 {sorted(cls.__dataclass_fields__)}")
        if isinstance(known.get("stats"), str):
            known["stats"] = [known["stats"]]
        return cls(**known)

    # -- 层号换算 ------------------------------------------------------------ #
    def planner_layers(self, l0: int) -> list[int]:
        """特征用到的层，换成规划器口径（1-based）。"""
        if self.layers is None:
            return list(range(1, l0 + 1))
        return [int(l) - self.layer_base + 1 for l in self.layers]

    def expected_dim(self, l0: int, n_experts: int) -> int:
        return len(self.planner_layers(l0)) * n_experts * len(self.stats)


def layer_matrix(
    layers: Mapping[int, LayerStat], want: Sequence[int], n_experts: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把逐层统计按 `want`（规划器口径）的顺序排成 count[L,E]、weight[L,E]、tokens[L]。

    缺层就抛 —— 缺层意味着捎带链路上有一跳没把统计带过来，悄悄补零会让
    LR 在一个它从没见过的输入上打出一个看似正常的分数。
    """
    missing = [l for l in want if l not in layers]
    if missing:
        have = sorted(layers)
        raise KeyError(f"缺第 {missing} 层的统计（规划器口径）；收到的是 {have or '无'}")
    cnt = np.stack([np.asarray(layers[l].count, dtype=np.float64) for l in want])
    wgt = np.stack([np.asarray(layers[l].mass, dtype=np.float64) for l in want])
    tok = np.array([float(layers[l].n_tokens) for l in want])
    if cnt.shape[1] != n_experts:
        raise ValueError(f"统计是 {cnt.shape[1]} 个专家/层，模型是 {n_experts}")
    return cnt, wgt, tok


def _features(spec: FeatureSpec, cnt: np.ndarray, wgt: np.ndarray,
              tok: np.ndarray) -> np.ndarray:
    t = np.maximum(tok, 1.0)[:, None]
    blocks = []
    for s in spec.stats:
        if s == "count":
            m = cnt.copy()
        elif s == "weight":
            m = wgt.copy()
        elif s == "mean_weight":
            m = np.divide(wgt, cnt, out=np.zeros_like(wgt), where=cnt > 0)
        elif s == "freq":
            m = cnt / t
        else:  # weight_freq
            m = wgt / t
        if s in ("count", "weight"):
            if spec.normalize == "per_token":
                m = m / t
            elif spec.normalize == "per_layer":
                z = m.sum(axis=1, keepdims=True)
                m = np.divide(m, z, out=np.zeros_like(m), where=z > 0)
        if spec.log1p:
            m = np.log1p(m)
        blocks.append(m)                                   # 每个 [L, E]
    if spec.layout == "layer_major":
        x = np.concatenate(blocks, axis=1).reshape(-1)     # 层0[s1 s2] 层1[s1 s2]…
    else:
        x = np.concatenate([b.reshape(-1) for b in blocks])
    if spec.vector_norm == "l1":
        x = x / max(float(np.abs(x).sum()), 1e-12)
    elif spec.vector_norm == "l2":
        x = x / max(float(np.linalg.norm(x)), 1e-12)
    return x


# --------------------------------------------------------------------------- #
# 装载：sklearn / npz / json → 纯 numpy
# --------------------------------------------------------------------------- #
def _affine_of(step: Any) -> tuple[np.ndarray | float, np.ndarray | float] | None:
    """sklearn 的仿射预处理 → (m, s)，使 step(x) = (x − m) / s。不认识就 None。"""
    name = type(step).__name__
    if name == "StandardScaler":
        m = step.mean_ if getattr(step, "mean_", None) is not None else 0.0
        s = step.scale_ if getattr(step, "scale_", None) is not None else 1.0
        return np.asarray(m, dtype=np.float64), np.asarray(s, dtype=np.float64)
    if name == "MinMaxScaler":
        sc = np.asarray(step.scale_, dtype=np.float64)
        sc = np.where(sc == 0, 1.0, sc)
        return -np.asarray(step.min_, dtype=np.float64) / sc, 1.0 / sc
    if name == "MaxAbsScaler":
        return 0.0, np.asarray(step.scale_, dtype=np.float64)
    return None


def _is_identity(step: Any) -> bool:
    return step is None or step == "passthrough" or (
        type(step).__name__ == "FunctionTransformer" and getattr(step, "func", 0) is None)


def _from_sklearn(obj: Any) -> dict:
    steps = [s for _, s in obj.steps] if hasattr(obj, "steps") else [obj]
    est = steps[-1]
    if not hasattr(est, "coef_"):
        raise ValueError(f"最后一步是 {type(est).__name__}，不是线性模型（没有 coef_）")
    shift: Any = 0.0
    scale: Any = 1.0
    for st in steps[:-1]:
        if _is_identity(st):
            continue
        aff = _affine_of(st)
        if aff is None:
            raise ValueError(
                f"Pipeline 里有一步 {type(st).__name__}，它不是仿射变换，没法化成 "
                f"(x−m)/s 下发给节点。支持 StandardScaler / MinMaxScaler / "
                f"MaxAbsScaler。其它预处理请写进 --lr-feature-fn，"
                f"或者导出成 JSON（见 --help 里的格式）")
        m, s = aff
        # 复合：((x − shift)/scale − m)/s = (x − (shift + m·scale)) / (scale·s)
        shift = shift + np.asarray(m) * np.asarray(scale)
        scale = np.asarray(scale) * np.asarray(s)

    coef = np.atleast_2d(np.asarray(est.coef_, dtype=np.float64))
    icpt = np.atleast_1d(np.asarray(est.intercept_, dtype=np.float64))
    classes = list(est.classes_)
    if coef.shape[0] == 1 and len(classes) == 2:
        proba = "binary"
    else:
        # sklearn predict_proba 的口径：ovr 时逐类 sigmoid 再归一，否则 softmax。
        # 1.5 之后 multi_class 参数废弃（值为 "deprecated"），一律 multinomial。
        mc = getattr(est, "multi_class", "auto")
        solver = getattr(est, "solver", "lbfgs")
        ovr = mc == "ovr" or (mc == "auto" and solver == "liblinear")
        proba = "ovr" if ovr else "softmax"
    out = {"classes": classes, "coef": coef, "intercept": icpt, "proba": proba}
    if not (np.isscalar(shift) and shift == 0.0):
        out["shift"] = np.broadcast_to(np.asarray(shift, dtype=np.float64),
                                       (coef.shape[1],)).copy()
    if not (np.isscalar(scale) and scale == 1.0):
        out["scale"] = np.broadcast_to(np.asarray(scale, dtype=np.float64),
                                       (coef.shape[1],)).copy()
    return out


_SELECTOR_FEATURES = {
    # analyze_30b.py --save-selector 的 "feature" 字段 → 本模块的特征描述。
    # 口径见 selector/SELECTOR_USAGE.md 第 3 节：前 n_layers 层、只数选中次数、
    # 每层单独归一成分布再展平（层优先）。
    "prefill_counts_body": dict(stats=["count"], normalize="per_layer", tokens="body"),
    "prefill_counts_full": dict(stats=["count"], normalize="per_layer", tokens="all"),
}


def _selector_meta(d: Mapping, out: dict) -> dict:
    """task 选择器自带的元数据：类别名、特征口径、层数、专家数。

    `selector_body_L8.joblib` 是 {"pipeline", "classes", "feature", "n_layers",
    "n_experts", ...}；`_params.npz` 是 {mean, scale, coef, intercept, classes,
    n_layers, n_experts}，没有 feature 字段 —— 与 task_selector.load_selector
    一样，按 body 口径处理。
    """
    names = d.get("classes")
    cls = out["classes"]
    # pipeline 的 classes_ 是 0..n-1 的下标，名字在外层 dict 里
    if (names is not None and len(names) == len(cls)
            and all(isinstance(c, (int, np.integer)) for c in cls)
            and sorted(int(c) for c in cls) == list(range(len(cls)))):
        out["classes"] = [str(names[int(c)]) for c in cls]
    if "n_layers" in d:
        n = int(np.asarray(d["n_layers"]))
        feat = str(np.asarray(d.get("feature", "prefill_counts_body")))
        if feat not in _SELECTOR_FEATURES:
            raise ValueError(f"选择器的特征口径 {feat!r} 不认识；"
                             f"认识的是 {sorted(_SELECTOR_FEATURES)}")
        if "features" not in out:
            out["features"] = {"layers": list(range(n)), "layer_base": 0,
                               **_SELECTOR_FEATURES[feat]}
        out["meta"] = {"feature": feat, "n_layers": n,
                       "n_experts": int(np.asarray(d["n_experts"]))
                       if "n_experts" in d else None,
                       "trained_on": d.get("trained_on")}
    return out


def _from_mapping(d: Mapping) -> dict:
    # 常见的「自己打包」形式：{"model": clf, "scaler": sc, "features": {...}}
    for k in ("model", "clf", "lr", "classifier", "estimator", "pipeline"):
        if k in d and (hasattr(d[k], "coef_") or hasattr(d[k], "steps")):
            obj = d[k]
            sc = d.get("scaler")
            if sc is not None and hasattr(obj, "coef_"):
                from types import SimpleNamespace
                obj = SimpleNamespace(steps=[("scaler", sc), ("lr", obj)])
            out = _from_sklearn(obj)
            for extra in ("features", "feature_fn", "classes_map"):
                if extra in d:
                    out[extra] = d[extra]
            return _selector_meta(d, out)
    need = ("classes", "coef", "intercept")
    if not all(k in d for k in need):
        raise ValueError(f"认不出模型：需要 sklearn 对象，或含 {need} 的字典；"
                         f"收到的键 {sorted(d)[:12]}")
    coef = np.atleast_2d(np.asarray(d["coef"], dtype=np.float64))
    out = {"classes": list(d["classes"]), "coef": coef,
           "intercept": np.atleast_1d(np.asarray(d["intercept"], dtype=np.float64))}
    out["proba"] = d.get("proba") or ("binary" if coef.shape[0] == 1 else "softmax")
    sc = d.get("scaler") or {}
    for src, dst in (("mean", "shift"), ("shift", "shift"), ("scale", "scale")):
        v = sc.get(src, d.get(src)) if isinstance(sc, Mapping) else d.get(src)
        if v is not None:
            out[dst] = np.asarray(v, dtype=np.float64)
    for extra in ("features", "feature_fn", "classes_map"):
        if extra in d:
            out[extra] = d[extra]
    return _selector_meta(d, out)


def load_lr_params(path: str | Path) -> dict:
    """读一个训练好的 LR，化成 {classes, coef[C,D], intercept[C], proba, shift?, scale?}。

    支持：
    * `.pkl` / `.pickle` / `.joblib` —— sklearn 的 LogisticRegression(CV)，或以它
      结尾、前面是 Standard/MinMax/MaxAbsScaler 的 Pipeline，或
      `{"model": clf, "scaler": sc}` 这样的字典。需要控制机上有 sklearn。
    * `.npz` —— 键 coef, intercept, classes，可选 mean, scale, proba。
    * `.json` —— 同上的键；可选 "scaler": {"mean": [...], "scale": [...]}，
      "features": FeatureSpec 的字段，"feature_fn": "file.py:func"。
    """
    p = Path(path)
    suf = p.suffix.lower()
    if suf == ".json":
        raw = _from_mapping(json.loads(p.read_text(encoding="utf-8")))
    elif suf == ".npz":
        z = np.load(p, allow_pickle=True)
        d = {k: (z[k].tolist() if k in ("classes", "proba", "feature") else z[k])
             for k in z.files}
        raw = _from_mapping(d)
    else:
        try:
            import joblib
            obj = joblib.load(p)
        except ImportError:
            with open(p, "rb") as f:
                obj = pickle.load(f)
        raw = _from_mapping(obj) if isinstance(obj, Mapping) else _from_sklearn(obj)
    raw["classes"] = [c.item() if hasattr(c, "item") else c for c in raw["classes"]]
    raw.setdefault("meta", None)
    C, D = raw["coef"].shape
    nC = len(raw["classes"])
    if raw["proba"] == "binary":
        if nC != 2 or C != 1:
            raise ValueError(f"二分类模型要求 coef 1 行、2 个类，实际 {C} 行、{nC} 个类")
    elif C != nC:
        raise ValueError(f"coef 有 {C} 行，类别却有 {nC} 个")
    if raw["intercept"].shape[0] != C:
        raise ValueError(f"intercept 有 {raw['intercept'].shape[0]} 个，coef 有 {C} 行")
    for k in ("shift", "scale"):
        if k in raw and raw[k].shape != (D,):
            raise ValueError(f"预处理 {k} 的长度 {raw[k].shape} 与特征维度 {D} 不符")
    return raw


def parse_class_map(s: str | None, classes: Sequence) -> dict[str, str]:
    """`--lr-classes` → {模型类别(字符串化): task 名}。

    两种写法：`0=mbpp,1=gsm8k,2=math`（按标签），或 `mbpp,gsm8k,math`（按
    classes_ 的顺序）。不给时类别本身就当 task 名用。
    """
    if not s:
        return {str(c): str(c) for c in classes}
    parts = [x.strip() for x in s.split(",") if x.strip()]
    if all("=" in x for x in parts):
        m = {k.strip(): v.strip() for k, v in (x.split("=", 1) for x in parts)}
        unknown = sorted(set(m) - {str(c) for c in classes})
        if unknown:
            raise ValueError(f"--lr-classes 里的 {unknown} 不是模型的类别；"
                             f"模型的类别是 {list(classes)}")
        return {str(c): m.get(str(c), str(c)) for c in classes}
    if len(parts) != len(classes):
        raise ValueError(f"--lr-classes 给了 {len(parts)} 个名字，模型有 "
                         f"{len(classes)} 个类 {list(classes)}")
    return {str(c): n for c, n in zip(classes, parts)}


def _compile_fn(src: str, name: str) -> Callable:
    ns: dict = {"np": np, "numpy": np}
    exec(compile(src, f"<lr-feature-fn:{name}>", "exec"), ns)
    if name not in ns or not callable(ns[name]):
        raise ValueError(f"特征函数文件里没有可调用的 {name}")
    return ns[name]


# --------------------------------------------------------------------------- #
class LRClassifier:
    """前段 tail 上的 LR 识别器。接口与 HistogramClassifier 对齐：

    * `predict_stats(MoEStats) -> Verdict`（节点调这个）
    * `to_wire()` / `from_wire()`（随清单下发）
    """

    kind = "lr"

    def __init__(
        self,
        *,
        classes: Sequence[str],
        coef: np.ndarray,
        intercept: np.ndarray,
        proba: str,
        spec: FeatureSpec,
        l0: int,
        n_experts: int,
        priors: Mapping[str, float],
        shift: np.ndarray | None = None,
        scale: np.ndarray | None = None,
        tau_hi: float = 0.55,
        tau_lo: float = 0.40,
        feature_fn: tuple[str, str] | None = None,
        source: str = "",
        labels: Sequence[str] | None = None,
    ):
        self.classes = [str(c) for c in classes]
        """每个输出对应的**池**（task 名）。可以重复：`--lr-classes humaneval=mbpp`
        把 humaneval 并进 mbpp 池，两者的概率相加。"""
        self.labels = [str(c) for c in (labels if labels is not None else classes)]
        """模型自己的类别名（重命名之前）。原始概率按它上报。"""
        self.coef = np.asarray(coef, dtype=np.float64)
        self.intercept = np.asarray(intercept, dtype=np.float64)
        self.proba = proba
        self.spec = spec
        self.l0 = int(l0)
        self.n_experts = int(n_experts)
        self.priors = {k: float(v) for k, v in priors.items()}
        self.shift = None if shift is None else np.asarray(shift, dtype=np.float64)
        self.scale = None if scale is None else np.asarray(scale, dtype=np.float64)
        self.tau_hi, self.tau_lo = float(tau_hi), float(tau_lo)
        self.feature_fn = feature_fn
        self._fn = _compile_fn(*feature_fn) if feature_fn else None
        self.source = source
        self.layers = spec.planner_layers(self.l0)
        if len(self.labels) != len(self.classes):
            raise ValueError("labels 与 classes 长度不一致")
        self._check()
        self.pools = sorted(set(self.classes))

    # -- 自检（在控制机上跑，不等请求到节点才炸）----------------------------- #
    def _check(self) -> None:
        bad = [l for l in self.layers if not 1 <= l <= self.l0]
        if bad:
            base = self.spec.layer_base
            raise ValueError(
                f"LR 用到了第 {[l - 1 + base for l in bad]} 层（{base}-based），"
                f"但前段只有 {base}..{self.l0 - 1 + base} 共 {self.l0} 层 —— "
                f"tail(f) 拿不到前段以外的统计。要么重新规划让 L₀ ≥ "
                f"{max(self.layers)}（examples/task_deploy.py --min-l0 {max(self.layers)}），"
                f"要么用只含前 {self.l0} 层的模型")
        D = self.coef.shape[1]
        if self._fn is None:
            want = self.spec.expected_dim(self.l0, self.n_experts)
            if D != want:
                E, k = self.n_experts, len(self.spec.stats)
                hint = ""
                if D % (E * k) == 0:
                    n = D // (E * k)
                    hint = (f"\n  {D} = {n} 层 × {E} 专家 × {k} 种统计 —— 模型像是用 "
                            f"{n} 层训练的，而当前取的是 {len(self.layers)} 层"
                            f"（L₀={self.l0}）。用 --lr-features 指定 layers，"
                            f"或把 --l0 设成 {n}")
                elif D % E == 0:
                    hint = (f"\n  {D} = {D // E} × {E}：看起来是 {D // E} 块「每块 {E} 个专家」，"
                            f"检查 stats 的个数与 layers")
                raise ValueError(
                    f"特征维度对不上：模型要 {D} 维，按特征描述拼出来是 {want} 维"
                    f"（{len(self.layers)} 层 × {self.n_experts} 专家 × "
                    f"{len(self.spec.stats)} 种统计 {self.spec.stats}）{hint}")
        else:
            # 自定义特征函数：拿一份全零统计试跑一遍，维度不对现在就报
            z = np.zeros((len(self.layers), self.n_experts))
            x = np.asarray(self._fn(z, z.copy(), np.ones(len(self.layers)),
                                    [l - 1 + self.spec.layer_base for l in self.layers]),
                           dtype=np.float64).reshape(-1)
            if x.shape[0] != D:
                raise ValueError(f"特征函数输出 {x.shape[0]} 维，模型要 {D} 维")
        for k in ("shift", "scale"):
            v = getattr(self, k)
            if v is not None and v.shape != (D,):
                raise ValueError(f"{k} 长度 {v.shape[0]} ≠ 特征维度 {D}")
        missing = sorted(set(self.priors) - set(self.classes))
        # 没部署的类 —— 判给它们的请求没有专属后段（见 predict_stats）
        self.unrecognised = missing
        if len(set(self.classes) & set(self.priors)) == 0:
            raise ValueError(
                f"模型的类别 {self.classes} 与部署的 task {sorted(self.priors)} "
                f"一个都对不上。类别是数字的话用 --lr-classes 0=mbpp,1=gsm8k 映射")

    # -- 构造 -------------------------------------------------------------- #
    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        l0: int,
        n_experts: int,
        priors: Mapping[str, float],
        class_map: str | None = None,
        features: "FeatureSpec | Mapping | str | Path | None" = None,
        feature_fn: str | None = None,
        tau_hi: float | None = None,
        tau_lo: float = 0.40,
    ) -> "LRClassifier":
        """`tau_hi=None`：选择器文件用 0.9（SELECTOR_USAGE.md 推荐的 --min-prob），
        其它 LR 用 0.55（与直方图识别器相同）。"""
        raw = load_lr_params(path)
        meta = raw.get("meta") or {}
        if meta.get("n_experts") and int(meta["n_experts"]) != int(n_experts):
            raise ValueError(
                f"选择器是按 {meta['n_experts']} 个专家/层训练的，部署的模型是 "
                f"{n_experts} 个 —— 不是同一个模型（选择器只对训练它的那个模型有效）")
        if tau_hi is None:
            tau_hi = 0.9 if meta else 0.55
        spec = FeatureSpec.from_any(features if features is not None
                                    else raw.get("features"))
        cmap = parse_class_map(class_map or _cmap_str(raw.get("classes_map")),
                               raw["classes"])
        fn_ref = feature_fn or raw.get("feature_fn")
        fn = None
        if fn_ref:
            f, _, name = str(fn_ref).rpartition(":")
            if not f or not name:
                raise ValueError(f"--lr-feature-fn 要写成 文件.py:函数名，收到 {fn_ref!r}")
            fp = Path(f)
            if not fp.is_absolute() and not fp.exists():
                fp = Path(path).parent / f
            fn = (fp.read_text(encoding="utf-8"), name)
        clf = cls(
            classes=[cmap[str(c)] for c in raw["classes"]],
            labels=_display_labels(raw["classes"], cmap),
            coef=raw["coef"], intercept=raw["intercept"], proba=raw["proba"],
            shift=raw.get("shift"), scale=raw.get("scale"),
            spec=spec, l0=l0, n_experts=n_experts, priors=priors,
            tau_hi=tau_hi, tau_lo=tau_lo, feature_fn=fn, source=str(path),
        )
        clf.meta = meta
        return clf

    meta: dict = {}
    """选择器文件自带的元数据（特征口径、训练样本数……）。只在控制机上用。"""

    @property
    def needs_body(self) -> bool:
        """特征只数 body token —— 协调器要给 prefill 附上 body 区间。"""
        return self.spec.tokens == "body"

    def to_wire(self) -> dict:
        return {
            "kind": "lr",
            "classes": self.classes,
            "labels": self.labels,
            "coef": self.coef.tolist(),
            "intercept": self.intercept.tolist(),
            "proba": self.proba,
            "shift": None if self.shift is None else self.shift.tolist(),
            "scale": None if self.scale is None else self.scale.tolist(),
            "spec": asdict(self.spec),
            "l0": self.l0,
            "n_experts": self.n_experts,
            "priors": self.priors,
            "tau_hi": self.tau_hi,
            "tau_lo": self.tau_lo,
            "feature_fn": list(self.feature_fn) if self.feature_fn else None,
            "source": self.source,
        }

    @classmethod
    def from_wire(cls, d: Mapping) -> "LRClassifier":
        return cls(
            classes=d["classes"], coef=np.asarray(d["coef"]),
            intercept=np.asarray(d["intercept"]), proba=d["proba"],
            shift=None if d.get("shift") is None else np.asarray(d["shift"]),
            scale=None if d.get("scale") is None else np.asarray(d["scale"]),
            spec=FeatureSpec.from_any(d["spec"]), l0=d["l0"],
            n_experts=d["n_experts"], priors=d["priors"],
            tau_hi=d["tau_hi"], tau_lo=d["tau_lo"],
            feature_fn=tuple(d["feature_fn"]) if d.get("feature_fn") else None,
            source=d.get("source", ""), labels=d.get("labels"),
        )

    def describe(self) -> str:
        sc = "" if self.shift is None and self.scale is None else " + 仿射预处理"
        fn = f"，特征函数 {self.feature_fn[1]}()" if self.feature_fn else ""
        base = self.spec.layer_base
        ly = [l - 1 + base for l in self.layers]
        span = f"{ly[0]}–{ly[-1]}" if ly == list(range(ly[0], ly[-1] + 1)) else str(ly)
        merged = {l: c for l, c in zip(self.labels, self.classes) if l != c}
        alias = f"，并池 {merged}" if merged else ""
        return (f"LR {len(self.labels)} 类 {self.labels}{alias}，{self.coef.shape[1]} 维"
                f"（层 {span}，{base}-based × {self.n_experts} 专家 × "
                f"{'+'.join(self.spec.stats)}，归一 {self.spec.normalize}"
                f"{'，log1p' if self.spec.log1p else ''}"
                f"{'，只数 body token' if self.needs_body else ''}）{sc}{fn}，"
                f"概率 {self.proba}，τ_hi={self.tau_hi:g}")

    # -- 推断 -------------------------------------------------------------- #
    def features(self, layers: Mapping[int, LayerStat]) -> np.ndarray:
        cnt, wgt, tok = layer_matrix(layers, self.layers, self.n_experts)
        if self._fn is not None:
            x = self._fn(cnt, wgt, tok, [l - 1 + self.spec.layer_base for l in self.layers])
            return np.asarray(x, dtype=np.float64).reshape(-1)
        return _features(self.spec, cnt, wgt, tok)

    def proba_raw(self, x: np.ndarray) -> np.ndarray:
        if self.shift is not None:
            x = x - self.shift
        if self.scale is not None:
            x = x / np.where(self.scale == 0, 1.0, self.scale)
        z = self.coef @ x + self.intercept
        if self.proba == "binary":
            p1 = 1.0 / (1.0 + np.exp(-z[0]))
            return np.array([1.0 - p1, p1])
        if self.proba == "ovr":
            p = 1.0 / (1.0 + np.exp(-z))
            return p / max(float(p.sum()), 1e-12)
        e = np.exp(z - z.max())
        return e / e.sum()

    def _prior_verdict(self, why: str) -> Verdict:
        best = max(self.priors, key=lambda u: self.priors.get(u, 0.0))
        return Verdict(best, 0.0, "prior", {}, error=why)

    def predict_stats(self, st: MoEStats) -> Verdict:
        if self.needs_body:
            unmasked = [l for l in self.layers
                        if l in st.layers and not st.layers[l].masked]
            if unmasked:
                # 整条 prompt（含对话模板）的计数喂给按 body 训练的模型，分数看起来
                # 照样像回事 —— 但特征分布整个偏了。宁可退到先验也不给一个假分数。
                return self._prior_verdict(
                    "模型要只数 body token 的统计，但收到的是整条 prompt 的 —— "
                    "协调器没有附 body 区间（请求不是走文本入口，或 count_body 没开）")
        try:
            x = self.features(st.layers)
        except (KeyError, ValueError) as e:
            # 识别不出来不能让请求卡死：退到最大先验池，与低置信同一条路，
            # 并把原因带回协调器（日志里能看到是哪一层没到）
            return self._prior_verdict(f"LR 特征拼不出来：{e}")
        if self.needs_body and not any(st.layers[l].n_tokens for l in self.layers):
            return self._prior_verdict("这条 prompt 没有 body token —— 特征全零，没法判")
        p = self.proba_raw(x)
        raw = {c: float(v) for c, v in zip(self.labels, p)}
        pooled: dict[str, float] = {}
        for c, v in zip(self.classes, p):
            pooled[c] = pooled.get(c, 0.0) + float(v)
        # 置信度用**池的真实概率**，不在已部署的池之间重归一 —— 模型把 70% 给了
        # 一个没部署的类时，剩下那个池的 30% 不该被放大成「很确定」。
        scores = {u: v for u, v in pooled.items() if u in self.priors}
        best = max(pooled, key=pooled.get)
        if best not in self.priors:
            # 判给了一个没有后段的类（如 no_robots / math 没部署）。框架里没有
            # 全量兜底池（命题 III.7.5），只能走低置信那条路：绑最大先验池、全程
            # 监控 miss，由通道二决定要不要换绑。
            v = self._prior_verdict("")
            v.confidence = scores.get(v.task, 0.0)
            v.scores, v.raw, v.error = scores, raw, None
            v.note = f"判为 {best}（p={pooled[best]:.2f}），这个类没有部署后段"
            return v
        c = pooled[best]
        if c >= self.tau_hi:
            return Verdict(best, c, "commit", scores, raw=raw)
        if c >= self.tau_lo:
            return Verdict(best, c, "observe", scores, raw=raw)
        v = self._prior_verdict("")
        v.confidence, v.scores, v.raw, v.error = c, scores, raw, None
        return v


def _display_labels(classes: Sequence, cmap: Mapping[str, str]) -> list[str]:
    """原始概率按什么名字上报：类别本来就有名字就用它（humaneval 并进 mbpp 池时
    仍报 humaneval）；类别是 0、1、2 这种下标时用映射后的 task 名，重名再缀下标。"""
    labs = [str(c) for c in classes]
    if not all(l.lstrip("-").isdigit() for l in labs):
        return labs
    names = [cmap[l] for l in labs]
    return [n if names.count(n) == 1 else f"{n}#{l}" for n, l in zip(names, labs)]


def _cmap_str(m: Any) -> str | None:
    if not m:
        return None
    if isinstance(m, Mapping):
        return ",".join(f"{k}={v}" for k, v in m.items())
    return ",".join(map(str, m))


def make_classifier(wire: Mapping | None):
    """节点侧：按 `kind` 还原识别器。没有 kind 的是老格式的直方图识别器。"""
    if not wire:
        return None
    if wire.get("kind") == "lr":
        return LRClassifier.from_wire(wire)
    from .identify import HistogramClassifier
    return HistogramClassifier.from_wire(wire)
