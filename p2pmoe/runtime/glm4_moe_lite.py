"""GLM-4.7-Flash（`glm4_moe_lite`）的执行层。

与 `torch_model.py`（Qwen3-MoE）、`qwen3_next.py` **保持同一套契约**::

    GlmSegmentModel(cfg, layer_experts, weights).forward(req, x) -> (y, MoEStats)
    .drop_kv(req) / .has_kv(req) / .kv_tokens(req) / .embed_tokens(ids) / .logits(h) / .sample(h)

所以节点、协调器、layerbench 一行都不用为它改，换的只是「一层里面算什么」。
数值口径对齐 transformers 5.x 的 `modeling_glm4_moe_lite.py`（`tests/test_glm4_moe_lite.py`
用同形状的小模型逐层对照）。

与 Qwen3-MoE 的四处结构差异
--------------------------
**1. 第 0 层是 dense。** `first_k_dense_replace=1`（或 config 的 `mlp_layer_types`）：
第 0 层的 MLP 是普通 SwiGLU（`intermediate_size=10240`），没有路由、没有专家。
规划器口径（1-based）下就是第 1 层。清单给这一层列的专家号一律忽略。

**2. MLA 注意力。** q 走低秩（`q_a_proj → q_a_layernorm → q_b_proj`），k/v 由一个
512 维的压缩向量经 `kv_b_proj` 展开；rope 只作用在 64 维的 `k_rot`（所有头共享）上，
且是**交错**排布（`rope_interleave=True`，与 transformers 的默认一致）。
KV 按**展开后**的 K（qk_nope + qk_rope = 256/头）与 V（256/头）缓存 —— 每 token 每层
20 头 × 512 × 2B = 20KiB，与 Petals 方案、HF 默认缓存的大小一致；只存压缩向量（576 × 2B）
省显存，但 decode 每步要把整段上下文重新展开一遍，不划算。

**3. 路由是 sigmoid + 偏置。** 打分 `sigmoid(router(x))`（fp32）；**选**专家用
`分数 + e_score_correction_bias`，**权重**用不带偏置的分数，归一后再乘
`routed_scaling_factor`（1.8）。所以每个 token 的 top-k 权重和是 1.8 不是 1。
画像、逐层统计（hist / LayerStat.mass）按这个原始权重记 —— 与 verify 里 GLM 的采集口径
（`*_prefill_persample.npz` 的 wsum）一致；`miss_mass` 除掉 1.8，仍是「缺失质量占比」。

**4. 共享专家。** 每个 MoE 层还有一个恒定激活的 `shared_experts`（中间维
`moe_intermediate_size × n_shared_experts`），直接加在路由专家的输出上（没有门控）。
它**不参与裁剪**，承载该层的节点必须装它。

checkpoint 里还有 `model.layers.47.*`（MTP，多 token 预测层）—— transformers 加载时
也忽略它，这里不读。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Sequence

import numpy as np

from .model import MoEStats, torch_layer_counts
from .torch_model import _fast_decode_enabled, _lin, _rms_norm, _rope_table, _rope_theta

__all__ = ["GlmModelConfig", "GlmBlock", "GlmSegmentModel", "glm_keys",
           "is_glm4_moe_lite", "glm_synth_weights", "glm_layer_bytes"]

FP32_KEYS = ("mlp.gate.weight", "mlp.gate.e_score_correction_bias")
"""transformers 里 `_keep_in_fp32_modules_strict` 的那两个：路由器权重与选择偏置一律 fp32。"""


def is_glm4_moe_lite(cfg: Mapping) -> bool:
    arch = str(cfg.get("model_type", "")).lower()
    archs = [str(a).lower() for a in cfg.get("architectures", []) or []]
    return arch == "glm4_moe_lite" or any("glm4moelite" in a for a in archs)


# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GlmModelConfig:
    n_layers: int
    n_experts: int
    top_k: int
    d_model: int
    moe_intermediate: int
    intermediate: int
    n_shared: int
    n_heads: int
    q_lora_rank: int | None
    kv_lora_rank: int
    qk_nope: int
    qk_rope: int
    v_head: int
    vocab: int
    mlp_layer_types: tuple[str, ...]
    rms_eps: float = 1e-5
    rope_theta: float = 1_000_000.0
    rope_interleave: bool = True
    norm_topk_prob: bool = True
    routed_scaling: float = 1.0
    n_group: int = 1
    topk_group: int = 1
    tie_word_embeddings: bool = False
    dtype: str = "bfloat16"
    lora_norm_eps: float = 1e-6
    """q_a_layernorm / kv_a_layernorm 的 eps。transformers 建这两个 norm 时没传 eps，
    用的是类默认值 1e-6，不是 config 的 rms_norm_eps —— 照抄。"""

    @classmethod
    def from_hf(cls, cfg: Mapping) -> "GlmModelConfig":
        n = int(cfg["num_hidden_layers"])
        types = cfg.get("mlp_layer_types")
        if not types:
            k = int(cfg.get("first_k_dense_replace", 1))
            types = ["dense"] * k + ["sparse"] * (n - k)
        rp = cfg.get("rope_parameters") or cfg.get("rope_scaling") or {}
        if isinstance(rp, Mapping) and rp.get("rope_type", rp.get("type", "default")) not in (
                "default", None):
            raise NotImplementedError(f"GLM 执行层只实现了默认 RoPE，config 给的是 {rp}")
        if float((rp or {}).get("partial_rotary_factor", cfg.get("partial_rotary_factor", 1.0))
                 or 1.0) != 1.0:
            raise NotImplementedError("GLM 执行层只实现了 partial_rotary_factor = 1")
        dt = cfg.get("torch_dtype", cfg.get("dtype", "bfloat16"))
        return cls(
            n_layers=n,
            n_experts=int(cfg.get("n_routed_experts", cfg.get("num_local_experts", 0))),
            top_k=int(cfg["num_experts_per_tok"]),
            d_model=int(cfg["hidden_size"]),
            moe_intermediate=int(cfg["moe_intermediate_size"]),
            intermediate=int(cfg["intermediate_size"]),
            n_shared=int(cfg.get("n_shared_experts", 1) or 0),
            n_heads=int(cfg["num_attention_heads"]),
            q_lora_rank=(None if cfg.get("q_lora_rank") is None else int(cfg["q_lora_rank"])),
            kv_lora_rank=int(cfg["kv_lora_rank"]),
            qk_nope=int(cfg["qk_nope_head_dim"]),
            qk_rope=int(cfg["qk_rope_head_dim"]),
            v_head=int(cfg["v_head_dim"]),
            vocab=int(cfg["vocab_size"]),
            mlp_layer_types=tuple(str(t) for t in types),
            rms_eps=float(cfg.get("rms_norm_eps", 1e-5)),
            rope_theta=_rope_theta(cfg),
            rope_interleave=bool(cfg.get("rope_interleave", True)),
            norm_topk_prob=bool(cfg.get("norm_topk_prob", True)),
            routed_scaling=float(cfg.get("routed_scaling_factor", 1.0)),
            n_group=int(cfg.get("n_group", 1) or 1),
            topk_group=int(cfg.get("topk_group", 1) or 1),
            tie_word_embeddings=bool(cfg.get("tie_word_embeddings", False)),
            dtype=str(dt).replace("torch.", ""),
        )

    # ---- layerbench / 节点共用的几个名字（与 TorchModelConfig 同名） ---- #
    @property
    def qk_head(self) -> int:
        return self.qk_nope + self.qk_rope

    @property
    def head_dim(self) -> int:
        return self.qk_head

    @property
    def n_kv_heads(self) -> int:
        return self.n_heads          # 展开后每个头都有自己的 K/V

    def is_dense(self, layer: int) -> bool:
        """layer 是规划器口径（1-based）。"""
        return self.mlp_layer_types[int(layer) - 1] == "dense"

    @property
    def first_sparse(self) -> int:
        """第一个 MoE 层（1-based）。"""
        return next(i + 1 for i, t in enumerate(self.mlp_layer_types) if t != "dense")

    @property
    def torch_dtype(self):
        import torch

        return {"bfloat16": torch.bfloat16, "float16": torch.float16,
                "float32": torch.float32}[self.dtype]

    def cast(self, tensors: dict) -> dict:
        """加载后统一 dtype：路由器两项 fp32，其余转成 cfg.dtype。"""
        import torch

        out = {}
        for k, t in tensors.items():
            want = torch.float32 if k.endswith(FP32_KEYS) else self.torch_dtype
            out[k] = t if t.dtype == want else t.to(want)
        return out


# --------------------------------------------------------------------------- #
def glm_keys(plan, cfg: GlmModelConfig) -> set[str]:
    """本节点需要的 key 集合（`weights.KeyPlan` → checkpoint 命名）。

    层号 1-based → checkpoint 0-based。dense 层（第 1 层）只拉普通 MLP；清单里给它列的
    专家号忽略（Petals / 全装方案的清单会按 n_experts 列满）。
    """
    keys: set[str] = set()
    for layer, experts in plan.layer_experts.items():
        l = int(layer)
        if not 1 <= l <= cfg.n_layers:
            raise ValueError(f"层 {l} 超出 1..{cfg.n_layers}（第 {cfg.n_layers + 1} 层是 MTP，不部署）")
        p = f"model.layers.{l - 1}"
        a = f"{p}.self_attn"
        keys |= {f"{p}.input_layernorm.weight", f"{p}.post_attention_layernorm.weight",
                 f"{a}.kv_a_proj_with_mqa.weight", f"{a}.kv_a_layernorm.weight",
                 f"{a}.kv_b_proj.weight", f"{a}.o_proj.weight"}
        if cfg.q_lora_rank is None:
            keys.add(f"{a}.q_proj.weight")
        else:
            keys |= {f"{a}.q_a_proj.weight", f"{a}.q_a_layernorm.weight", f"{a}.q_b_proj.weight"}
        if cfg.is_dense(l):
            keys |= {f"{p}.mlp.{n}.weight" for n in ("gate_proj", "up_proj", "down_proj")}
            continue
        keys |= {f"{p}.mlp.gate.weight", f"{p}.mlp.gate.e_score_correction_bias"}
        if cfg.n_shared:
            keys |= {f"{p}.mlp.shared_experts.{n}.weight"
                     for n in ("gate_proj", "up_proj", "down_proj")}
        for e in experts:
            keys |= {f"{p}.mlp.experts.{int(e)}.{n}.weight"
                     for n in ("gate_proj", "up_proj", "down_proj")}
    if plan.with_embed:
        keys.add("model.embed_tokens.weight")
    if plan.with_lm_head:
        keys.add("model.norm.weight")
        keys.add("model.embed_tokens.weight" if cfg.tie_word_embeddings else "lm_head.weight")
    return keys


def glm_layer_bytes(cfg: GlmModelConfig, n_resident: int, dtype_bytes: int,
                    *, dense: bool = False) -> int:
    """一层的权重字节数（路由器按 fp32 记）。"""
    d, H = cfg.d_model, cfg.n_heads
    if cfg.q_lora_rank is None:
        q = d * H * cfg.qk_head
    else:
        q = d * cfg.q_lora_rank + cfg.q_lora_rank + cfg.q_lora_rank * H * cfg.qk_head
    attn = (q + d * (cfg.kv_lora_rank + cfg.qk_rope) + cfg.kv_lora_rank
            + cfg.kv_lora_rank * H * (cfg.qk_nope + cfg.v_head) + H * cfg.v_head * d + 2 * d)
    if dense:
        return (attn + 3 * d * cfg.intermediate) * dtype_bytes
    shared = 3 * d * cfg.moe_intermediate * cfg.n_shared
    exp = 3 * d * cfg.moe_intermediate * n_resident
    return (attn + shared + exp) * dtype_bytes + (cfg.n_experts * d + cfg.n_experts) * 4


def glm_synth_weights(cfg: GlmModelConfig, layers: Sequence[int], experts: Sequence[int], *,
                      device: str, dtype, seed: int = 0) -> dict:
    """同形状的随机权重（layerbench 用）。key 与 checkpoint 一致。"""
    import torch

    g = torch.Generator(device="cpu").manual_seed(seed)
    d, H, m = cfg.d_model, cfg.n_heads, cfg.moe_intermediate

    def rnd(*shape, dt=dtype):
        return (torch.randn(*shape, generator=g) * 0.02).to(device=device, dtype=dt)

    def ones(n):
        return torch.ones(n, device=device, dtype=dtype)

    w: dict = {}
    for l in layers:
        p = f"model.layers.{int(l) - 1}"
        a = f"{p}.self_attn"
        if cfg.q_lora_rank is None:
            w[f"{a}.q_proj.weight"] = rnd(H * cfg.qk_head, d)
        else:
            w[f"{a}.q_a_proj.weight"] = rnd(cfg.q_lora_rank, d)
            w[f"{a}.q_a_layernorm.weight"] = ones(cfg.q_lora_rank)
            w[f"{a}.q_b_proj.weight"] = rnd(H * cfg.qk_head, cfg.q_lora_rank)
        w[f"{a}.kv_a_proj_with_mqa.weight"] = rnd(cfg.kv_lora_rank + cfg.qk_rope, d)
        w[f"{a}.kv_a_layernorm.weight"] = ones(cfg.kv_lora_rank)
        w[f"{a}.kv_b_proj.weight"] = rnd(H * (cfg.qk_nope + cfg.v_head), cfg.kv_lora_rank)
        w[f"{a}.o_proj.weight"] = rnd(d, H * cfg.v_head)
        w[f"{p}.input_layernorm.weight"] = ones(d)
        w[f"{p}.post_attention_layernorm.weight"] = ones(d)
        if cfg.is_dense(l):
            f = cfg.intermediate
            w[f"{p}.mlp.gate_proj.weight"] = rnd(f, d)
            w[f"{p}.mlp.up_proj.weight"] = rnd(f, d)
            w[f"{p}.mlp.down_proj.weight"] = rnd(d, f)
            continue
        w[f"{p}.mlp.gate.weight"] = rnd(cfg.n_experts, d, dt=torch.float32)
        w[f"{p}.mlp.gate.e_score_correction_bias"] = rnd(cfg.n_experts, dt=torch.float32)
        s = m * cfg.n_shared
        if cfg.n_shared:
            w[f"{p}.mlp.shared_experts.gate_proj.weight"] = rnd(s, d)
            w[f"{p}.mlp.shared_experts.up_proj.weight"] = rnd(s, d)
            w[f"{p}.mlp.shared_experts.down_proj.weight"] = rnd(d, s)
        for e in experts:
            w[f"{p}.mlp.experts.{e}.gate_proj.weight"] = rnd(m, d)
            w[f"{p}.mlp.experts.{e}.up_proj.weight"] = rnd(m, d)
            w[f"{p}.mlp.experts.{e}.down_proj.weight"] = rnd(d, m)
    return w


# --------------------------------------------------------------------------- #
def _bytes(ts) -> int:
    return int(sum(t.numel() * t.element_size() for t in ts if t is not None))


class GlmBlock:
    """一层 GLM-4.7-Flash。MoE 层**只有 `resident` 里的专家会被 materialize**；dense 层没有专家。"""

    def __init__(self, cfg: GlmModelConfig, layer: int, resident: Iterable[int],
                 weights: Mapping[str, "object"], *, device: str = "cpu",
                 miss_policy: str = "drop"):
        import torch

        self.cfg = cfg
        self.layer = int(layer)
        self.device = device
        if miss_policy not in ("drop", "drop_noscale", "local_topk"):
            raise ValueError(f"未知的 miss_policy {miss_policy!r}")
        self.miss_policy = miss_policy
        self.dense = cfg.is_dense(self.layer)
        self.resident = frozenset() if self.dense else frozenset(int(e) for e in resident)
        if not self.dense and not self.resident:
            raise ValueError(f"layer {layer} 的驻留专家集为空")

        p = f"model.layers.{self.layer - 1}"

        def w(name: str):
            k = f"{p}.{name}"
            if k not in weights:
                raise KeyError(f"缺少权重 {k}")
            return weights[k]

        a = "self_attn"
        if cfg.q_lora_rank is None:
            self.wq, self.wqa, self.qa_norm, self.wqb = w(f"{a}.q_proj.weight"), None, None, None
        else:
            self.wq = None
            self.wqa, self.qa_norm, self.wqb = (w(f"{a}.q_a_proj.weight"),
                                                w(f"{a}.q_a_layernorm.weight"),
                                                w(f"{a}.q_b_proj.weight"))
        self.wkva = w(f"{a}.kv_a_proj_with_mqa.weight")
        self.kva_norm = w(f"{a}.kv_a_layernorm.weight")
        self.wkvb = w(f"{a}.kv_b_proj.weight")
        self.wo = w(f"{a}.o_proj.weight")
        self.ln_in, self.ln_post = w("input_layernorm.weight"), w("post_attention_layernorm.weight")
        self.scaling = cfg.qk_head ** -0.5

        self._gate, self._up, self._down = {}, {}, {}
        self.router = self.bias = None
        self.sh_gate = self.sh_up = self.sh_down = None
        if self.dense:
            self.d_gate, self.d_up, self.d_down = (w("mlp.gate_proj.weight"),
                                                   w("mlp.up_proj.weight"),
                                                   w("mlp.down_proj.weight"))
            self._resident_idx = None
            return
        self.router = w("mlp.gate.weight").float()
        self.bias = w("mlp.gate.e_score_correction_bias").float()
        if cfg.n_shared:
            self.sh_gate, self.sh_up, self.sh_down = (w("mlp.shared_experts.gate_proj.weight"),
                                                      w("mlp.shared_experts.up_proj.weight"),
                                                      w("mlp.shared_experts.down_proj.weight"))
        for e in sorted(self.resident):
            self._gate[e] = w(f"mlp.experts.{e}.gate_proj.weight")
            self._up[e] = w(f"mlp.experts.{e}.up_proj.weight")
            self._down[e] = w(f"mlp.experts.{e}.down_proj.weight")
        self._resident_idx = torch.tensor(sorted(self.resident), device=device, dtype=torch.long)

    # -- 计量 -------------------------------------------------------------- #
    def _base_tensors(self):
        ts = [self.wq, self.wqa, self.qa_norm, self.wqb, self.wkva, self.kva_norm, self.wkvb,
              self.wo, self.ln_in, self.ln_post]
        if self.dense:
            return ts + [self.d_gate, self.d_up, self.d_down]
        return ts + [self.router, self.bias, self.sh_gate, self.sh_up, self.sh_down]

    @property
    def resident_bytes(self) -> int:
        return _bytes(self._base_tensors()) + _bytes(
            t for d in (self._gate, self._up, self._down) for t in d.values())

    @property
    def full_bytes(self) -> int:
        if self.dense:
            return self.resident_bytes
        e0 = min(self.resident)
        per = _bytes((self._gate[e0], self._up[e0], self._down[e0]))
        return _bytes(self._base_tensors()) + per * self.cfg.n_experts

    # -- attention（MLA） --------------------------------------------------- #
    def _rope(self, x, cos, sin):
        """x: [T, h, r]；cos/sin: [T, 1, r]（cat(freqs, freqs) 形式）。"""
        import torch

        from .torch_model import _rotate_half

        if not self.cfg.rope_interleave:
            return x * cos + _rotate_half(x) * sin
        h = x.shape[-1] // 2
        c, s = cos[..., :h], sin[..., :h]
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)

    def _attend(self, x, cache: dict):
        import torch

        c = self.cfg
        T, H = x.shape[0], c.n_heads
        if self.wq is not None:
            q = _lin(x, self.wq)
        else:
            q = _lin(_rms_norm(_lin(x, self.wqa), self.qa_norm, c.lora_norm_eps), self.wqb)
        q = q.view(T, H, c.qk_head)
        q_pass, q_rot = q[..., :c.qk_nope], q[..., c.qk_nope:]

        ckv = _lin(x, self.wkva)
        kv_nope = _rms_norm(ckv[:, :c.kv_lora_rank], self.kva_norm, c.lora_norm_eps)
        k_rot = ckv[:, c.kv_lora_rank:].view(T, 1, c.qk_rope)

        past = cache.get("k")
        offset = 0 if past is None else past.shape[0]
        cos_t, sin_t = _rope_table(c.qk_rope, c.rope_theta, x.device, x.dtype, offset + T)
        cos, sin = cos_t[offset:offset + T, None, :], sin_t[offset:offset + T, None, :]
        q_rot, k_rot = self._rope(q_rot, cos, sin), self._rope(k_rot, cos, sin)

        kv = _lin(kv_nope, self.wkvb).view(T, H, c.qk_nope + c.v_head)
        k = torch.cat([kv[..., :c.qk_nope], k_rot.expand(T, H, c.qk_rope)], dim=-1)
        v = kv[..., c.qk_nope:]
        qf = torch.cat([q_pass, q_rot], dim=-1)

        if past is not None:
            k = torch.cat([past, k], dim=0)
            v = torch.cat([cache["v"], v], dim=0)
        else:
            v = v.contiguous()
        cache["k"], cache["v"] = k, v
        T_all = k.shape[0]

        qh, kh, vh = qf.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        scores = (qh @ kh.transpose(1, 2)) * self.scaling
        if T > 1:
            idx = (torch.arange(T_all, device=x.device)[None, :]
                   > (offset + torch.arange(T, device=x.device))[:, None])
            scores = scores.masked_fill(idx[None, :, :], torch.finfo(scores.dtype).min)
        out = torch.softmax(scores.float(), dim=-1).to(scores.dtype) @ vh
        return _lin(out.transpose(0, 1).reshape(T, H * c.v_head), self.wo)

    # -- MLP ---------------------------------------------------------------- #
    @staticmethod
    def _swiglu(x, g, u, d):
        import torch

        return _lin(torch.nn.functional.silu(_lin(x, g)) * _lin(x, u), d)

    def _route(self, z):
        """→ (scores[T,E] fp32, choice[T,E] fp32, topw[T,k] fp32（含 ×1.8）, topi[T,k])。"""
        import torch

        c = self.cfg
        scores = torch.sigmoid(_lin(z.float(), self.router))
        choice = scores + self.bias
        if c.n_group > 1:
            T = z.shape[0]
            g = choice.view(T, c.n_group, c.n_experts // c.n_group).topk(2, dim=-1)[0].sum(-1)
            gi = torch.topk(g, k=c.topk_group, dim=-1, sorted=False)[1]
            gm = torch.zeros_like(g).scatter_(1, gi, 1)
            sm = gm.unsqueeze(-1).expand(-1, c.n_group, c.n_experts // c.n_group).reshape(T, -1)
            choice = choice.masked_fill(~sm.bool(), float("-inf"))
        topi = torch.topk(choice, k=c.top_k, dim=-1)[1]
        topw = self._weights(scores, topi)
        return scores, choice, topw, topi

    def _weights(self, scores, idx):
        w = scores.gather(1, idx)
        if self.cfg.norm_topk_prob:
            w = w / (w.sum(-1, keepdim=True) + 1e-20)
        return w * self.cfg.routed_scaling

    def _choose(self, scores, choice, topw, topi, here):
        """→ (use_w, use_i, use_here)：按 miss_policy 决定实际算哪些专家（权重 fp32）。"""
        import torch

        if self.miss_policy == "local_topk":
            nmask = getattr(self, "_nonres_mask", None)
            if nmask is None or nmask.device != choice.device:
                mask = torch.zeros(self.cfg.n_experts, dtype=torch.bool, device=choice.device)
                mask[self._resident_idx] = True
                nmask = self._nonres_mask = ~mask[None, :]
            kk = min(self.cfg.top_k, len(self.resident))
            use_i = torch.topk(choice.masked_fill(nmask, float("-inf")), kk, dim=-1)[1]
            return self._weights(scores, use_i), use_i, torch.ones_like(use_i, dtype=torch.bool)
        kept = topw * here
        if self.miss_policy == "drop_noscale":
            return kept, topi, here
        # drop：跳过缺失专家，剩下的按原来的总量（×1.8 之后的和）重新分配
        denom = kept.sum(-1, keepdim=True)
        use_w = torch.where(denom > 0, kept * (topw.sum(-1, keepdim=True) / denom.clamp_min(1e-20)),
                            torch.zeros_like(kept))
        return use_w, topi, here

    def _experts(self, z, use_w, use_i, use_here, *, decode: bool):
        import torch

        out = torch.zeros_like(z)
        if decode:
            pick = torch.where(use_here, use_i, torch.full_like(use_i, -1))
            sel = pick.reshape(-1).tolist()                       # 唯一一次同步
            k = use_i.shape[1]
            for j, e in sorted(((j, e) for j, e in enumerate(sel) if e >= 0), key=lambda t: t[1]):
                ti, slot = divmod(j, k)
                h = self._swiglu(z[ti:ti + 1], self._gate[e], self._up[e], self._down[e])
                out[ti:ti + 1] += (h * use_w[ti, slot]).to(out.dtype)
            return out
        for e in torch.unique(use_i[use_here]).tolist():
            if e not in self._gate:
                continue
            tok, slot = ((use_i == e) & use_here).nonzero(as_tuple=True)
            h = self._swiglu(z[tok], self._gate[e], self._up[e], self._down[e])
            out.index_add_(0, tok, (h * use_w[tok, slot, None]).to(out.dtype))
        return out

    def _shared(self, z):
        if self.sh_gate is None:
            return 0
        return self._swiglu(z, self.sh_gate, self.sh_up, self.sh_down)

    # -- 前向 -------------------------------------------------------------- #
    def forward(self, x, cache: dict, count_mask=None):
        """x: [T, d_model]。返回 (y, 本层统计)；dense 层的统计为空（不记进逐层统计）。"""
        import torch

        c = self.cfg
        h = x + self._attend(_rms_norm(x, self.ln_in, c.rms_eps), cache)
        z = _rms_norm(h, self.ln_post, c.rms_eps)
        if self.dense:
            return h + self._swiglu(z, self.d_gate, self.d_up, self.d_down), None
        T = z.shape[0]
        scores, choice, topw, topi = self._route(z)
        here = torch.isin(topi, self._resident_idx)
        miss_tok = int((~here).any(-1).sum().item())
        miss_mass = float((topw * ~here).sum().item()) / c.routed_scaling
        hist = torch.zeros(c.n_experts, dtype=torch.float64, device=z.device)
        hist.index_add_(0, topi.reshape(-1), topw.reshape(-1).double())
        use_w, use_i, use_here = self._choose(scores, choice, topw, topi, here)
        out = self._experts(z, use_w, use_i, use_here, decode=False)
        h_, cnt, lm, lt = torch_layer_counts(torch, topi, topw, hist, c.n_experts, count_mask)
        st = MoEStats(hist=h_, count=cnt, lmass=lm, ltok=lt, n_token_layer=T,
                      miss_token_layer=miss_tok, miss_mass=miss_mass)
        return h + (out + self._shared(z)), st

    def forward_decode(self, x, cache: dict, acc: dict):
        """decode 快路径（单 token）：每层只同步一次，统计累加在 GPU 上的 acc 里。"""
        import torch

        c = self.cfg
        h = x + self._attend(_rms_norm(x, self.ln_in, c.rms_eps), cache)
        z = _rms_norm(h, self.ln_post, c.rms_eps)
        if self.dense:
            return h + self._swiglu(z, self.d_gate, self.d_up, self.d_down)
        scores, choice, topw, topi = self._route(z)
        here = torch.isin(topi, self._resident_idx)
        acc["miss_tok"] += (~here).any(-1).sum()
        acc["miss_mass"] += (topw * ~here).sum().double() / c.routed_scaling
        acc["hist"].index_add_(0, topi.reshape(-1), topw.reshape(-1).double())
        acc["n"] += 1
        use_w, use_i, use_here = self._choose(scores, choice, topw, topi, here)
        out = self._experts(z, use_w, use_i, use_here, decode=True)
        return h + (out + self._shared(z))


# --------------------------------------------------------------------------- #
class GlmSegmentModel:
    """一台节点上承载的层区间（GLM-4.7-Flash）。KV 按 (req, layer) 分桶。"""

    fast_decode = True

    def __init__(self, cfg: GlmModelConfig, layer_experts: Mapping[int, Sequence[int]],
                 weights: Mapping[str, "object"], *, device: str = "cpu",
                 miss_policy: str = "drop"):
        self.cfg = cfg
        self.device = device
        self.miss_policy = miss_policy
        self.layers = sorted(int(l) for l in layer_experts)
        for l in self.layers:
            if not 1 <= l <= cfg.n_layers:
                raise ValueError(f"层 {l} 超出 1..{cfg.n_layers}")
        self.blocks = {int(l): GlmBlock(cfg, int(l), es, weights, device=device,
                                        miss_policy=miss_policy)
                       for l, es in layer_experts.items()}
        self.embed = weights.get("model.embed_tokens.weight")
        self.final_norm = weights.get("model.norm.weight")
        self.lm_head = weights.get(
            "model.embed_tokens.weight" if cfg.tie_word_embeddings else "lm_head.weight")
        self._kv: dict[str, dict[int, dict]] = {}
        self.profiler = None

    def enable_profiling(self) -> None:
        from .profile import LayerProfiler

        self.profiler = LayerProfiler(self.cfg.n_experts)

    @property
    def resident_bytes(self) -> int:
        return sum(b.resident_bytes for b in self.blocks.values())

    @property
    def full_bytes(self) -> int:
        return sum(b.full_bytes for b in self.blocks.values())

    def embed_tokens(self, ids: Sequence[int]):
        import torch

        if self.embed is None:
            raise RuntimeError("本节点没有词嵌入（只有前段的 head 需要）")
        idx = torch.as_tensor(list(ids), dtype=torch.long, device=self.device)
        return self.embed[idx]

    def _coerce(self, x):
        import torch

        if isinstance(x, torch.Tensor):
            return x.to(device=self.device, dtype=self.cfg.torch_dtype)
        arr = np.array(x, dtype=np.float32, copy=True, order="C")
        return torch.from_numpy(arr).to(device=self.device, dtype=self.cfg.torch_dtype)

    def _forward_decode(self, x, kv: dict):
        import torch

        E = self.cfg.n_experts
        dev = x.device
        acc = {"hist": torch.zeros(E, dtype=torch.float64, device=dev),
               "miss_tok": torch.zeros((), dtype=torch.int64, device=dev),
               "miss_mass": torch.zeros((), dtype=torch.float64, device=dev), "n": 0}
        h = x
        for l in self.layers:
            h = self.blocks[l].forward_decode(h, kv.setdefault(l, {}), acc)
        v = torch.cat([acc["hist"], acc["miss_tok"].double()[None],
                       acc["miss_mass"][None]]).cpu().numpy()
        return h, MoEStats(hist=v[:E].copy(), n_token_layer=acc["n"] * x.shape[0],
                           miss_token_layer=int(v[E]), miss_mass=float(v[E + 1]))

    def forward(self, req: str, x, count_mask=None, *, decode: bool = False):
        x = self._coerce(x)
        kv = self._kv.setdefault(req, {})
        if (decode and count_mask is None and self.profiler is None
                and _fast_decode_enabled()):
            return self._forward_decode(x, kv)
        total = MoEStats.zeros(self.cfg.n_experts)
        h = x
        for l in self.layers:
            h, st = self.blocks[l].forward(h, kv.setdefault(l, {}), count_mask=count_mask)
            if st is None:                 # dense 层：没有路由，不进统计
                continue
            if self.profiler is not None:
                self.profiler.record(l, st.hist, st.n_token_layer)
            total = total.merge(st.at_layer(l))
        return h, total

    def logits(self, h):
        if self.final_norm is None or self.lm_head is None:
            raise RuntimeError("本节点没有输出头（只有后段的 tail 需要）")
        return _lin(_rms_norm(h, self.final_norm, self.cfg.rms_eps), self.lm_head)

    def sample(self, h, *, temperature: float = 0.0, seed: int = 0) -> int:
        import torch

        lg = self.logits(h[-1:])[0].float()
        if temperature <= 0:
            return int(torch.argmax(lg).item())
        g = torch.Generator(device="cpu").manual_seed(seed)
        p = torch.softmax(lg.cpu() / temperature, dim=-1)
        return int(torch.multinomial(p, 1, generator=g).item())

    def drop_kv(self, req: str) -> bool:
        return self._kv.pop(req, None) is not None

    def has_kv(self, req: str) -> bool:
        return req in self._kv

    def kv_tokens(self, req: str) -> int:
        kv = self._kv.get(req)
        if not kv:
            return 0
        k = kv.get(self.layers[0], {}).get("k")
        return 0 if k is None else int(k.shape[0])

    def active_requests(self) -> list[str]:
        return sorted(self._kv)
