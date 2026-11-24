"""按 checkpoint 自报的架构，从 safetensors 只读需要的张量、建出一段模型。

节点 configure 时（`node._load_torch`）与逐层计时的 disk 模式（`layer_bench` 里量盘上那几层）
共用这一份 —— 两边对「哪个架构要哪些 key、怎么转 dtype」必须是同一套规则。
"""

from __future__ import annotations

import re
from typing import Mapping, Sequence

__all__ = ["load_segment_model", "disk_layout"]


def load_segment_model(hf_cfg: Mapping, model_dir: str,
                       layer_experts: Mapping[int, Sequence[int]], *,
                       with_embed: bool, with_lm_head: bool,
                       device: str = "cpu", miss_policy: str = "drop"):
    """→ (SegmentModel, 模型配置, LoadReport, 架构名)。缺 key 抛 KeyError。

    按 config 的 model_type / architectures 分派。**不要按「有没有某个字段」去猜** ——
    猜错的代价是加载一个结构不对的模型，而它不会报错，只会算出垃圾。"""
    from .glm4_moe_lite import is_glm4_moe_lite
    from .weights import KeyPlan, SelectiveLoader, WeightIndex, qwen3_next_keys, qwen_moe_keys

    arch = str(hf_cfg.get("model_type", "")).lower()
    archs = [str(a).lower() for a in hf_cfg.get("architectures", []) or []]
    idx = WeightIndex(model_dir)
    plan = KeyPlan(layer_experts=layer_experts, with_embed=with_embed, with_lm_head=with_lm_head)
    if is_glm4_moe_lite(hf_cfg):
        # GLM-4.7-Flash：MLA + dense 第 0 层 + 共享专家 + sigmoid 路由（glm4_moe_lite.py）。
        # 路由器两项要 fp32 —— 先按 checkpoint 原样读，再逐项转
        from .glm4_moe_lite import GlmModelConfig, GlmSegmentModel, glm_keys

        mcfg = GlmModelConfig.from_hf(hf_cfg)
        tensors, rep = SelectiveLoader(idx).load(glm_keys(plan, mcfg), device=device, dtype=None)
        if rep.missing:
            raise KeyError(f"checkpoint 缺 {len(rep.missing)} 个 key，首个: {rep.missing[0]}")
        model = GlmSegmentModel(mcfg, layer_experts, mcfg.cast(tensors), device=device,
                                miss_policy=miss_policy)
        return model, mcfg, rep, "glm4_moe_lite"
    is_next = arch == "qwen3_next" or any("qwen3next" in a for a in archs)
    if is_next:
        from .qwen3_next import NextModelConfig, TorchNextSegmentModel

        mcfg = NextModelConfig.from_hf(hf_cfg)
        keys = qwen3_next_keys(plan, layer_types=list(mcfg.layer_types),
                               shared_expert=mcfg.shared_intermediate > 0,
                               tie_word_embeddings=mcfg.tie_word_embeddings)
        Seg = TorchNextSegmentModel
    else:
        from .torch_model import TorchModelConfig, TorchSegmentModel

        mcfg = TorchModelConfig.from_hf(hf_cfg)
        keys = qwen_moe_keys(plan, tie_word_embeddings=mcfg.tie_word_embeddings)
        Seg = TorchSegmentModel
    tensors, rep = SelectiveLoader(idx).load(keys, device=device, dtype=mcfg.torch_dtype)
    if rep.missing:
        raise KeyError(f"checkpoint 缺 {len(rep.missing)} 个 key，首个: {rep.missing[0]}")
    model = Seg(mcfg, layer_experts, tensors, device=device, miss_policy=miss_policy)
    return model, mcfg, rep, ("qwen3_next" if is_next else "qwen3_moe")


_EXPERT = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.")
_LAYER = re.compile(r"^model\.layers\.(\d+)\.")


def disk_layout(keys, hf_cfg: Mapping) -> tuple[dict[int, list[int]], bool, bool]:
    """盘上有哪些张量 → (层号 1-based → 专家, 有没有词嵌入, 有没有输出头)。

    dense 层（GLM 第 1 层）没有专家，给空列表。checkpoint 里 num_hidden_layers 之后的层
    （GLM 的 MTP 层）不算。"""
    n = int(hf_cfg.get("num_hidden_layers", 0)) or 10 ** 9
    layers: dict[int, set[int]] = {}
    for k in keys:
        m = _LAYER.match(k)
        if not m or int(m.group(1)) >= n:
            continue
        s = layers.setdefault(int(m.group(1)) + 1, set())
        e = _EXPERT.match(k)
        if e:
            s.add(int(e.group(2)))
    tie = bool(hf_cfg.get("tie_word_embeddings"))
    keys = set(keys)
    return ({l: sorted(v) for l, v in sorted(layers.items())},
            "model.embed_tokens.weight" in keys,
            "model.norm.weight" in keys and (tie or "lm_head.weight" in keys))
