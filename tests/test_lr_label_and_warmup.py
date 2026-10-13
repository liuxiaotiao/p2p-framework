"""sel_static 实测暴露的三件事：

1. LR 判给没部署的类（humaneval/math）时，路由退到先验池 gsm8k —— 报表却拿
   路由结果算「识别准确率」，把判对的 12 条中的 8 条记成「判成 gsm8k ✗」。
2. 预热只按测试集顺序取前 N 条（前两条都是 gsm8k），其他通道的第一条正式请求
   首 token 多出 ~11s。
3. torch MoE 层遍历**全部**驻留专家、每个 `if sel.any()` 一次 GPU 同步。
"""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from p2pmoe.deploy.control import _shadow_acc, _warmup_plan
from p2pmoe.runtime.lr_classifier import verdict_label


# ---------------------------------------------------------------- 1. 识别口径 #
def test_label_is_lr_argmax_not_routed_pool():
    wire = {"task": "gsm8k", "conf": 0.0, "zone": "prior",
            "raw": {"gsm8k": 0.0, "math": 0.0, "mbpp": 0.0, "humaneval": 1.0,
                    "no_robots": 0.0}}
    assert verdict_label(wire) == "humaneval"


def test_label_falls_back_to_task_without_raw():
    assert verdict_label({"task": "mbpp", "conf": 0.9}) == "mbpp"
    assert verdict_label(None) is None


def test_label_strips_index_suffix_of_duplicate_names():
    assert verdict_label({"task": "mbpp", "raw": {"mbpp#2": 0.8, "mbpp#3": 0.2}}) == "mbpp"


def test_shadow_accuracy_counts_undeployed_classes_correctly():
    rs = [
        {"true_task": "humaneval", "shadow": {"task": "gsm8k",
                                              "raw": {"humaneval": 0.99, "gsm8k": 0.01}}},
        {"true_task": "gsm8k", "shadow": {"task": "gsm8k",
                                          "raw": {"humaneval": 0.0, "gsm8k": 1.0}}},
    ]
    assert _shadow_acc(rs) == 1.0


# ---------------------------------------------------------------- 2. 预热覆盖 #
def test_static_warmup_hits_every_channel():
    coord = SimpleNamespace(front_pools={"gsm8k": deque(["F0", "F1"]),
                                         "mbpp": deque(["F2", "F4"]),
                                         "no_robots": deque(["F3"])})
    plan = _warmup_plan(coord, ["gsm8k", "mbpp", "no_robots"], True, 2)
    assert sorted(plan) == ["gsm8k", "gsm8k", "mbpp", "mbpp", "no_robots"]


def test_dynamic_warmup_covers_backs_and_fronts():
    coord = SimpleNamespace(back_capacity={"gsm8k": 2, "mbpp": 2, "no_robots": 1},
                            free_fronts=deque(["F0", "F1", "F2", "F3", "F4", "F5"]))
    plan = _warmup_plan(coord, ["gsm8k", "mbpp", "no_robots"], False, 2)
    assert len(plan) == 6                       # 前段比后段多一条 → 补到 6
    assert plan.count("no_robots") >= 1 and plan.count("mbpp") >= 2


def test_warmup_respects_larger_request():
    coord = SimpleNamespace(front_pools={"gsm8k": deque(["F0"])})
    assert len(_warmup_plan(coord, ["gsm8k"], True, 4)) == 4


# ---------------------------------------------------------- 3. 只算选中的专家 #
def test_moe_block_only_touches_selected_experts():
    torch = pytest.importorskip("torch")
    from p2pmoe.runtime.torch_model import (
        TorchModelConfig, TorchPartialExpertMoEBlock,
    )

    E, d, I, k = 32, 16, 8, 2
    cfg = TorchModelConfig.from_hf({
        "hidden_size": d, "num_attention_heads": 2, "num_key_value_heads": 1,
        "head_dim": 8, "num_experts": E, "num_experts_per_tok": k,
        "moe_intermediate_size": I, "num_hidden_layers": 1, "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0, "norm_topk_prob": True, "vocab_size": 10,
        "torch_dtype": "float32",
    })
    g = torch.Generator().manual_seed(0)
    r = lambda *s: torch.randn(*s, generator=g) * 0.1   # noqa: E731
    p = "model.layers.0"
    w = {f"{p}.self_attn.q_proj.weight": r(16, d), f"{p}.self_attn.k_proj.weight": r(8, d),
         f"{p}.self_attn.v_proj.weight": r(8, d), f"{p}.self_attn.o_proj.weight": r(d, 16),
         f"{p}.self_attn.q_norm.weight": torch.ones(8),
         f"{p}.self_attn.k_norm.weight": torch.ones(8),
         f"{p}.input_layernorm.weight": torch.ones(d),
         f"{p}.post_attention_layernorm.weight": torch.ones(d),
         f"{p}.mlp.gate.weight": r(E, d)}
    for e in range(E):
        w[f"{p}.mlp.experts.{e}.gate_proj.weight"] = r(I, d)
        w[f"{p}.mlp.experts.{e}.up_proj.weight"] = r(I, d)
        w[f"{p}.mlp.experts.{e}.down_proj.weight"] = r(d, I)
    blk = TorchPartialExpertMoEBlock(cfg, 1, range(E), w)

    # 参照：旧实现 —— 遍历全部驻留专家
    x = r(1, d)
    y_new, _ = blk.forward(x.clone(), {})

    touched = []

    class Spy(dict):
        def __getitem__(self, e):
            touched.append(e)
            return dict.__getitem__(self, e)
    blk._gate = Spy(blk._gate)
    blk.forward(x.clone(), {})
    assert 0 < len(touched) <= k               # decode 1 个 token：最多碰 k 个专家
    assert len(set(touched)) == len(touched)

    # 结果与逐个遍历全部专家的写法逐位一致（同样按专家号升序累加）
    from p2pmoe.runtime.torch_model import _rms_norm
    blk._gate = dict(blk._gate)
    h = x + blk._attend(_rms_norm(x, blk.ln_in, cfg.rms_eps), {})
    z = _rms_norm(h, blk.ln_post, cfg.rms_eps)
    pr = torch.softmax((z @ blk.router.T).float(), -1)
    tw, ti = torch.topk(pr, k, -1)
    tw = tw / tw.sum(-1, keepdim=True)
    ref = torch.zeros_like(z)
    for e in range(E):
        sel = ti == e
        if not sel.any():
            continue
        tok, slot = sel.nonzero(as_tuple=True)
        xi = z[tok]
        hi = torch.nn.functional.silu(xi @ blk._gate[e].T) * (xi @ blk._up[e].T)
        ref.index_add_(0, tok, (hi @ blk._down[e].T) * tw[tok, slot][:, None])
    np.testing.assert_array_equal((h + ref).numpy(), y_new.numpy())


def _tiny_block(resident, policy):
    torch = pytest.importorskip("torch")
    from p2pmoe.runtime.torch_model import TorchModelConfig, TorchPartialExpertMoEBlock

    E, d, I, k = 32, 16, 8, 8
    cfg = TorchModelConfig.from_hf({
        "hidden_size": d, "num_attention_heads": 2, "num_key_value_heads": 1,
        "head_dim": 8, "num_experts": E, "num_experts_per_tok": k,
        "moe_intermediate_size": I, "num_hidden_layers": 1, "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0, "norm_topk_prob": True, "vocab_size": 10,
        "torch_dtype": "float32",
    })
    g = torch.Generator().manual_seed(1)
    r = lambda *s: torch.randn(*s, generator=g) * 0.1   # noqa: E731
    p = "model.layers.0"
    w = {f"{p}.self_attn.q_proj.weight": r(16, d), f"{p}.self_attn.k_proj.weight": r(8, d),
         f"{p}.self_attn.v_proj.weight": r(8, d), f"{p}.self_attn.o_proj.weight": r(d, 16),
         f"{p}.self_attn.q_norm.weight": torch.ones(8),
         f"{p}.self_attn.k_norm.weight": torch.ones(8),
         f"{p}.input_layernorm.weight": torch.ones(d),
         f"{p}.post_attention_layernorm.weight": torch.ones(d),
         f"{p}.mlp.gate.weight": r(E, d)}
    for e in range(E):
        w[f"{p}.mlp.experts.{e}.gate_proj.weight"] = r(I, d)
        w[f"{p}.mlp.experts.{e}.up_proj.weight"] = r(I, d)
        w[f"{p}.mlp.experts.{e}.down_proj.weight"] = r(d, I)
    return torch, TorchPartialExpertMoEBlock(cfg, 1, resident, w, miss_policy=policy), r(5, d)


def test_local_topk_always_uses_k_resident_experts():
    """「直接在裁剪后的专家里取 top-8」：每个 token 都用满 8 个，且全是驻留的。"""
    resident = list(range(0, 32, 3))            # 11 个驻留，远少于 32
    torch, blk, x = _tiny_block(resident, "local_topk")
    picked = []
    orig = blk.forward

    real_topk = torch.topk

    def spy_topk(t, k, dim=-1, **kw):
        out = real_topk(t, k, dim=dim, **kw)
        picked.append(out.indices)
        return out
    torch.topk = spy_topk
    try:
        _, st = orig(x, {})
    finally:
        torch.topk = real_topk
    # 第一次 topk 是全量（统计用），第二次是驻留集内部（计算用）
    full_i, local_i = picked[0], picked[1]
    assert local_i.shape == (5, 8)
    assert set(local_i.reshape(-1).tolist()) <= set(resident)
    # miss 统计仍按全量路由：11/32 的驻留，全量 top-8 不可能全命中
    assert st.miss_token_layer > 0
    assert not set(full_i.reshape(-1).tolist()) <= set(resident)


def test_cli_default_is_local_topk():
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    src = (root / "p2pmoe/deploy/control.py").read_text(encoding="utf-8")
    assert re.search(r'"--miss-policy", default="local_topk"', src)
    sh = (root / "deploy_15.sh").read_text(encoding="utf-8")
    assert "MISS_POLICY=${MISS_POLICY:-local_topk}" in sh
    assert '--miss-policy "$MISS_POLICY"' in sh
