"""decode 快路径与原路径**逐位一致**。

sel_static3：每层 decode 4–5ms，A30 与 T4 一样 —— 不是算力，是每层 ~12 次
GPU→CPU 同步加上 numpy 簿记。快路径每层只同步一次、统计在 GPU 上累加，整段
搬一次。它只能改快，不能改结果：这里逐位比较隐状态、hist 与 miss 计数。
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from p2pmoe.runtime.torch_model import TorchModelConfig, TorchSegmentModel  # noqa: E402

E, D, I, K, L = 32, 16, 8, 4, 3


def _model(resident, policy):
    cfg = TorchModelConfig.from_hf({
        "hidden_size": D, "num_attention_heads": 4, "num_key_value_heads": 2,
        "head_dim": 8, "num_experts": E, "num_experts_per_tok": K,
        "moe_intermediate_size": I, "num_hidden_layers": L, "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0, "norm_topk_prob": True, "vocab_size": 50,
        "torch_dtype": "float32",
    })
    g = torch.Generator().manual_seed(7)
    r = lambda *s: torch.randn(*s, generator=g) * 0.3   # noqa: E731
    w = {"model.embed_tokens.weight": r(50, D), "model.norm.weight": torch.ones(D),
         "lm_head.weight": r(50, D)}
    for li in range(L):
        p = f"model.layers.{li}"
        w.update({f"{p}.self_attn.q_proj.weight": r(32, D),
                  f"{p}.self_attn.k_proj.weight": r(16, D),
                  f"{p}.self_attn.v_proj.weight": r(16, D),
                  f"{p}.self_attn.o_proj.weight": r(D, 32),
                  f"{p}.self_attn.q_norm.weight": torch.ones(8),
                  f"{p}.self_attn.k_norm.weight": torch.ones(8),
                  f"{p}.input_layernorm.weight": torch.ones(D),
                  f"{p}.post_attention_layernorm.weight": torch.ones(D),
                  f"{p}.mlp.gate.weight": r(E, D)})
        for e in range(E):
            w[f"{p}.mlp.experts.{e}.gate_proj.weight"] = r(I, D)
            w[f"{p}.mlp.experts.{e}.up_proj.weight"] = r(I, D)
            w[f"{p}.mlp.experts.{e}.down_proj.weight"] = r(D, I)
    return TorchSegmentModel(cfg, {li + 1: resident for li in range(L)}, w,
                             miss_policy=policy)


@pytest.mark.parametrize("policy", ["local_topk", "drop", "drop_noscale"])
@pytest.mark.parametrize("resident", ["all", "partial"])
def test_fast_decode_is_bitwise_identical(policy, resident):
    res = list(range(E)) if resident == "all" else list(range(0, E, 2))
    m = _model(res, policy)
    prompt = m.embed_tokens([3, 9, 27, 4, 11])
    m.forward("slow", prompt)
    m.forward("fast", prompt)
    tok = 5
    miss_seen = 0
    for _ in range(6):
        x = m.embed_tokens([tok])
        ys, ss = m.forward("slow", x)                 # 原路径
        yf, sf = m.forward("fast", x, decode=True)    # 快路径
        assert torch.equal(ys, yf)
        np.testing.assert_array_equal(ss.hist, sf.hist)
        assert ss.n_token_layer == sf.n_token_layer
        assert ss.miss_token_layer == sf.miss_token_layer
        assert ss.miss_mass == pytest.approx(sf.miss_mass, rel=1e-12, abs=1e-12)
        miss_seen += sf.miss_token_layer
        tok = m.sample(yf)
    assert m.kv_tokens("slow") == m.kv_tokens("fast") == 11
    if resident == "partial":
        assert miss_seen > 0, "半数驻留却一次 miss 都没有 —— 测试没覆盖到缺专家的分支"


def test_env_switch_falls_back(monkeypatch):
    m = _model(list(range(E)), "local_topk")
    m.forward("a", m.embed_tokens([1, 2]))
    monkeypatch.setenv("P2P_FAST_DECODE", "0")
    _, st = m.forward("a", m.embed_tokens([3]), decode=True)
    assert st.layers, "关掉快路径后应走原路径（带逐层统计）"


def test_prefill_and_profiling_never_take_the_fast_path():
    m = _model(list(range(E)), "local_topk")
    _, st = m.forward("a", m.embed_tokens([1, 2, 3]), decode=True,
                      count_mask=[True, True, False])
    assert st.layers                       # 带 count_mask（前段 prefill）→ 原路径
    m.enable_profiling()
    _, st = m.forward("a", m.embed_tokens([4]), decode=True)
    assert st.layers                       # 开了画像 → 原路径


def test_qwen3_next_model_is_not_routed_to_fast_path():
    from p2pmoe.runtime.qwen3_next import TorchNextSegmentModel
    assert not getattr(TorchNextSegmentModel, "fast_decode", False)
