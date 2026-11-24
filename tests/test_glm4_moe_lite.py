"""GLM-4.7-Flash（glm4_moe_lite）的接入。

* 不需要 torch 的部分（规格、清单、各节点要拉的张量）在哪都跑；
* 执行层与 transformers 官方实现逐层对照，需要 torch + transformers ≥ 5（有 glm4_moe_lite）—— 缺就跳过。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CFG = ROOT / "glm47flash" / "config.json"
PLAN = ROOT / "glm47flash" / "plan_petals_cap8.json"


def _cfg() -> dict:
    if not CFG.exists():
        pytest.skip("缺 glm47flash/config.json")
    return json.loads(CFG.read_text(encoding="utf-8"))


def test_spec_matches_petals_sizing():
    """规划器口径：全装一层 1.27GB、KV 每 token 每层 20KiB —— 与 Petals 方案的 layer_gb_bf16 /
    kv_bytes_per_token_layer 一致（展开后的 MLA K/V）。"""
    from p2pmoe.planner.hf_config import model_spec_from_hf

    spec, info = model_spec_from_hf(_cfg(), ctx_max=16384)
    assert (spec.n_layers, spec.n_experts, spec.top_k) == (47, 64, 4)
    assert spec.weight_gb_per_layer(64) == pytest.approx(1.271, abs=0.002)
    assert spec.kv_gb_per_layer * 1e9 / 16384 == pytest.approx(20480)
    assert info.shared_intermediate == 1536


def test_keys_per_node_follow_glm_layout():
    from p2pmoe.deploy.fetch import keys_for_node
    from p2pmoe.planner.manifest import DeploymentManifest

    if not PLAN.exists():
        pytest.skip("缺 glm47flash/plan_petals_cap8.json")
    cfg = _cfg()
    man = DeploymentManifest.from_json(PLAN.read_text(encoding="utf-8"))
    head = keys_for_node(man, "N06", config=cfg)               # 前段，第 1–8 层
    assert "model.embed_tokens.weight" in head and "lm_head.weight" not in head
    assert "model.layers.0.mlp.gate_proj.weight" in head       # dense 第 0 层
    assert not any(k.startswith("model.layers.0.mlp.experts.") for k in head)
    assert "model.layers.1.mlp.gate.e_score_correction_bias" in head
    assert "model.layers.1.mlp.shared_experts.down_proj.weight" in head
    assert "model.layers.7.mlp.experts.63.up_proj.weight" in head
    assert "model.layers.7.self_attn.kv_a_proj_with_mqa.weight" in head
    assert not any(".q_proj." in k or ".k_proj." in k for k in head)
    tail = keys_for_node(man, "N10", config=cfg)               # 后段末台，第 41–47 层
    assert {"model.norm.weight", "lm_head.weight"} <= tail
    assert "model.layers.46.mlp.experts.0.gate_proj.weight" in tail
    assert not any(k.startswith("model.layers.47.") for k in tail)   # MTP 层不拉
    # 一个 MoE 层：MLA 7（q_a / q_a_ln / q_b / kv_a / kv_a_ln / kv_b / o）+ norm 2 + 路由 2 + 共享 3 + 64 × 3
    assert len([k for k in head if k.startswith("model.layers.3.")]) == 7 + 2 + 2 + 3 + 64 * 3


def test_mtp_layer_is_rejected():
    from p2pmoe.runtime.glm4_moe_lite import GlmModelConfig, glm_keys
    from p2pmoe.runtime.weights import KeyPlan

    cfg = GlmModelConfig.from_hf(_cfg())
    with pytest.raises(ValueError, match="MTP"):
        glm_keys(KeyPlan(layer_experts={48: [0]}), cfg)


# --------------------------------------------------------------------------- #
def _tiny_hf(rope_interleave: bool):
    torch = pytest.importorskip("torch")
    tr = pytest.importorskip("transformers")
    if not hasattr(tr, "Glm4MoeLiteConfig"):
        pytest.skip("transformers 没有 glm4_moe_lite（要 ≥ 5）")
    torch.manual_seed(0)
    c = tr.Glm4MoeLiteConfig(
        vocab_size=97, hidden_size=64, intermediate_size=96, moe_intermediate_size=24,
        num_hidden_layers=5, num_attention_heads=4, num_key_value_heads=4, n_shared_experts=1,
        n_routed_experts=16, routed_scaling_factor=1.8, kv_lora_rank=16, q_lora_rank=24,
        qk_rope_head_dim=8, v_head_dim=16, qk_nope_head_dim=12, n_group=1, topk_group=1,
        num_experts_per_tok=4, first_k_dense_replace=1, norm_topk_prob=True,
        max_position_embeddings=256, rope_theta=10000.0, rms_norm_eps=1e-5,
        tie_word_embeddings=False, attn_implementation="eager", rope_interleave=rope_interleave)
    m = tr.Glm4MoeLiteForCausalLM(c).eval().float()
    with torch.no_grad():
        for n, p in m.named_parameters():
            p.copy_(1 + 0.1 * torch.randn_like(p) if "norm" in n else 0.2 * torch.randn_like(p))
        for layer in m.model.layers:
            if hasattr(layer.mlp, "gate"):
                b = layer.mlp.gate.e_score_correction_bias
                b.copy_(0.3 * torch.randn_like(b))
    # 拆回 checkpoint 的逐专家命名（transformers 加载时才把它们融成 gate_up_proj）
    I, w = c.moe_intermediate_size, {}
    for k, v in m.state_dict().items():
        if ".mlp.experts.gate_up_proj" in k:
            p = k.replace(".experts.gate_up_proj", "")
            for e in range(v.shape[0]):
                w[f"{p}.experts.{e}.gate_proj.weight"] = v[e, :I].clone()
                w[f"{p}.experts.{e}.up_proj.weight"] = v[e, I:].clone()
        elif ".mlp.experts.down_proj" in k:
            p = k.replace(".experts.down_proj", "")
            for e in range(v.shape[0]):
                w[f"{p}.experts.{e}.down_proj.weight"] = v[e].clone()
        else:
            w[k] = v.clone()
    d = json.loads(c.to_json_string())
    d["torch_dtype"] = "float32"
    return torch, m, w, d


@pytest.mark.parametrize("interleave", [True, False])
@pytest.mark.parametrize("fast", ["1", "0"])
def test_matches_transformers_prefill_and_decode(monkeypatch, interleave, fast):
    """两段（第 1–2 层 / 第 3–5 层）接起来，prefill 9 个 token 再 decode 4 步，
    逐步的 logits 与 transformers 的整模型前向（带 KV cache）一致。"""
    torch, m, w, d = _tiny_hf(interleave)
    from p2pmoe.runtime.glm4_moe_lite import GlmModelConfig, GlmSegmentModel, glm_keys
    from p2pmoe.runtime.weights import KeyPlan

    monkeypatch.setenv("P2P_FAST_DECODE", fast)
    cfg = GlmModelConfig.from_hf(d)
    assert cfg.mlp_layer_types[0] == "dense" and cfg.rope_interleave is interleave
    E = range(cfg.n_experts)
    spec = [{1: list(E), 2: list(E)}, {3: list(E), 4: list(E), 5: list(E)}]
    for i, s in enumerate(spec):     # 名单点到的 key 在 checkpoint 里都有
        assert not glm_keys(KeyPlan(s, with_embed=i == 0, with_lm_head=i == 1), cfg) - set(w)
    segs = [GlmSegmentModel(cfg, s, w) for s in spec]
    ids = torch.randint(0, 97, (1, 9))
    nxt = [torch.randint(0, 97, (1, 1)) for _ in range(4)]
    with torch.no_grad():
        o = m(ids, use_cache=True)
        ref, cache = [o.logits[0]], o.past_key_values
        for t in nxt:
            o = m(t, past_key_values=cache, use_cache=True)
            ref.append(o.logits[0])
            cache = o.past_key_values
        got = []
        h = segs[0].embed_tokens(ids[0].tolist())
        for s in segs:
            h, st = s.forward("r", h)
        got.append(segs[1].logits(h))
        assert sorted(st.layers) == [3, 4, 5]
        for t in nxt:
            h = segs[0].embed_tokens(t[0].tolist())
            for s in segs:
                h, st = s.forward("r", h, decode=True)
            got.append(segs[1].logits(h))
    for a, b in zip(got, ref):
        assert float((a - b).abs().max()) < 1e-4
    # 统计口径：每 token 每个 MoE 层的权重和 = routed_scaling_factor
    assert float(st.hist.sum()) == pytest.approx(1.8 * 3, rel=1e-5)
    assert segs[1].kv_tokens("r") == 13


def test_resident_subset_and_miss_policies():
    torch, m, w, d = _tiny_hf(True)
    from p2pmoe.runtime.glm4_moe_lite import GlmModelConfig, GlmSegmentModel

    cfg = GlmModelConfig.from_hf(d)
    x = torch.randn(6, cfg.d_model)
    full = GlmSegmentModel(cfg, {2: list(range(16))}, w)
    with torch.no_grad():
        y_full, st_full = full.forward("a", x)
    assert st_full.miss_token_layer == 0
    for pol in ("drop", "drop_noscale", "local_topk"):
        half = GlmSegmentModel(cfg, {2: list(range(0, 16, 2))}, w, miss_policy=pol)
        with torch.no_grad():
            y, st = half.forward("a", x)
        assert st.miss_token_layer > 0 and 0 < st.miss_mass <= 6
        assert bool(torch.isfinite(y).all())
        assert st.layers[2].count.sum() == 6 * 4       # 统计永远按全量路由


def test_dense_layer_is_aligned_and_sized():
    """激活统计 / 驻留集只有 46 个 MoE 层：补上 dense 第 1 层（均匀分布、不进识别），
    第 1 层权重按 dense 记（注意力 + SwiGLU 0.169GB），与驻留专家数无关。"""
    from p2pmoe.planner.experts import ActivationProfile
    from p2pmoe.planner.from_data import pad_dense_layers
    from p2pmoe.planner.hf_config import model_spec_from_hf

    spec, _ = model_spec_from_hf(_cfg())
    assert spec.dense_layers == {1}
    assert spec.layer_gb(1, 64) == pytest.approx(0.169, abs=0.001)
    assert spec.layer_gb(2, 64) == pytest.approx(1.271, abs=0.002)
    p = ActivationProfile("t", 46, 64, tuple(tuple([0.0] * 63 + [1.0]) for _ in range(46)))
    q = pad_dense_layers(p, spec.dense_layers, 47)
    assert q.n_layers == 47 and q.at(1) == tuple([1 / 64] * 64) and q.at(2)[63] == 1.0


def test_theory_plan_is_reproducible(tmp_path, monkeypatch):
    want = ROOT / "glm47flash" / "plan_glm47_1f_theory.json"
    es = ROOT / "glm47flash" / "glm47_expert_sets_90_union.json"
    if not (want.exists() and es.exists() and (ROOT / "task_glm47").exists()):
        pytest.skip("缺 glm47flash/ 或 task_glm47/")
    import examples.task_deploy as td

    out = tmp_path / "p.json"
    monkeypatch.setattr(sys, "argv", [
        "task_deploy", "--data", str(ROOT / "task_glm47"), "--config", str(CFG),
        "--phase", "prefill", "--tasks", "gsm8k=1,mbpp=1,no_robots=1", "--expert-sets", str(es),
        "--l0", "3", "--front-full", "--single-front", "--endpoint-w", "0", "--save-plan", str(out)])
    assert td.main() == 0
    assert out.read_bytes() == want.read_bytes()
    plan = json.loads(want.read_text())
    f = plan["segments"]["F0"]
    assert f["splits"] == [[1, 3]] and len(f["nodes"]) == 1
