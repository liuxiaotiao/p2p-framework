"""逐层计时（deploy/layer_bench.py）与规划器接入（from_data.apply_layer_times）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from p2pmoe.deploy import layer_bench as LB
from p2pmoe.runtime import layer_bench as RB
from p2pmoe.planner.from_data import apply_layer_times, load_topology

pytest.importorskip("torch")
from p2pmoe.sim.fake_checkpoint import TINY_QWEN3_MOE  # noqa: E402


def _run(node="X", **kw):
    args = dict(device="cpu", n_layers=2, ctx=(8, 16), prefill=(8,), decode_steps=3,
                repeats=2, warmup=1, node=node)
    args.update(kw)
    return RB.run_bench(TINY_QWEN3_MOE, **args)


def test_run_reports_every_requested_point():
    r = _run(resident=(None, 4))
    assert set(r["variants"]) == {"all", "4"}
    v = r["variants"]["all"]
    assert set(v["decode"]) == {"8", "16"} and set(v["prefill"]) == {"8"}
    s = v["decode"]["8"]
    assert s["n"] == 3 and s["median"] > 0 and s["std"] >= 0
    assert v["prefill"]["8"]["per_token_median"] == pytest.approx(v["prefill"]["8"]["median"] / 8, rel=1e-3)
    assert r["model"]["n_experts"] == TINY_QWEN3_MOE["num_experts"]


def test_weights_have_checkpoint_shapes():
    import torch
    from p2pmoe.runtime.torch_model import TorchModelConfig
    from p2pmoe.runtime.weights import KeyPlan, qwen_moe_keys

    cfg = TorchModelConfig.from_hf(TINY_QWEN3_MOE)
    w = RB.synth_weights(cfg, [1, 2], [0, 3], device="cpu", dtype=torch.float32)
    assert set(w) == qwen_moe_keys(KeyPlan(layer_experts={1: [0, 3], 2: [0, 3]}))


def test_refuses_when_gpu_memory_is_taken(monkeypatch):
    monkeypatch.setattr(RB, "_gpu_info", lambda device: {"name": "fake", "free_gib": 0.0})
    with pytest.raises(RB.BenchError, match="loaded"):
        _run()


def test_merge_groups_by_gpu_and_picks_the_key():
    a, b, c = _run("N01"), _run("N02"), _run("N03")
    for r, g in ((a, "A30"), (b, "A30"), (c, "T4")):
        r["gpu"]["name"] = g
    m = LB.merge([a, b, c])
    assert set(m["per_gpu"]) == {"A30", "T4"} and m["per_gpu"]["A30"]["nodes"] == ["N01", "N02"]
    assert "decode@8" in m["keys"] and "prefill@8" in m["keys"]
    ms = LB.per_node_ms(m, "decode@16")
    assert set(ms) == {"N01", "N02", "N03"}
    assert ms["N01"] == a["variants"]["all"]["decode"]["16"]["median"]
    assert m["per_gpu"]["A30"]["decode@16"]["n"] == 2


def test_cli_run_and_merge(tmp_path, capsys):
    cfgdir = tmp_path / "w"
    cfgdir.mkdir()
    (cfgdir / "config.json").write_text(json.dumps(TINY_QWEN3_MOE))
    f = tmp_path / "N01.json"
    assert LB.main(["run", "--model-dir", str(cfgdir), "--node", "N01", "--device", "cpu",
                    "--layers", "1", "--ctx", "8", "--prefill", "8", "--steps", "2",
                    "--repeats", "1", "--warmup", "0", "--out", str(f)]) == 0
    out = tmp_path / "m.json"
    assert LB.main(["merge", str(f), "--key", "decode@8", "--out", str(out)]) == 0
    m = json.loads(out.read_text())
    assert m["per_node"]["N01"]["decode@8"] > 0


def _topo(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({
        "nodes": [{"id": "A1", "compute_capacity": 100.0, "memory_capacity": 24, "role": "A30"},
                  {"id": "A2", "compute_capacity": 100.0, "memory_capacity": 24, "role": "A30"},
                  {"id": "T1", "compute_capacity": 40.0, "memory_capacity": 16, "role": "T4"},
                  {"id": "R1", "compute_capacity": 80.0, "memory_capacity": 24, "role": "RTX6000"}],
        "edges": [{"u": "A1", "v": "A2", "bandwidth": 10000, "propagation_delay": 1.0}]}))
    return load_topology(p)


def test_apply_layer_times_fallbacks(tmp_path):
    topo = _topo(tmp_path)
    new, rows = apply_layer_times(topo, {"A1": 2.0, "T1": 6.0})
    ms = {n.id: n.ms_per_layer for n in new.nodes}
    src = {r[0]: r[4] for r in rows}
    assert ms["A1"] == 2.0 and src["A1"] == "实测"
    assert ms["A2"] == 2.0 and src["A2"].startswith("同型号")          # 同为 A30
    # R1 没有同型号：k = median(2.0×100, 6.0×40) = 220 → 220/80
    assert ms["R1"] == pytest.approx(220 / 80, rel=1e-3) and src["R1"] == "按算力比例外推"
    assert [n.mem_gb for n in new.nodes] == [n.mem_gb for n in topo.nodes]


def test_apply_layer_times_without_measurements_is_identity(tmp_path):
    topo = _topo(tmp_path)
    new, _ = apply_layer_times(topo, {})
    assert new is topo


# --------------------------------------------------------------------------- #
# 经 agent：控制机一条命令，各 agent 在自己那里量完回传
# --------------------------------------------------------------------------- #
import threading  # noqa: E402

from p2pmoe.runtime.node import NodeServer  # noqa: E402

OPTS = {"ctx": [8], "prefill": [8], "decode_steps": 2, "warmup": 0, "repeats": 1, "n_layers": 1}


def _loaded_model(layers=(3, 4, 5), experts=(0, 1, 2, 5)):
    import torch
    from p2pmoe.runtime.torch_model import TorchModelConfig, TorchSegmentModel

    cfg = TorchModelConfig.from_hf(TINY_QWEN3_MOE)
    w = RB.synth_weights(cfg, layers, experts, device="cpu", dtype=torch.float32)
    return TorchSegmentModel(cfg, {l: list(experts) for l in layers}, w, device="cpu",
                             miss_policy="local_topk")


@pytest.fixture
def agents():
    """两个真在监听的 agent：a 已装权重（3 层），b 没装。"""
    srvs = {}
    for name in ("a", "b"):
        s = NodeServer(name, host="127.0.0.1", port=0)
        threading.Thread(target=s.serve_forever, daemon=True).start()
        srvs[name] = s
    srvs["a"].model = _loaded_model()
    yield srvs
    for s in srvs.values():
        s._stop.set()


def test_loaded_agent_times_its_own_layers(agents):
    r = agents["a"].layer_bench({"mode": "auto", **OPTS})
    assert r["ok"], r
    res = r["result"]
    assert res["setup"]["source"] == "loaded" and res["setup"]["n_layers"] == 3
    v = res["variants"]["all"]
    assert v["layers"] == [3, 5] and v["resident"] == 4
    assert v["decode"]["8"]["median"] > 0 and v["prefill"]["8"]["median"] > 0
    assert not agents["a"].model.active_requests(), "计时用的 KV 必须清掉"


def test_loaded_mode_needs_weights(agents):
    r = agents["b"].layer_bench({"mode": "loaded", **OPTS})
    assert not r["ok"] and "configure" in r["why"]


def test_loaded_agent_refuses_while_serving(agents):
    agents["a"].model.forward("real-req", __import__("torch").zeros(2, 64))
    r = agents["a"].layer_bench({"mode": "loaded", **OPTS})
    assert not r["ok"] and "在途" in r["why"]
    agents["a"].model.drop_kv("real-req")


def test_remote_collects_from_every_agent(agents, tmp_path, monkeypatch):
    """控制机：一条 run_remote，a 量已加载的层，b 用随机权重（cpu）。"""
    import p2pmoe.runtime.layer_bench as rbm

    real = rbm.run_bench
    monkeypatch.setattr(rbm, "run_bench", lambda cfg, **kw: real(cfg, **{**kw, "device": "cpu"}))
    addrs = {n: ("127.0.0.1", s.port) for n, s in agents.items()}
    seen = []
    got, bad = LB.run_remote(addrs, hf_config=TINY_QWEN3_MOE, opts=OPTS, timeout=60,
                             on_done=lambda n, r, w: seen.append(n))
    assert not bad, bad
    assert sorted(seen) == ["a", "b"]
    assert got["a"]["setup"]["source"] == "loaded"
    assert got["b"]["setup"]["source"] == "synthetic"
    m = LB.merge(list(got.values()), key="decode@8")
    assert set(LB.per_node_ms(m, "decode@8")) == {"a", "b"}
    assert m["per_node"]["a"]["source"] == "loaded"


def test_remote_reports_unreachable_agents():
    got, bad = LB.run_remote({"x": ("127.0.0.1", 1)}, hf_config=None, opts=OPTS, timeout=2)
    assert not got and "x" in bad


def test_cli_remote(agents, tmp_path, capsys):
    addrs = ",".join(f"{n}=127.0.0.1:{s.port}" for n, s in agents.items() if n == "a")
    out = tmp_path / "m.json"
    rc = LB.main(["remote", "--agents", addrs, "--mode", "loaded", "--ctx", "8", "--prefill", "8",
                  "--steps", "2", "--repeats", "1", "--warmup", "0", "--timeout", "60",
                  "--out-dir", str(tmp_path / "d"), "--out", str(out), "--key", "decode@8"])
    assert rc == 0
    assert (tmp_path / "d" / "a.json").exists()
    assert json.loads(out.read_text())["per_node"]["a"]["decode@8"] > 0
    assert "已加载权重" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# 显存：实测 → 规划器
# --------------------------------------------------------------------------- #
def test_bench_reports_kv_and_weight_bytes():
    r = _run()
    m = r["variants"]["all"]["memory"]
    c = TINY_QWEN3_MOE
    assert m["kv_bytes_per_token_layer"] == 2 * c["num_key_value_heads"] * c["head_dim"] * 4
    d, mi = c["hidden_size"], c["moe_intermediate_size"]
    assert m["expert_gb"] == pytest.approx(3 * d * mi * 4 / 1e9, rel=1e-6)
    assert m["weights_per_layer_gb"] == pytest.approx(
        m["base_per_layer_gb"] + c["num_experts"] * m["expert_gb"], rel=1e-6)


def test_loaded_bench_reports_per_layer_weights(agents):
    res = agents["a"].layer_bench({"mode": "loaded", **OPTS})["result"]
    m = res["variants"]["all"]["memory"]
    assert set(m["weights_by_layer_gb"]) == {"3", "4", "5"}
    assert m["weights_total_gb"] == pytest.approx(sum(m["weights_by_layer_gb"].values()), rel=1e-3)


def test_merge_carries_memory():
    a = _run("N01")
    a["device_memory"] = {"total_gb": 25.3, "context_gb": 0.4}
    a["variants"]["all"]["memory"].update(overhead_ratio=1.02, prefill_extra_gb={"128": 0.1, "2048": 0.7})
    m = LB.merge([a])
    mem = LB.per_node_memory(m)["N01"]
    assert mem["total_gb"] == 25.3 and mem["prefill_extra_gb"] == 0.7 and mem["prefill_extra_at"] == 2048
    assert m["per_gpu"]["cpu"]["memory"]["total_gb"]["mean"] == 25.3


def test_apply_layer_memory(tmp_path):
    from p2pmoe.planner.from_data import apply_layer_memory

    topo = _topo(tmp_path)          # A1/A2 A30、T1 T4、R1 RTX6000，标称 24/24/16/24
    meas = {"A1": {"total_gb": 25.3, "context_gb": 0.4, "prefill_extra_gb": 0.6, "overhead_ratio": 1.0},
            "R1": {"total_gb": 23.59, "context_gb": 0.5, "prefill_extra_gb": 0.8, "overhead_ratio": 1.1}}
    new, rows = apply_layer_memory(topo, meas, margin_gb=0.5)
    u = {n.id: n.usable_gb for n in new.nodes}
    assert u["A1"] == pytest.approx(25.3 - 0.4 - 0.6 - 0.5, abs=1e-3)
    assert u["R1"] == pytest.approx((23.59 - 0.5 - 0.8 - 0.5) / 1.1, abs=1e-3)
    assert u["A2"] == pytest.approx(u["A1"], abs=1e-3)                 # 同型号均值
    t1 = next(n for n in new.nodes if n.id == "T1")
    assert t1.mem_gb == 16.0                                            # 容量没测：保留标称
    assert "标称" in next(r[4] for r in rows if r[0] == "T1")


def test_head_and_tail_extras_count_only_at_the_ends():
    from p2pmoe.planner.types import SegmentSpec

    s = SegmentSpec(kind="back", task="t", layer_lo=9, layer_hi=12, gb_per_layer=lambda l: 1.0,
                    kv_gb_per_layer=0.0, head_extra_gb=0.3, tail_extra_gb=0.7)
    assert s.resident_gb(9, 12) == pytest.approx(5.0)
    assert s.resident_gb(10, 11) == pytest.approx(2.0)
    assert s.resident_gb(9, 10) == pytest.approx(2.3)
    assert s.resident_gb(11, 12) == pytest.approx(2.7)


@pytest.mark.parametrize("exhaustive", [False, True])
def test_solver_respects_the_tail_extra(exhaustive):
    """一台正好装下 4 层权重：不计输出头时一台成段；计上就必须两台。"""
    from p2pmoe.planner.network import MeasurementCache
    from p2pmoe.planner.solver import deploy_path
    from p2pmoe.planner.types import Node, Objective, SegmentSpec
    from p2pmoe.sim.network import SimNetwork

    nm = {f"n{i}": Node(id=f"n{i}", tier="x", mem_gb=4.0, ms_per_layer=1.0, reserve_gb=0.0)
          for i in range(3)}
    net = MeasurementCache(SimNetwork(sorted(nm), seed=0), k=8, j_cap_ms=1e9)
    base = dict(kind="back", task="t", layer_lo=9, layer_hi=12, gb_per_layer=lambda l: 1.0,
                kv_gb_per_layer=0.0)
    one = deploy_path(SegmentSpec(**base), sorted(nm), nm, net, Objective(),
                      exhaustive_short=exhaustive)
    two = deploy_path(SegmentSpec(**base, tail_extra_gb=0.6), sorted(nm), nm, net, Objective(),
                      exhaustive_short=exhaustive)
    assert one is not None and one.hops == 0
    assert two is not None and two.hops == 1
    assert two.splits[-1][1] - two.splits[-1][0] + 1 <= 3      # tail 那台最多 3 层 + 输出头


def test_length_ranges():
    assert LB._ints("128:1024:128") == [128, 256, 384, 512, 640, 768, 896, 1024]
    assert LB._ints("128:1024:128,2048") == [128, 256, 384, 512, 640, 768, 896, 1024, 2048]
    assert LB._ints("128:512") == [128, 256, 384, 512]          # 步长省略 = 起点
    assert LB._ints("1024,128,1024") == [128, 1024]             # 去重、排序
    for bad in ("128:64:8", "0:128:8", "128:1024:0", "1:2:3:4"):
        with pytest.raises(ValueError):
            LB._ints(bad)
    assert RB.DEFAULT_CTX[:8] == tuple(range(128, 1025, 128)) and RB.DEFAULT_CTX[-1] == 2048
