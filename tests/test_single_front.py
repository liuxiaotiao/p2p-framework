"""单前段（--single-front）：只建一条前段、放一台，所有请求共用；后段照旧建并收紧。

* 规划：Step 4–6 不做，前段 1 台、层 1..L₀，清单里记 slots = 后段条数，组合矩阵 1 × 后段数；
* 协调器：空闲队列里放 slots 个前段名额，同时在途的请求数 = 后段条数；
* qwen3-30b/plan_1105_*_1f.json 逐字节可复现（REPRODUCE 5.15）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from p2pmoe.runtime.coordinator import Coordinator

from test_serving_loop import FakeManifest, FakePool, classify, run_to_completion

ROOT = Path(__file__).resolve().parents[1]


def make(slots: int):
    man = FakeManifest(n_front=1, backs={"X": 2, "Y": 1})
    man.segments["F0"]["slots"] = slots
    c = Coordinator(man, baselines={"X": 0.1, "Y": 0.1}, priors={"X": 0.6, "Y": 0.4},
                    alarm_factor=99.0)
    c.pool = FakePool()
    c.max_tokens = 2
    return c


def test_one_front_serves_slots_requests_at_once():
    c = make(3)
    assert list(c.free_fronts) == ["F0", "F0", "F0"]
    rs = [c.submit(f"r{i}", [i]) for i in range(4)]
    assert [r.front for r in rs[:3]] == ["F0"] * 3 and rs[3].front == ""   # 第 4 条排队
    classify(c, "r0", "X")
    run_to_completion(c, "r0")
    assert rs[3].front == "F0"                                             # 名额还回来就接走


def test_default_is_one_request_per_front():
    man = FakeManifest(n_front=2, backs={"X": 1})
    c = Coordinator(man, baselines={}, priors={"X": 1.0})
    assert list(c.free_fronts) == ["F0", "F1"]


CASES = [
    ("plan_1105_vllm_1f", ["--layer-times", "qwen3-30b/layer_bench_vllm_1105.json",
                           "--layer-time-key", "req@p512g512"]),
    ("plan_1105_doc_1f", ["--layer-times", "qwen3-30b/layer_bench_wash1_1105.json",
                          "--mem-prefill-tokens", "512"]),
    # 理论计算（--theory-times：ms/层按 compute_capacity 反比，实测表只取显存）
    ("plan_1105_vllm_1f_theory", ["--layer-times", "qwen3-30b/layer_bench_vllm_1105.json",
                                  "--layer-time-key", "req@p512g512", "--theory-times"]),
    ("plan_1105_doc_1f_theory", ["--layer-times", "qwen3-30b/layer_bench_wash1_1105.json",
                                 "--mem-prefill-tokens", "512", "--theory-times"]),
]


@pytest.mark.parametrize("name,extra", CASES)
def test_single_front_plan_is_reproducible(tmp_path, monkeypatch, name, extra):
    want = ROOT / "qwen3-30b" / f"{name}.json"
    es = ROOT / "qwen3-30b" / "expert_sets_90" / "expert_sets_90_union.json"
    lb = ROOT / extra[1]
    if not (want.exists() and lb.exists() and es.exists() and (ROOT / "task4").exists()):
        pytest.skip("缺 qwen3-30b/ 或 task4/ 数据")
    sys.path.insert(0, str(ROOT))
    import examples.task_deploy as td

    out = tmp_path / "p.json"
    monkeypatch.setattr(sys, "argv", [
        "task_deploy", "--data", str(ROOT / "task4"), "--config", "qwen3-30b-a3b",
        "--phase", "prefill", "--tasks", "gsm8k=1,mbpp=1,no_robots=1", "--expert-sets", str(es),
        "--l0", "2", "--front-full", extra[0], str(lb), *extra[2:],
        "--measured-mem", "--mem-margin-gb", "0", "--kv-reserve-gb", "2", "--endpoint-w", "0",
        "--single-front", "--save-plan", str(out)])
    assert td.main() == 0
    assert out.read_bytes() == want.read_bytes()
    plan = json.loads(want.read_text())
    fronts = {s: v for s, v in plan["segments"].items() if v["role"].startswith("front")}
    backs = [s for s, v in plan["segments"].items() if v["role"].startswith("back:")]
    assert list(fronts) == ["F0"] and len(fronts["F0"]["nodes"]) == 1
    assert fronts["F0"]["splits"] == [[1, 2]]
    assert fronts["F0"]["slots"] == len(backs)
    assert len(plan["pairings"]) == len(backs) and not plan["violations"]


def test_vllm_plan_is_balanced_by_round():
    """单前段按轮次分组收紧：各 task 的第 1 条一组、第 2 条一组。vLLM 方案里第 2 组原来是
    mbpp 10.0 / no_robots 38.7 / gsm8k 83.5ms，收紧后三条都在 38–44ms。"""
    want = ROOT / "qwen3-30b" / "plan_1105_vllm_1f.json"
    if not want.exists():
        pytest.skip("缺 qwen3-30b/plan_1105_vllm_1f.json")
    segs = json.loads(want.read_text())["segments"]
    for k in (0, 1):
        d = [v["delay_ms"] for sid, v in segs.items()
             if v["role"].startswith("back:") and sid.endswith(str(k))]
        assert len(d) == 3
        assert (max(d) - min(d)) / sorted(d)[1] < 0.25, (k, d)


def test_tighten_takes_one_spec_per_segment():
    """按组收紧时一组里是不同 task 的段：spec 逐段给，长度对不上就报错。"""
    from p2pmoe.planner import tighten as tg
    from p2pmoe.planner.solver import segment_from_nodes
    from p2pmoe.planner.types import PlannerConfig

    from test_beam_complete_solutions import _instance

    nm, net, spec = _instance(0, 6, 8)
    ids = sorted(nm)
    s = segment_from_nodes(spec, [ids[0]], [(spec.layer_lo, spec.layer_hi)], nm, net)
    with pytest.raises(ValueError, match="spec 给了"):
        tg.tighten_lex([s, s], [spec], (0, 1e9), set(), nm, net, PlannerConfig())
    r = tg.tighten_lex([s], [spec], (0, 1e9), set(), nm, net, PlannerConfig())
    assert r.segments == [s]


def test_tighten_exact_switch(monkeypatch):
    """tighten_exact / single_front 开时 tighten_lex 调 exact_path，关时调 deploy_path。"""
    from p2pmoe.planner import tighten as tg
    from p2pmoe.planner.types import PlannerConfig

    from test_beam_complete_solutions import _instance

    nm, net, spec = _instance(0, 6, 8)
    seen = []
    monkeypatch.setattr(tg, "exact_path", lambda *a, **k: seen.append("exact") or None)
    monkeypatch.setattr(tg, "deploy_path", lambda *a, **k: seen.append("beam") or None)
    from p2pmoe.planner.solver import segment_from_nodes
    ids = sorted(nm)
    lo, hi = spec.layer_lo, spec.layer_hi
    s = segment_from_nodes(spec, [ids[0]], [(lo, hi)], nm, net)
    for cfg, want in ((PlannerConfig(), "beam"), (PlannerConfig(tighten_exact=True), "exact"),
                      (PlannerConfig(single_front=True), "exact")):
        seen.clear()
        tg.tighten_lex([s], spec, (s.delay_ms + 50, s.delay_ms + 60), set(ids[1:]), nm, net, cfg)
        assert seen and set(seen) == {want}


def test_theory_times_keep_topology_ms_and_hold_front_node():
    """--theory-times：方案里每台的计算按拓扑理论值；PyTorch 显存下第一次规划后段把
    三台 T4 吃光，重规划时硬保留 N13 给前段（日志里有这一行）。"""
    log = ROOT / "qwen3-30b" / "plan_1105_doc_1f_theory.log"
    plan = ROOT / "qwen3-30b" / "plan_1105_doc_1f_theory.json"
    if not (log.exists() and plan.exists()):
        pytest.skip("缺 plan_1105_doc_1f_theory")
    text = log.read_text(encoding="utf-8")
    assert "--theory-times" in text and "硬保留 ['N13']" in text
    p = json.loads(plan.read_text())
    assert p["segments"]["F0"]["nodes"] == ["N13"]
    # A30 两台装 46 层：理论 0.35ms/层 → 16.1ms
    assert p["segments"]["Bmbpp0"]["compute_ms"] == pytest.approx(16.1)
