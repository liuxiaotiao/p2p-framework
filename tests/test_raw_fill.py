"""--quota-equal + --raw-fill：每个 task 均分 ⌊N_work / task 数⌋ 条（照原算法建、盲绑任意组合），
均分剩下的节点建原始模型路径（后段全部专家，进共享池 any；前段并入前段池）。

* planner/exact.py 的精确放置 = 暴力最优（原始模型后段用它）；
* 原始模型后段的驻留集：前 L₀ 层抄前段，之后全部专家；
* 协调器（动态）：专属后段全忙 → 溢出到原始模型路径；它放出来时接所有池里等得最久的那条；
  统计仍按识别出来的 task；
* qwen3-30b/plan_vllm_eq_raw.json、plan_doc_v2_eq_raw.json 逐字节可复现（REPRODUCE 5.12）。
"""

from __future__ import annotations

import json
import sys
from math import ceil
from pathlib import Path

import pytest

from p2pmoe.planner.exact import exact_path
from p2pmoe.planner.experts import ExpertPlacement
from p2pmoe.planner.raw import raw_placement
from p2pmoe.planner.types import Objective
from p2pmoe.runtime.coordinator import SHARED_POOL, Coordinator

from test_beam_complete_solutions import _brute, _instance, _seg_score
from test_serving_loop import FakeManifest, FakePool, classify, run_to_completion

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# 1. 精确放置
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("small", [False, True])
@pytest.mark.parametrize("l0", [4, 8])
@pytest.mark.parametrize("seed", range(4))
def test_exact_path_matches_brute_force(seed, l0, small):
    nm, net, spec = _instance(seed, 6, l0, small)
    obj = Objective()
    n_min = max(1, ceil(spec.total_gb() / max(v.usable_gb for v in nm.values()) - 1e-9))
    want, _ = _brute(spec, nm, net, obj, min(n_min + 1, len(nm)))
    seg = exact_path(spec, sorted(nm), nm, net, obj, hop_slack=1)
    if want == float("inf"):
        assert seg is None
        return
    assert seg is not None
    assert _seg_score(seg, spec, nm, net, obj) == pytest.approx(want, abs=1e-6)
    assert seg.splits[0][0] == spec.layer_lo and seg.splits[-1][1] == spec.layer_hi
    assert all(b[0] == a[1] + 1 for a, b in zip(seg.splits, seg.splits[1:]))
    assert all(spec.resident_gb(lo, hi) <= nm[v].usable_gb + 1e-9
               for v, (lo, hi) in zip(seg.nodes, seg.splits))


def test_raw_placement_copies_front_then_full() -> None:
    front = ExpertPlacement(name="u", n_layers=6, sets=tuple(frozenset({0, 1}) for _ in range(6)),
                            achieved_coverage=(1.0,) * 6)
    p = raw_placement(front, 6, 8, 2)
    assert p.at(1) == p.at(2) == frozenset({0, 1})
    assert all(p.at(l) == frozenset(range(8)) for l in range(3, 7))


# --------------------------------------------------------------------------- #
# 2. 协调器：专属后段全忙 → 原始模型路径
# --------------------------------------------------------------------------- #
def make() -> tuple[Coordinator, FakePool]:
    man = FakeManifest(n_front=4, backs={"X": 1, "Y": 1, SHARED_POOL: 1})
    c = Coordinator(man, baselines={"X": 0.1, "Y": 0.1}, priors={"X": 0.5, "Y": 0.5},
                    alarm_factor=99.0)
    c.pool = FakePool()
    c.max_tokens = 2
    return c, c.pool


def test_own_pool_first_then_overflow_to_raw_path() -> None:
    c, _ = make()
    for i in range(3):
        c.submit(f"r{i}", [i])
    classify(c, "r0", "X")
    r0 = c.records["r0"]
    assert (r0.back, r0.back_pool, r0.task) == ("BX0", "X", "X")
    classify(c, "r1", "X")                       # X 唯一一条占着 → 原始模型路径
    r1 = c.records["r1"]
    assert (r1.back, r1.back_pool, r1.task) == ("Bany0", SHARED_POOL, "X")
    assert r1.correct is None or r1.task == "X"   # 统计仍按识别结果
    assert any("溢出到原始模型路径" in e for e in r1.events)
    classify(c, "r2", "X")                       # 两边都忙 → 排队
    assert c.records["r2"].back == ""


def test_raw_path_released_serves_longest_waiter_of_any_pool() -> None:
    c, _ = make()
    for i in range(4):
        c.submit(f"r{i}", [i])
    classify(c, "r0", "X")
    classify(c, "r1", "Y")
    classify(c, "r2", "Y")                       # → Bany0
    classify(c, "r3", "X")                       # 排队（X）
    assert c.records["r2"].back == "Bany0" and c.records["r3"].back == ""
    run_to_completion(c, "r2")                   # 原始模型路径放出来 → 接 X 的队首
    r3 = c.records["r3"]
    assert (r3.back, r3.back_pool, r3.task) == ("Bany0", SHARED_POOL, "X")
    assert list(c.free_backs[SHARED_POOL]) == []


def test_own_back_released_serves_only_its_pool() -> None:
    c, _ = make()
    for i in range(4):
        c.submit(f"r{i}", [i])
    classify(c, "r0", "X")
    classify(c, "r1", "Y")
    classify(c, "r2", "Y")                       # → Bany0
    classify(c, "r3", "Y")                       # 排队（Y）
    run_to_completion(c, "r0")                   # X 的后段放出来 —— 不接 Y
    assert c.records["r3"].back == "" and list(c.free_backs["X"]) == ["BX0"]
    run_to_completion(c, "r2")                   # 原始模型路径放出来，放回共享池后再被 r3 接走
    assert c.records["r3"].back == "Bany0"


def test_overflowed_request_returns_back_to_shared_pool() -> None:
    c, _ = make()
    c.submit("r0", [0])
    c.submit("r1", [1])
    classify(c, "r0", "X")
    classify(c, "r1", "X")
    run_to_completion(c, "r1")
    assert list(c.free_backs[SHARED_POOL]) == ["Bany0"] and not c.free_backs["X"]


# --------------------------------------------------------------------------- #
# 3. 仓库里的方案文件
# --------------------------------------------------------------------------- #
CASES = [
    ("plan_vllm_eq_raw", ["--layer-times", "qwen3-30b/layer_bench_vllm.json",
                          "--layer-time-key", "req@p512g512"], 15),
    ("plan_doc_v2_eq_raw", ["--layer-times", "qwen3-30b/layer_bench_wash1.json",
                            "--mem-prefill-tokens", "512"], 13),
]


@pytest.mark.parametrize("name,lb_args,n_used", CASES)
def test_eq_raw_plan_file_is_reproducible(tmp_path, monkeypatch, capsys, name, lb_args, n_used):
    want = ROOT / "qwen3-30b" / f"{name}.json"
    lb = ROOT / lb_args[1]
    es = ROOT / "qwen3-30b" / "expert_sets_90" / "expert_sets_90_union.json"
    if not (want.exists() and lb.exists() and es.exists() and (ROOT / "task3").exists()):
        pytest.skip("缺 qwen3-30b/ 或 task3/ 数据")
    sys.path.insert(0, str(ROOT))
    import examples.task_deploy as td

    out = tmp_path / "p.json"
    monkeypatch.setattr(sys, "argv", [
        "task_deploy", "--data", str(ROOT / "task3"), "--config", "qwen3-30b-a3b",
        "--phase", "prefill", "--tasks", "gsm8k=1,mbpp=1,no_robots=1", "--expert-sets", str(es),
        "--l0", "8", "--front-full", lb_args[0], str(lb), *lb_args[2:],
        "--measured-mem", "--mem-margin-gb", "0", "--kv-reserve-gb", "2", "--endpoint-w", "0",
        "--quota-equal", "--raw-fill", "--save-plan", str(out)])
    assert td.main() == 0
    assert out.read_bytes() == want.read_bytes()
    assert "⌊5/3⌋ = 1 条" in capsys.readouterr().out
    plan = json.loads(want.read_text())
    roles = [n["role"] for n in plan["nodes"]]
    fronts = {n["segment"] for n in plan["nodes"] if n["role"] == "front"}
    backs = {n["segment"]: n["role"] for n in plan["nodes"] if n["role"].startswith("back:")}
    assert sorted(backs.values()) == ["back:any", "back:gsm8k", "back:mbpp", "back:no_robots"]
    assert len(fronts) == 4
    assert len(plan["pairings"]) == 4 * 4                       # 盲绑：全网格
    assert all(len(l["experts"]) == 128 for n in plan["nodes"] if n["role"] == "back:any"
               for l in n["layers"])
    assert len(set(n["node"] for n in plan["nodes"])) == n_used
    assert "front-standby" not in roles
    assert not plan["violations"]
    assert {plan["segments"][s].get("origin") for s in fronts} == {None, "raw"}


def test_fill_idle_and_raw_fill_are_exclusive(monkeypatch):
    sys.path.insert(0, str(ROOT))
    import examples.task_deploy as td

    monkeypatch.setattr(sys, "argv", ["task_deploy", "--fill-idle", "--raw-fill"])
    with pytest.raises(SystemExit, match="二选一"):
        td.main()
