"""Step 3 的建段顺序（PlannerConfig.build_order）。

bottleneck（默认）：公平比最低的池先建 —— 分得最少的池先挑节点。之前（constrained，
形态最少的池先建）no_robots 总是最后建、只能捡剩下的 T4/RTX，通道比别的慢一截。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ES = ROOT / "qwen3-30b" / "expert_sets_90" / "expert_sets_90_union.json"
LB = ROOT / "qwen3-30b" / "layer_bench_wash1.json"


def _run(monkeypatch, out, *extra):
    if not (ES.exists() and LB.exists() and (ROOT / "task3").exists()):
        pytest.skip("缺 qwen3-30b/ 或 task3/ 数据")
    sys.path.insert(0, str(ROOT))
    import examples.task_deploy as td

    monkeypatch.setattr(sys, "argv", [
        "task_deploy", "--data", str(ROOT / "task3"), "--config", "qwen3-30b-a3b",
        "--phase", "prefill", "--tasks", "gsm8k=1,mbpp=1,no_robots=1", "--expert-sets", str(ES),
        "--l0", "8", "--front-full", "--layer-times", str(LB), "--measured-mem",
        "--mem-prefill-tokens", "512", "--mem-margin-gb", "0", "--kv-reserve-gb", "2",
        "--endpoint-w", "0", "--save-plan", str(out), *extra])
    return td.main()


def test_defaults():
    from p2pmoe.planner.types import PlannerConfig
    from p2pmoe.sim.scenario import appendix_c

    assert PlannerConfig().build_order == "bottleneck"
    assert appendix_c(seed=7).cfg.build_order == "constrained"   # 算例 C 钉在旧顺序


def test_bottleneck_builds_the_scarcest_pool_first(tmp_path, monkeypatch, capsys):
    out = tmp_path / "p.json"
    assert _run(monkeypatch, out) == 0
    log = capsys.readouterr().out
    assert "瓶颈池优先" in log and ": no_robots → gsm8k → mbpp" in log
    plan = json.loads(out.read_text())
    nr = [n["node"] for n in plan["nodes"] if n["role"] == "back:no_robots"]
    assert nr and all(n in {f"N0{i}" for i in range(1, 9)} for n in nr)   # 拿到的是 A30


def test_constrained_keeps_the_old_order(tmp_path, monkeypatch, capsys):
    out = tmp_path / "p.json"
    assert _run(monkeypatch, out, "--build-order", "constrained") == 0
    assert "受约束者优先）: gsm8k → mbpp → no_robots" in capsys.readouterr().out


def test_bn_plan_file_is_reproducible(tmp_path, monkeypatch, capsys):
    want = ROOT / "qwen3-30b" / "plan_doc_v2_bn.json"
    if not want.exists():
        pytest.skip("缺 qwen3-30b/plan_doc_v2_bn.json")
    out = tmp_path / "p.json"
    assert _run(monkeypatch, out) == 0
    assert out.read_bytes() == want.read_bytes()


def test_bad_value_raises(monkeypatch):
    from p2pmoe.planner.pipeline import plan
    from p2pmoe.sim.scenario import UNION_EXPERTS, appendix_c

    st = appendix_c(seed=7)
    st.cfg.build_order = "alphabetical"
    with pytest.raises(ValueError, match="build_order"):
        plan(st.nodes, st.model, st.tasks, UNION_EXPERTS, st.net, st.cfg, st.p_curve)


def test_bn_fill_plan_file_is_reproducible(tmp_path, monkeypatch, capsys):
    """plan_doc_v2_bn 的 4 条主通道不动，剩下的 N13–N15 由补位（--fill-idle）再建一条。"""
    want = ROOT / "qwen3-30b" / "plan_doc_v2_bn_fill.json"
    if not want.exists():
        pytest.skip("缺 qwen3-30b/plan_doc_v2_bn_fill.json")
    out = tmp_path / "p.json"
    assert _run(monkeypatch, out, "--fill-idle") == 0
    assert out.read_bytes() == want.read_bytes()
    core = json.loads((ROOT / "qwen3-30b" / "plan_doc_v2_bn.json").read_text())
    full = json.loads(want.read_text())
    key = lambda p: {(n["node"], n["role"], tuple(n["layer_range"])) for n in p["nodes"]}
    assert key(core) <= key(full)                       # 主流程的 12 台原样保留
    assert {n["node"] for n in full["nodes"]} == {f"N{i:02d}" for i in range(1, 16)}
