"""2026-11-05 新拓扑（task4/、*_1105.json）上的按 task 分配方案：六步流水线（按 λ 最大余额配额）
+ 前段单节点（--front-max-nodes 1）+ 剩余节点按 task 补位（--fill-idle）。REPRODUCE 5.13。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    ("plan_1105_vllm_fill", ["--layer-times", "qwen3-30b/layer_bench_vllm_1105.json",
                             "--layer-time-key", "req@p512g512", "--rho-w", "1"]),
    ("plan_1105_doc_fill", ["--layer-times", "qwen3-30b/layer_bench_wash1_1105.json",
                            "--mem-prefill-tokens", "512"]),
    # 只按 decode 规划（不把 prefill 摊进每层耗时），对照 req@p512g512
    ("plan_1105_vllm_dec_fill", ["--layer-times", "qwen3-30b/layer_bench_vllm_1105.json",
                                 "--layer-time-key", "decode@p512g512", "--rho-w", "1"]),
]


@pytest.mark.parametrize("name,extra", CASES)
def test_plan_1105_fill_is_reproducible(tmp_path, monkeypatch, name, extra):
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
        "--l0", "8", "--front-full", extra[0], str(lb), *extra[2:],
        "--measured-mem", "--mem-margin-gb", "0", "--kv-reserve-gb", "2", "--endpoint-w", "0",
        "--front-max-nodes", "1", "--fill-idle", "--save-plan", str(out)])
    assert td.main() == 0
    assert out.read_bytes() == want.read_bytes()
    plan = json.loads(want.read_text())
    segs = plan["segments"].values()
    assert all(len(v["nodes"]) == 1 for v in segs if v["role"].startswith("front"))
    assert not any(v["role"] == "back:any" for v in segs)            # 不建原始模型路径
    assert len({n["node"] for n in plan["nodes"]}) == 15
    assert not plan["violations"]


def test_plan_1105_vllm_core_only_matches_fill_core(tmp_path, monkeypatch):
    """plan_1105_vllm.json = 同一条命令去掉 --fill-idle：段、层区间、专家与补位版的核心部分逐项相同，
    所以在补位版的权重上直接换清单就能跑，不用重新拉权重。"""
    want = ROOT / "qwen3-30b" / "plan_1105_vllm.json"
    fill = ROOT / "qwen3-30b" / "plan_1105_vllm_fill.json"
    es = ROOT / "qwen3-30b" / "expert_sets_90" / "expert_sets_90_union.json"
    lb = ROOT / "qwen3-30b" / "layer_bench_vllm_1105.json"
    if not (want.exists() and fill.exists() and lb.exists() and (ROOT / "task4").exists()):
        pytest.skip("缺 qwen3-30b/ 或 task4/ 数据")
    sys.path.insert(0, str(ROOT))
    import examples.task_deploy as td

    out = tmp_path / "p.json"
    monkeypatch.setattr(sys, "argv", [
        "task_deploy", "--data", str(ROOT / "task4"), "--config", "qwen3-30b-a3b",
        "--phase", "prefill", "--tasks", "gsm8k=1,mbpp=1,no_robots=1", "--expert-sets", str(es),
        "--l0", "8", "--front-full", "--layer-times", str(lb), "--layer-time-key", "req@p512g512",
        "--rho-w", "1", "--measured-mem", "--mem-margin-gb", "0", "--kv-reserve-gb", "2",
        "--endpoint-w", "0", "--front-max-nodes", "1", "--save-plan", str(out)])
    assert td.main() == 0
    assert out.read_bytes() == want.read_bytes()
    core = json.loads(want.read_text())
    full = json.loads(fill.read_text())
    by = {n["node"]: n for n in full["nodes"]}
    for n in core["nodes"]:
        m = by[n["node"]]
        assert (m["segment"], m["layer_range"], m["layers"]) == (n["segment"], n["layer_range"], n["layers"])


def test_plan_1105_vllm_dec_core_only_is_reproducible(tmp_path, monkeypatch):
    want = ROOT / "qwen3-30b" / "plan_1105_vllm_dec.json"
    es = ROOT / "qwen3-30b" / "expert_sets_90" / "expert_sets_90_union.json"
    lb = ROOT / "qwen3-30b" / "layer_bench_vllm_1105.json"
    if not (want.exists() and lb.exists() and (ROOT / "task4").exists()):
        pytest.skip("缺 qwen3-30b/ 或 task4/ 数据")
    sys.path.insert(0, str(ROOT))
    import examples.task_deploy as td

    out = tmp_path / "p.json"
    monkeypatch.setattr(sys, "argv", [
        "task_deploy", "--data", str(ROOT / "task4"), "--config", "qwen3-30b-a3b",
        "--phase", "prefill", "--tasks", "gsm8k=1,mbpp=1,no_robots=1", "--expert-sets", str(es),
        "--l0", "8", "--front-full", "--layer-times", str(lb), "--layer-time-key", "decode@p512g512",
        "--rho-w", "1", "--measured-mem", "--mem-margin-gb", "0", "--kv-reserve-gb", "2",
        "--endpoint-w", "0", "--front-max-nodes", "1", "--save-plan", str(out)])
    assert td.main() == 0
    assert out.read_bytes() == want.read_bytes()
