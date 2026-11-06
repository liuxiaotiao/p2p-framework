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
