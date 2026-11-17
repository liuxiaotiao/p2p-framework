"""先定前段（--front-first）：按计算定唯一前段 → 按组（轮次）围绕它选后段首节点 → 按组从首节点建链
→ 按组收紧。REPRODUCE 5.16。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VLLM = ["--layer-times", "qwen3-30b/layer_bench_vllm_1105.json", "--layer-time-key", "req@p512g512"]
DOC = ["--layer-times", "qwen3-30b/layer_bench_wash1_1105.json", "--mem-prefill-tokens", "512"]
CASES = [("plan_1105_vllm_ff", VLLM), ("plan_1105_doc_ff", DOC),
         ("plan_1105_vllm_ff_a95", VLLM + ["--avail", "0.95"]),
         ("plan_1105_doc_ff_a95", DOC + ["--avail", "0.95"])]


def _run(monkeypatch, out, extra):
    es = ROOT / "qwen3-30b" / "expert_sets_90" / "expert_sets_90_union.json"
    sys.path.insert(0, str(ROOT))
    import examples.task_deploy as td

    lb = [x if not x.startswith("qwen3-30b/") else str(ROOT / x) for x in extra]
    monkeypatch.setattr(sys, "argv", [
        "task_deploy", "--data", str(ROOT / "task4"), "--config", "qwen3-30b-a3b",
        "--phase", "prefill", "--tasks", "gsm8k=1,mbpp=1,no_robots=1", "--expert-sets", str(es),
        "--l0", "2", "--front-full", *lb,
        "--measured-mem", "--mem-margin-gb", "0", "--kv-reserve-gb", "2", "--endpoint-w", "0",
        "--front-first", "--save-plan", str(out)])
    return td.main()


@pytest.mark.parametrize("name,extra", CASES)
def test_front_first_plan_is_reproducible(tmp_path, monkeypatch, capsys, name, extra):
    want = ROOT / "qwen3-30b" / f"{name}.json"
    if not (want.exists() and (ROOT / "task4").exists() and (ROOT / extra[1]).exists()):
        pytest.skip("缺 qwen3-30b/ 或 task4/ 数据")
    out = tmp_path / "p.json"
    assert _run(monkeypatch, out, extra) == 0
    assert out.read_bytes() == want.read_bytes()
    text = capsys.readouterr().out
    assert "[先定前段]" in text and "[首节点] 第 1 组" in text
    plan = json.loads(want.read_text())
    f = plan["segments"]["F0"]
    assert len(f["nodes"]) == 1 and f["splits"] == [[1, 2]]
    assert not plan["violations"]


def test_front_is_fastest_node_and_first_group_heads_are_its_neighbours():
    """vLLM 表里第 1–2 层算得最快的是 N01（A30）；第 1 组首节点都在它所在的站点（RUTG，0.05ms 内）。"""
    want = ROOT / "qwen3-30b" / "plan_1105_vllm_ff.json"
    if not want.exists():
        pytest.skip("缺 qwen3-30b/plan_1105_vllm_ff.json")
    plan = json.loads(want.read_text())
    assert plan["segments"]["F0"]["nodes"] == ["N01"]
    heads = {sid: v["head"] for sid, v in plan["segments"].items()
             if v["role"].startswith("back:") and sid.endswith("0")}
    assert set(heads.values()) <= {"N02", "N03", "N04", "N05"}
