"""tighten_lex 不能让一台节点进两条段。

回归场景（vLLM 实测、θ=1、η=1、w_cap=14 的规划）：前段收紧时，过快的 F[N11] 重建成
F[N10+N11] —— 拿走了过慢的 F[N10] 的节点；F[N10] 随后重建失败，旧代码「建不出就退回原段」，
于是 N10 同时属于两条前段，清单校验报「[排他] 节点 N10 同时属于 F0 与 F1」，却照样存盘。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import p2pmoe.planner.tighten as TG
from p2pmoe.planner.types import Node, PlannerConfig, Segment

ROOT = Path(__file__).resolve().parents[1]


def _seg(nodes, ms):
    return Segment(kind="front", task=None, nodes=tuple(nodes),
                   splits=tuple((i + 1, i + 1) for i in range(len(nodes))),
                   compute_ms=ms, hop_ms=0.0)


def test_failed_rebuild_cannot_fall_back_onto_taken_nodes(monkeypatch):
    fast, slow = _seg(["a"], 10.0), _seg(["b"], 30.0)
    calls = []

    def fake(spec, avail, *a, **k):
        calls.append(sorted(avail))
        # 第一次（重建过快者）拿走 b；第二次（重建过慢者）建不出来
        return _seg(["a", "b"], 20.0) if len(calls) == 1 else None

    monkeypatch.setattr(TG, "deploy_path", fake)
    nodes = {v: Node(id=v, tier="x", mem_gb=24.0, ms_per_layer=1.0) for v in "abc"}
    res = TG.tighten_lex([fast, slow], spec=None, band=(15.0, 25.0), free_nodes=set(),
                         nodes=nodes, net=None, cfg=PlannerConfig())
    used = [v for s in res.segments for v in s.nodes]
    assert len(used) == len(set(used))
    assert res.segments == [fast, slow] and res.rolled_back == 1 and res.committed == 0


def test_failed_rebuild_may_keep_its_own_untouched_nodes(monkeypatch):
    """原段的节点没被别人拿走时，建不出来照旧退回原段（行为不变）。"""
    fast, slow = _seg(["a"], 10.0), _seg(["b"], 30.0)
    calls = []

    def fake(spec, avail, *a, **k):
        calls.append(1)
        return _seg(["c"], 20.0) if len(calls) == 1 else None

    monkeypatch.setattr(TG, "deploy_path", fake)
    nodes = {v: Node(id=v, tier="x", mem_gb=24.0, ms_per_layer=1.0) for v in "abc"}
    res = TG.tighten_lex([fast, slow], spec=None, band=(15.0, 25.0), free_nodes={"c"},
                         nodes=nodes, net=None, cfg=PlannerConfig())
    assert [s.nodes for s in res.segments] == [("c",), ("b",)]
    assert res.free_nodes == {"a"}


def test_vllm_wide_band_plan_is_valid(tmp_path, monkeypatch, capsys):
    lb = ROOT / "qwen3-30b" / "layer_bench_vllm.json"
    es = ROOT / "qwen3-30b" / "expert_sets_90" / "expert_sets_90_union.json"
    if not (lb.exists() and es.exists() and (ROOT / "task3").exists()):
        pytest.skip("缺 qwen3-30b/ 或 task3/ 数据")
    sys.path.insert(0, str(ROOT))
    import examples.task_deploy as td

    out = tmp_path / "p.json"
    monkeypatch.setattr(sys, "argv", [
        "task_deploy", "--data", str(ROOT / "task3"), "--config", "qwen3-30b-a3b",
        "--phase", "prefill", "--tasks", "gsm8k=1,mbpp=1,no_robots=1", "--expert-sets", str(es),
        "--l0", "8", "--front-full", "--layer-times", str(lb), "--layer-time-key", "req@p512g512",
        "--measured-mem", "--mem-margin-gb", "0", "--kv-reserve-gb", "2", "--endpoint-w", "0",
        "--theta", "1.0", "--eta", "1.0", "--w-cap", "14", "--save-plan", str(out)])
    assert td.main() == 0
    used = [n["node"] for n in json.loads(out.read_text())["nodes"]]
    assert len(used) == len(set(used))
    assert "校验通过" in capsys.readouterr().out


def test_task_deploy_refuses_to_save_an_invalid_manifest(tmp_path, monkeypatch, capsys):
    es = ROOT / "qwen3-30b" / "expert_sets_90" / "expert_sets_90_union.json"
    if not (es.exists() and (ROOT / "task3").exists()):
        pytest.skip("缺 qwen3-30b/ 或 task3/ 数据")
    sys.path.insert(0, str(ROOT))
    import examples.task_deploy as td
    import p2pmoe.planner.manifest as M

    orig = M._validate
    monkeypatch.setattr(M, "_validate", lambda *a, **k: orig(*a, **k) + ["[排他] 人为注入"])
    out = tmp_path / "p.json"
    monkeypatch.setattr(sys, "argv", [
        "task_deploy", "--data", str(ROOT / "task3"), "--config", "qwen3-30b-a3b",
        "--phase", "prefill", "--tasks", "gsm8k=1,mbpp=1,no_robots=1", "--expert-sets", str(es),
        "--l0", "8", "--front-full", "--save-plan", str(out)])
    assert td.main() == 3
    assert not out.exists()
    assert "清单校验未通过，不保存" in capsys.readouterr().out
