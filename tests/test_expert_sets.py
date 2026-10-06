"""外部给定的逐层驻留集（qwen3-30b/expert_sets_90_union.json）进规划器。

那份文件是「每层 prefill / decode 各取覆盖 90% 激活量的最少专家，再取并集」，
不是激活分布 —— 规划器原来只会从分布按 --coverage 现取，接不上它。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest

from p2pmoe.planner.experts import ActivationProfile
from p2pmoe.planner.from_data import load_expert_sets
from p2pmoe.planner.types import PlannerConfig


def _write(tmp_path, obj) -> Path:
    p = tmp_path / "sets.json"
    p.write_text(json.dumps(obj))
    return p


SETS = {"threshold": 0.9, "rule": "r", "tasks": {
    "a": {"source": "X", "union": [[0, 1, 2], [1, 2, 3]], "prefill": [[0, 1], [1, 2]],
          "decode": [[2], [3]]},
    "b": {"union": [[4, 5], [5, 6]], "prefill": [[4, 5], [5, 6]], "decode": [[4], [6]]}}}


def test_reads_the_expert_sets_format(tmp_path) -> None:
    plc, info = load_expert_sets(_write(tmp_path, SETS), n_layers=2, n_experts=8)
    assert plc["a"].at(1) == {0, 1, 2} and plc["a"].at(2) == {1, 2, 3}
    assert plc["a"].achieved_coverage == (0.9, 0.9)          # 没画像时用名义值
    assert info["per_task"]["a"]["source"] == "X"
    pre, _ = load_expert_sets(_write(tmp_path, SETS), key="prefill")
    assert pre["a"].at(1) == {0, 1}


def test_reads_plain_id_format(tmp_path) -> None:
    plc, _ = load_expert_sets(_write(tmp_path, {"a": [[0, 1], [2, 3]]}))
    assert plc["a"].at(2) == {2, 3}
    plc, _ = load_expert_sets(_write(tmp_path, {"format": "ids", "tasks": {"a": [[0], [1]]}}))
    assert plc["a"].at(1) == {0}


def test_actual_coverage_comes_from_the_profile(tmp_path) -> None:
    prof = ActivationProfile(task="a", n_layers=2, n_experts=8, mass=(
        (0.5, 0.3, 0.1, 0.1, 0, 0, 0, 0), (0, 0.2, 0.2, 0.2, 0.4, 0, 0, 0)))
    plc, info = load_expert_sets(_write(tmp_path, SETS), tasks=["a"], profiles={"a": prof})
    assert plc["a"].achieved_coverage == pytest.approx((0.9, 0.6))
    assert info["per_task"]["a"]["coverage_src"] == "画像实测"


@pytest.mark.parametrize("kw,msg", [
    (dict(n_layers=3), "不是同一个模型"),
    (dict(n_experts=4), "模型只有 4 个"),
    (dict(min_experts=3), "少于 top-k"),
    (dict(tasks=["zz"]), "没有 task"),
])
def test_mismatches_are_rejected(tmp_path, kw, msg) -> None:
    with pytest.raises(ValueError, match=msg):
        load_expert_sets(_write(tmp_path, SETS), **kw)


def test_exhaustive_short_is_off_by_default() -> None:
    """打开它会改变 Qwen3-Next 在 L₀=11 下的方案，所以默认必须是关的。"""
    assert PlannerConfig().exhaustive_short is False
    for f in ("p2pmoe/planner/pipeline.py", "p2pmoe/planner/tighten.py"):
        src = (ROOT / f).read_text(encoding="utf-8")
        assert src.count("exhaustive_short=cfg.exhaustive_short") >= 2, f


@pytest.mark.slow
def test_the_real_qwen3_30b_plan(tmp_path, capsys) -> None:
    """用 qwen3-30b/ 下的真实驻留集出一份方案：L₀=8、前段装全、15 台全上岗。
    不开 --exhaustive-short 时 beam 截断会让这份规划失败（N09 一台就装得下前段）。"""
    es = ROOT / "qwen3-30b" / "expert_sets_90_union.json"
    if not es.exists() or not (ROOT / "task3" / "gsm8k_expert_activation.csv").exists():
        pytest.skip("没有 qwen3-30b/ 或 task3/ 的数据")
    import examples.task_deploy as td

    base = ["--data", str(ROOT / "task3"), "--config", "qwen3-30b-a3b", "--phase", "prefill",
            "--tasks", "gsm8k=1,mbpp=1,no_robots=1", "--expert-sets", str(es),
            "--l0", "8", "--front-full", "--reserve-gb", "2.5"]
    old = sys.argv
    try:
        sys.argv = ["task_deploy"] + base
        assert td.main() == 2, "不开 --exhaustive-short 应当规划失败（beam 截断）"
        plan, prof = tmp_path / "p.json", tmp_path / "prof.json"
        sys.argv = ["task_deploy"] + base + ["--exhaustive-short", "--fill-idle",
                                             "--save-plan", str(plan), "--save-profile", str(prof)]
        assert td.main() == 0
    finally:
        sys.argv = old
    from p2pmoe.planner.manifest import DeploymentManifest

    man = DeploymentManifest.from_json(plan.read_text())
    assert len({p.node for p in man.nodes}) == 15
    fronts = [p for p in man.nodes if p.role.startswith("front")]
    assert all(len(l.experts) == 128 for p in fronts for l in p.layers), "前段没装全"
    assert all(p.layer_range == (1, 8) for p in fronts)
    tasks = {p.role.split(":")[1] for p in man.nodes if p.role.startswith("back:")}
    assert tasks == {"gsm8k", "mbpp", "no_robots"}
    assert max(p.total_gb for p in man.nodes) < 21.5, "单台占用应留出 2.5GB 余量"
    ids = json.loads(prof.read_text())
    assert ids["format"] == "ids" and len(ids["tasks"]["gsm8k"]) == 48
