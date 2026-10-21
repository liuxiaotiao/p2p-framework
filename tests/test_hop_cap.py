"""求解器的链长上限（solver.deploy_path 的 hop_slack / PlannerConfig.max_hops_slack）。

链长上限按段的大小定：节点数 ≤ ⌈段驻留量 / 最大单机可用⌉ + hop_slack，即跳数
≤ hops_min（命题 III.5.2）+ hop_slack。它是搜索时的可行性剪枝，不是建完再否决。

回归场景（15 台真拓扑，L₀=8、前段装全部专家、预留 2.5GB、按文档取租金 = 内存
最大的 N 台、无端点项）：mbpp 后段 31.8GB、两台 24GB 卡就够，beam 却交出
N09+…+N04 共 9 台 8 跳的链，把剩余节点吃光，前段无处可放，整份规划失败。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from p2pmoe.planner.memory import hops_min
from p2pmoe.planner.solver import deploy_path
from p2pmoe.planner.types import Objective

from test_beam_complete_solutions import _brute, _instance, _seg_score

ROOT = Path(__file__).resolve().parents[1]


def _floor(spec, nm) -> int:
    return hops_min(spec.total_gb(), max(v.usable_gb for v in nm.values()))


@pytest.mark.parametrize("slack", [0, 1, 2])
@pytest.mark.parametrize("seed", range(8))
def test_result_respects_the_cap(seed, slack):
    """任何 beam 宽度下，交出的段都不超过 hops_min + slack 跳。"""
    nm, net, spec = _instance(seed, 9, 8, small=True)
    for bw in (1, 4, 12):
        seg = deploy_path(spec, sorted(nm), nm, net, Objective(), beam_width=bw, hop_slack=slack)
        if seg is not None:
            assert seg.hops <= _floor(spec, nm) + slack, (bw, seg.label())


@pytest.mark.parametrize("slack", [0, 1])
@pytest.mark.parametrize("seed", range(6))
def test_cap_only_removes_over_long_chains(seed, slack):
    """剪枝不误杀：不截断时，结果等于「≤ 上限台数的链」里的暴力最优。"""
    nm, net, spec = _instance(seed, 6, 16, small=True)
    obj = Objective()
    cap_nodes = _floor(spec, nm) + 1 + slack
    if cap_nodes > 3:
        pytest.skip("上限超过暴力枚举的 3 台")
    want, _ = _brute(spec, nm, net, obj, max_nodes=cap_nodes)
    seg = deploy_path(spec, sorted(nm), nm, net, obj, beam_width=10**6, prune_topk=99,
                      hop_slack=slack)
    if want == float("inf"):
        assert seg is None
        return
    assert seg is not None and len(seg.nodes) <= cap_nodes
    assert _seg_score(seg, spec, nm, net, obj) == pytest.approx(want)


@pytest.mark.parametrize("seed", range(4))
def test_legacy_beam_ignores_the_cap(seed):
    """--beam-legacy 只用来逐字节复现旧方案，链长上限对它不生效。"""
    nm, net, spec = _instance(seed, 9, 8, small=True)
    a = deploy_path(spec, sorted(nm), nm, net, Objective(), legacy_beam=True)
    b = deploy_path(spec, sorted(nm), nm, net, Objective(), legacy_beam=True, hop_slack=0)
    assert a == b


def test_no_cap_keeps_the_old_behaviour():
    """hop_slack=None 与不传一致（默认不限，调用方显式开启）。"""
    nm, net, spec = _instance(1, 9, 8, small=True)
    a = deploy_path(spec, sorted(nm), nm, net, Objective())
    b = deploy_path(spec, sorted(nm), nm, net, Objective(), hop_slack=None)
    assert a == b


def _strict_doc_run(monkeypatch, slack):
    """按文档的租金（取内存最大者）与目标函数（无端点项）跑 L₀=8 的真实规划，
    记录每次建后段的结果。slack=None 即不限链长（--hop-slack -1）。"""
    es = ROOT / "qwen3-30b" / "expert_sets_90" / "expert_sets_90_union.json"
    if not es.exists() or not (ROOT / "task3" / "gsm8k_expert_activation.csv").exists():
        pytest.skip("没有 qwen3-30b/ 或 task3/ 的数据")
    sys.path.insert(0, str(ROOT))
    import examples.task_deploy as td
    import p2pmoe.planner.pipeline as P

    built = []
    orig = P.deploy_path

    def spy(spec, avail, *a, **k):
        s = orig(spec, avail, *a, **k)
        if s is not None and spec.kind != "front":
            built.append((spec.task, spec.total_gb(), len(avail), s))
        return s

    monkeypatch.setattr(P, "deploy_path", spy)
    monkeypatch.setattr(sys, "argv", [
        "task_deploy", "--data", str(ROOT / "task3"), "--config", "qwen3-30b-a3b",
        "--phase", "prefill", "--tasks", "gsm8k=1,mbpp=1,no_robots=1", "--expert-sets", str(es),
        "--l0", "8", "--front-full", "--reserve-gb", "2.5",
        "--rent-policy", "largest", "--endpoint-w", "0",
        "--hop-slack", str(-1 if slack is None else slack)])
    td.main()
    return built


def test_real_topology_without_cap_builds_the_nine_node_chain(monkeypatch, capsys):
    """不限链长时复现那条 9 台 8 跳的 mbpp 后段 —— 证明下一条测试防住的是真问题。"""
    built = _strict_doc_run(monkeypatch, None)
    mbpp = [s for t, _, _, s in built if t == "mbpp"]
    assert mbpp and max(s.hops for s in mbpp) >= 7


def test_real_topology_with_cap_has_no_long_chain(monkeypatch, capsys):
    built = _strict_doc_run(monkeypatch, 1)
    assert built
    for task, total, _, s in built:
        floor = hops_min(total, 21.5)          # 24GB 卡 − 2.5GB 预留
        assert s.hops <= floor + 1, (task, s.label())
    mbpp = [s for t, _, _, s in built if t == "mbpp"]
    assert mbpp and all(len(s.nodes) <= 3 for s in mbpp)


def test_strict_doc_plan_file_is_reproducible(tmp_path, monkeypatch, capsys):
    """qwen3-30b/plan_doc_l8.json（严格文档 + L₀=8 + 前段全专家 + 预留 2.5GB +
    --hop-slack 0）能用 README/REPRODUCE 里那条命令逐字节重出。"""
    want = ROOT / "qwen3-30b" / "plan_doc_l8.json"
    es = ROOT / "qwen3-30b" / "expert_sets_90" / "expert_sets_90_union.json"
    if not (want.exists() and es.exists() and (ROOT / "task3").exists()):
        pytest.skip("缺 qwen3-30b/ 或 task3/ 数据")
    sys.path.insert(0, str(ROOT))
    import examples.task_deploy as td

    out = tmp_path / "p.json"
    monkeypatch.setattr(sys, "argv", [
        "task_deploy", "--data", str(ROOT / "task3"), "--config", "qwen3-30b-a3b",
        "--phase", "prefill", "--tasks", "gsm8k=1,mbpp=1,no_robots=1", "--expert-sets", str(es),
        "--l0", "8", "--front-full", "--reserve-gb", "2.5", "--hop-slack", "0",
        "--rent-policy", "largest", "--endpoint-w", "0", "--avail", "0.95", "--theta", "0.8",
        "--save-plan", str(out)])
    assert td.main() == 0
    assert out.read_bytes() == want.read_bytes()


def test_doc_v2_plan_file_is_reproducible(tmp_path, monkeypatch, capsys):
    """qwen3-30b/plan_doc_v2.json（文档六步 + 实测每层耗时/显存 + r_v=1、θ=1、不留租金、
    无端点项、不穷举；每台固定留 2GB 给 KV cache，其余全给权重）能用 REPRODUCE 5.7 那条命令逐字节重出。"""
    want = ROOT / "qwen3-30b" / "plan_doc_v2.json"
    lb = ROOT / "qwen3-30b" / "layer_bench_wash1.json"
    es = ROOT / "qwen3-30b" / "expert_sets_90" / "expert_sets_90_union.json"
    if not (want.exists() and lb.exists() and es.exists() and (ROOT / "task3").exists()):
        pytest.skip("缺 qwen3-30b/ 或 task3/ 数据")
    sys.path.insert(0, str(ROOT))
    import examples.task_deploy as td

    out = tmp_path / "p.json"
    monkeypatch.setattr(sys, "argv", [
        "task_deploy", "--data", str(ROOT / "task3"), "--config", "qwen3-30b-a3b",
        "--phase", "prefill", "--tasks", "gsm8k=1,mbpp=1,no_robots=1", "--expert-sets", str(es),
        "--l0", "8", "--front-full", "--layer-times", str(lb), "--measured-mem",
        "--mem-margin-gb", "0", "--kv-reserve-gb", "2", "--mem-prefill-tokens", "512", "--endpoint-w", "0",
        "--save-plan", str(out)])
    assert td.main() == 0
    assert out.read_bytes() == want.read_bytes()
