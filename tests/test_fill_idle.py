"""补位（--fill-idle）：规划完把空着的节点全部用上。

真机上的问题：15 台里，默认参数只用 8 台；L₀=11 时 15 台都有角色，但 4 台是
备胎，协调器从不把备胎放进池子。六步流水线为了组合矩阵均匀（盲绑零后悔），
会主动放弃远端站点和小显存的节点。

补位做三件事，按顺序：一前一后成对地加通道（后段给最堵的池）→ 拆开也拼不出
通道的备胎原样提拔 → 剩下的挂到显存最紧的节点旁边分摊（先挂补位段，补位段都
挂不上才动核心段）。

这里验的是这几条承诺，外加两条边界：
* 不开 --fill-idle 时什么都不变（论文描述的算法要能原样复现）；
* 补位段出公共带是**警告**不是违规（否则开了补位清单永远上不了线）。
"""

from __future__ import annotations

import re
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from p2pmoe.planner.manifest import DeploymentManifest
from p2pmoe.planner.network import MeasurementCache, Probe
from p2pmoe.planner.pipeline import plan
from p2pmoe.planner.solver import deploy_path
from p2pmoe.planner.types import Node, Objective, SegmentSpec
from p2pmoe.sim.scenario import appendix_c


def _run(seed: int, fill: bool):
    st = appendix_c(seed=seed, with_experts=True)
    cfg = replace(st.cfg, fill_idle=fill)
    res = plan(st.nodes, st.model, st.tasks, st.union_experts, st.net, cfg, st.p_curve)
    return st, res


@pytest.fixture(scope="module", params=[7, 1])
def pair(request):
    """(场景, 不补位的结果, 补位的结果)。两个种子：一个核心终审过、一个不过。"""
    st, off = _run(request.param, False)
    _, on = _run(request.param, True)
    return st, off, on


def _concurrency(res) -> int:
    return min(len(res.fronts_final), res.n_back_total)


# --------------------------------------------------------------------------- #
# 不开时什么都不变
# --------------------------------------------------------------------------- #
def test_off_is_the_default() -> None:
    st = appendix_c(seed=7, with_experts=True)
    assert st.cfg.fill_idle is False


def test_off_leaves_the_manifest_untouched(pair) -> None:
    """不开补位时清单里不能出现任何补位痕迹 —— 论文的算法要能逐字复现。"""
    _, off, _ = pair
    assert off.fill is None and off.audit_core is None
    man = off.manifest
    assert man.warnings == [] and man.idle == []
    for info in man.segments.values():
        assert "origin" not in info and "spread_nodes" not in info


# --------------------------------------------------------------------------- #
# 开了之后的承诺
# --------------------------------------------------------------------------- #
def test_every_node_gets_a_role(pair) -> None:
    """核心诉求：没有空位。"""
    st, off, on = pair
    all_ids = {n.id for n in st.nodes}
    used_off = {p.node for p in off.manifest.nodes}
    used_on = {p.node for p in on.manifest.nodes}
    assert used_off < all_ids, "前提：不补位时确实有空位，否则这个用例测不到东西"
    assert used_on == all_ids
    assert on.manifest.idle == []
    assert on.manifest.standby_fronts == [], "备胎也是空位 —— 协调器从不把它放进池子"


def test_concurrency_grows(pair) -> None:
    """加的是**通道**，不只是让节点有事干。并发上限是 min(前段, 后段)。"""
    _, off, on = pair
    assert _concurrency(on) > _concurrency(off)


def test_channels_stay_balanced(pair) -> None:
    """成对地加：只加一边吞吐不涨，白占机器。"""
    _, _, on = pair
    assert abs(len(on.fronts_final) - on.n_back_total) <= 1


def test_the_new_back_goes_to_the_most_starved_pool(pair) -> None:
    """第一条补位后段给公平比最低的池 —— 真机上就是排了 15–30 秒的 gsm8k。

    公平比按**实建**条数算，不是主流程的配额：Step 3/4 可能少建，配额会过时。
    """
    from p2pmoe.planner.capacity import fair_ratios

    st, off, on = pair
    lam = {t.name: t.lam for t in st.tasks}
    fair = fair_ratios({u: len(off.backs.get(u, [])) for u in lam}, lam)
    want = min(lam, key=lambda u: (fair[u], -lam[u]))
    first = next(l for l in on.fill.log if "新建后段" in l)
    got = re.search(r"新建后段（(\S+) 公平比", first).group(1)
    assert got == want, f"补位把第一条后段给了 {got}，而最堵的池是 {want}：{first}"


def test_core_segments_are_untouched_unless_unavoidable(pair) -> None:
    """核心段有终审担保，分摊先挂补位段。动了核心段必须在日志里点名。"""
    _, off, on = pair
    core_before = {tuple(s.nodes) for s in off.fronts_final} | {
        tuple(b.nodes) for v in off.backs.values() for b in v}
    touched = [l for l in on.fill.log if "动了核心段" in l]
    for sid, info in on.manifest.segments.items():
        if "spread_nodes" in info and info.get("origin") is None:
            assert touched, f"核心段 {sid} 被分摊改动了，日志却没有点名"
    if not touched:
        core_after = {tuple(s.nodes) for s in on.fronts_final} | {
            tuple(b.nodes) for v in on.backs.values() for b in v}
        assert core_before <= core_after, "没有点名动核心段，核心段却变了"


def test_band_breaches_of_fill_pairs_are_warnings_not_violations(pair) -> None:
    """补位段本来就建在带外。记成违规会让清单永远上不了线，等于没开。"""
    _, off, on = pair
    assert on.manifest.ok == off.manifest.ok, "补位不该让清单从能上线变成不能上线"
    relaxed_nodes = {s.nodes for s in on.fill.relaxed}
    relaxed_sids = {sid for sid, info in on.manifest.segments.items()
                    if tuple(info["nodes"]) in relaxed_nodes}
    for w in on.manifest.warnings:
        m = re.search(r"(F[\w-]*?\d+)×(B\S+?\d+)\b", w)
        assert m, w
        assert m.group(1) in relaxed_sids or m.group(2) in relaxed_sids, (
            f"警告里的组合 {m.group(0)} 两头都是核心段 —— 核心段出带应是违规，不该降级")


def test_the_core_audit_is_kept_for_comparison(pair) -> None:
    """开补位时 audit 是全网格的；核心那份留着对照，否则看不出补位付了多少代价。"""
    _, off, on = pair
    assert on.audit_core is not None
    assert on.audit_core.worst_rel_spread == pytest.approx(off.audit.worst_rel_spread)


def test_annotations_survive_a_json_roundtrip(pair) -> None:
    """--save-plan 存下来再 --load-plan 读回，补位标注不能丢。"""
    _, _, on = pair
    man = on.manifest
    back = DeploymentManifest.from_json(man.to_json())
    assert back.warnings == man.warnings
    assert back.idle == man.idle
    assert {k: v.get("origin") for k, v in back.segments.items()} == \
           {k: v.get("origin") for k, v in man.segments.items()}


# --------------------------------------------------------------------------- #
# 求解器：0 跳 / 1 跳穷举
# --------------------------------------------------------------------------- #
class _Flat:
    """所有链路同一个时延、零抖动。"""

    def probe(self, a: str, b: str, k: int) -> Probe:
        return Probe(p50=3.0, p95=3.0, k=k)


def _tiny():
    """一台大的（9.5GB）+ 三台小的（1.2GB），10 层每层 1GB。

    A 装 9 层、B 装 1 层就是一个 1 跳解。但 beam 宽度为 1 时，前缀按 A* 分数
    打平，留下的是「A 只装 1 层」，后面三台小机器每台只能再接 1 层，跑完
    节点也放不下 —— 返回 None。真机上同一类截断让 gsm8k 后段在
    {N12,N13,N14,N15} 上返回 None，而 N13+N14 两台就装得下。
    """
    nodes = {
        "A": Node(id="A", tier="big", mem_gb=10.5, ms_per_layer=1.0),
        "B": Node(id="B", tier="small", mem_gb=2.2, ms_per_layer=1.0),
        "C": Node(id="C", tier="small", mem_gb=2.2, ms_per_layer=1.0),
        "D": Node(id="D", tier="small", mem_gb=2.2, ms_per_layer=1.0),
    }
    spec = SegmentSpec(kind="back", task="x", layer_lo=1, layer_hi=10,
                       gb_per_layer=lambda l: 1.0, kv_gb_per_layer=0.0)
    net = MeasurementCache(_Flat(), k=8, j_cap_ms=25.0)
    return nodes, spec, net


def test_beam_truncation_can_miss_a_one_hop_solution() -> None:
    """前提：原求解器在这个例子上确实会漏 —— 否则下一条测不到东西。"""
    nodes, spec, net = _tiny()
    got = deploy_path(spec, list(nodes), nodes, net, Objective(), beam_width=1, prune_topk=4)
    assert got is None or got.hops > 1


def test_exhaustive_short_finds_it() -> None:
    nodes, spec, net = _tiny()
    got = deploy_path(spec, list(nodes), nodes, net, Objective(),
                      beam_width=1, prune_topk=4, exhaustive_short=True)
    assert got is not None and got.hops == 1
    assert got.nodes[0] == "A"


def test_exhaustive_short_is_off_by_default() -> None:
    """默认关：开了它主流程的结果会变，那要单独决定。"""
    import inspect
    sig = inspect.signature(deploy_path)
    assert sig.parameters["exhaustive_short"].default is False


# --------------------------------------------------------------------------- #
# 成对：从「前段 = 后段」出发，每一步都必须一前一后一起加
# --------------------------------------------------------------------------- #
def test_from_balanced_every_step_adds_a_full_channel() -> None:
    """只加后段吞吐不涨（并发 = min(前段, 后段)），只是白占机器。

    直接调 fill_idle，把主流程多出来的前段拿掉，让起点严格平衡：之后每一步
    要么一对一起提交，要么什么都不加。
    """
    from p2pmoe.planner.fill import fill_idle

    st, off = _run(7, False)
    nb = off.n_back_total
    fronts = off.fronts_final[:nb]
    res = fill_idle(fronts=fronts, standby=[], backs=off.backs, tasks=st.tasks,
                    model=st.model, union_experts=st.union_experts, l0=off.l0,
                    node_map={n.id: n for n in st.nodes}, net=st.net, cfg=st.cfg)
    added_f = sum(1 for s in res.fronts if res.origin.get(s) == "fill")
    added_b = sum(1 for v in res.backs.values() for s in v if res.origin.get(s) == "fill")
    assert added_f >= 1, "一条通道都没加上 —— 前提不成立"
    assert added_f == added_b, f"从平衡出发却加了 {added_f} 前 / {added_b} 后"
    assert len(res.fronts) == sum(len(v) for v in res.backs.values())


# --------------------------------------------------------------------------- #
# 真拓扑（task/：15 台、真激活数据）—— 两个在合成场景里测不到的真问题
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent


def _real(*extra: str):
    """跑 examples/task_deploy.py，截下 PlanResult。"""
    import contextlib
    import io

    if not (ROOT / "task" / "topo_fabric_mbps.json").exists():
        pytest.skip("缺 task/ 真实数据")
    sys.path.insert(0, str(ROOT))
    import examples.task_deploy as TD

    got: dict = {}
    orig = TD.plan

    def spy(*a, **k):
        got["res"] = orig(*a, **k)
        return got["res"]

    argv = sys.argv
    TD.plan = spy
    sys.argv = ["task_deploy", "--data", str(ROOT / "task"), "--tasks", "mbpp=5,gsm8k=3",
                "--coverage", "0.70", *extra]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            TD.main()
    finally:
        TD.plan = orig
        sys.argv = argv
    return got["res"]


@pytest.fixture(scope="module")
def deployed():
    """线上正在用的那份配置：L₀=11。主流程 3 条通道，N12–N15 是备胎。"""
    return _real("--l0", "11"), _real("--l0", "11", "--fill-idle")


def test_deployed_plan_gains_a_channel_not_just_a_busy_standby(deployed) -> None:
    """备胎原样提拔只多一条前段，并发还是 3；拆开能拼出一整条通道。

    能拼出来要靠补位里的 0/1 跳穷举：beam=12 时 gsm8k 后段在 {N12..N15} 上
    返回 None，而 N13+N14 两台就装得下。
    """
    off, on = deployed
    assert _concurrency(off) == 3 and off.standby, "前提：主流程 3 条 + 一条备胎"
    assert _concurrency(on) == 4
    assert {p.node for p in on.manifest.nodes} == {f"N{i:02d}" for i in range(1, 16)}
    gsm = [s for s in on.backs["gsm8k"] if on.fill.origin.get(s) == "fill"]
    assert gsm, "新通道该给 gsm8k —— 真机上排了 15–30 秒的就是它"


def test_core_audit_of_deployed_plan_is_unchanged(deployed) -> None:
    """补位不能把主流程本身的结果改掉：核心终审仍是那份 6.1%。"""
    off, on = deployed
    assert on.audit_core.worst_rel_spread == pytest.approx(off.audit.worst_rel_spread)
    assert off.audit.passed


def test_spreading_spares_core_segments_on_the_real_pool() -> None:
    """默认 L₀ 下最后两台要分摊。不分先后时 N15 会挂进核心段 mbpp[N06+N08]，
    那条有终审担保的通道凭空慢 30ms；应该挂到补位段上。"""
    res = _real("--fill-idle")
    assert not [l for l in res.fill.log if "动了核心段" in l], "\n".join(res.fill.log)
    assert ("N06", "N08") in {tuple(s.nodes) for s in res.backs["mbpp"]}
