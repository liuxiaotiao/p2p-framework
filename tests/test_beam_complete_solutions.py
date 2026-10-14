"""beam 的「完整解不参与截断」+「用最好的完整解剪半成品」。

旧缺陷：半成品的分数 = 已花 + 剩余部分的乐观下界，完整解的分数 = 真实代价；
两者同榜截断，半成品永远排在由它自己补完的完整解前面 → 完整解被挤掉，beam 交出
长链甚至 None。真拓扑上 no_robots 后段 1 跳解排第 217 名被截（test_expert_sets
里有那条回归）。这里查两件事：
  1. 完整解不会被截：beam_width=1 也不比「单台放下整段」差；
  2. 剪枝不误杀：不截断时，结果与暴力枚举的最优一致（下界确实是下界）。
"""

from __future__ import annotations

import itertools
import random

import pytest

from p2pmoe.planner.memory import make_back_spec
from p2pmoe.planner.network import MeasurementCache
from p2pmoe.planner.solver import deploy_path
from p2pmoe.planner.types import Objective, TaskProfile
from p2pmoe.sim.network import SimNetwork
from p2pmoe.sim.scenario import APPENDIX_C_MODEL, build_nodes


def _instance(seed: int, n: int, l0: int, small: bool = False):
    """APPENDIX_C_MODEL 共 32 层；l0 越小后段越大。small=True 去掉 48GB 的卡，逼出多跳。"""
    rng = random.Random(seed)
    pool = [v for v in build_nodes() if not (small and v.mem_gb >= 48)]
    nodes = rng.sample(pool, n)
    nm = {v.id: v for v in nodes}
    net = MeasurementCache(SimNetwork([v.id for v in nodes], seed=seed), k=8, j_cap_ms=40.0)
    spec = make_back_spec(APPENDIX_C_MODEL, TaskProfile("Y", 0.3, 6), l0)
    return nm, net, spec


def _score(spec, chain, splits, nm, net, obj):
    comp = sum((hi - lo + 1) * nm[v].ms_per_layer * spec.ms_per_layer_scale
               for v, (lo, hi) in zip(chain, splits))
    hop = sum(net.p50(a, b) for a, b in zip(chain, chain[1:]))
    jit = sum(net.jitter(a, b) for a, b in zip(chain, chain[1:]))
    return obj.score(comp, hop, jit, chain, nm)


def _brute(spec, nm, net, obj, max_nodes: int):
    """≤ max_nodes 台的全部链 × 全部切分。"""
    lo0, hi0 = spec.layer_lo, spec.layer_hi
    L = hi0 - lo0 + 1
    best = (float("inf"), None)
    ids = sorted(nm)
    for k in range(1, max_nodes + 1):
        for chain in itertools.permutations(ids, k):
            if not spec.head_ok(chain[0]) or not spec.tail_ok(chain[-1]):
                continue
            if any(net.blocked(a, b) for a, b in zip(chain, chain[1:])):
                continue
            for cuts in itertools.combinations(range(1, L), k - 1):
                b = [0, *cuts, L]
                splits = [(lo0 + b[i], lo0 + b[i + 1] - 1) for i in range(k)]
                if all(spec.resident_gb(lo, hi) <= nm[v].usable_gb + 1e-9
                       for v, (lo, hi) in zip(chain, splits)):
                    s = _score(spec, chain, splits, nm, net, obj)
                    if s < best[0]:
                        best = (s, (chain, splits))
    return best


def _seg_score(seg, spec, nm, net, obj):
    return _score(spec, seg.nodes, seg.splits, nm, net, obj)


@pytest.mark.parametrize("seed", range(6))
def test_complete_solutions_survive_a_width_one_beam(seed):
    """beam_width=1：旧代码里唯一的名额永远给乐观的半成品，单台放得下整段的
    0 跳解被截掉。现在完整解先收下，结果至少不比它差。"""
    nm, net, spec = _instance(seed, 8, 16)
    obj = Objective()
    whole = [v for v in nm if spec.resident_gb(spec.layer_lo, spec.layer_hi)
             <= nm[v].usable_gb + 1e-9 and spec.head_ok(v) and spec.tail_ok(v)]
    if not whole:
        pytest.skip("这个实例没有能单台装下整段的节点")
    single = min(_score(spec, (v,), ((spec.layer_lo, spec.layer_hi),), nm, net, obj)
                 for v in whole)
    seg = deploy_path(spec, sorted(nm), nm, net, obj, beam_width=1)
    assert seg is not None
    assert _seg_score(seg, spec, nm, net, obj) <= single + 1e-9


@pytest.mark.parametrize("small", [False, True])
@pytest.mark.parametrize("l0", [4, 8])
@pytest.mark.parametrize("seed", range(6))
def test_pruning_never_cuts_the_optimum(seed, l0, small):
    """不截断（beam 极宽、邻居不剪）时，分支定界的结果 = 暴力最优。
    若「乐观分是下界」不成立，剪枝会误杀最优解，这条就会失败。"""
    nm, net, spec = _instance(seed, 6, l0, small)
    obj = Objective(mu_ms=40.0, jitter_w=0.2)
    seg = deploy_path(spec, sorted(nm), nm, net, obj, beam_width=10**6, prune_topk=99)
    if seg is None or len(seg.nodes) > 3:
        pytest.skip("最优解超过 3 台，暴力枚举不覆盖")
    want, _ = _brute(spec, nm, net, obj, max_nodes=3)
    assert _seg_score(seg, spec, nm, net, obj) == pytest.approx(want, abs=1e-9)


@pytest.mark.parametrize("seed", range(6))
def test_target_mode_bound_is_admissible_too(seed):
    """tighten 用的「靠近目标」模式：partial 取 max(0, delay − target)，同样是下界。"""
    nm, net, spec = _instance(seed, 6, 4, small=True)
    free = deploy_path(spec, sorted(nm), nm, net, Objective(), beam_width=10**6, prune_topk=99)
    if free is None or len(free.nodes) > 3:
        pytest.skip("最优解超过 3 台")
    obj = Objective(target_ms=free.delay_ms * 1.3)
    seg = deploy_path(spec, sorted(nm), nm, net, obj, beam_width=10**6, prune_topk=99)
    want, _ = _brute(spec, nm, net, obj, max_nodes=3)
    if seg is None or len(seg.nodes) > 3:
        pytest.skip("目标模式下最优超过 3 台")
    assert _seg_score(seg, spec, nm, net, obj) == pytest.approx(want, abs=1e-9)


THREE_NODE = [(2, 4), (6, 2), (27, 2), (33, 2), (35, 0)]
"""最优解恰好要 3 台的实例（48GB 卡去掉、6 台、后段 28–32 层）。只有这种实例里
剪枝才真正在半成品上动刀：最优解在第 2 轮之后才补完，之前的半成品若被误剪就找不回。"""


def _three_node_instance(seed: int, l0: int):
    pool = [v for v in build_nodes() if v.mem_gb < 48]
    rng = random.Random(seed)
    for cur in (0, 2, 4):                       # 与筛选时同一个随机序列
        nodes = rng.sample(pool, 6)
        if cur == l0:
            break
    nm = {v.id: v for v in nodes}
    net = MeasurementCache(SimNetwork([v.id for v in nodes], seed=seed), k=8, j_cap_ms=40.0)
    return nm, net, make_back_spec(APPENDIX_C_MODEL, TaskProfile("Y", 0.3, 6), l0)


@pytest.mark.parametrize("seed,l0", THREE_NODE)
def test_pruning_keeps_three_node_optima(seed, l0):
    nm, net, spec = _three_node_instance(seed, l0)
    obj = Objective(mu_ms=40.0, jitter_w=0.2)
    seg = deploy_path(spec, sorted(nm), nm, net, obj, beam_width=10**6, prune_topk=99)
    assert seg is not None and len(seg.nodes) == 3
    want, _ = _brute(spec, nm, net, obj, max_nodes=3)
    assert _seg_score(seg, spec, nm, net, obj) == pytest.approx(want, abs=1e-9)


@pytest.mark.parametrize("small", [False, True])
@pytest.mark.parametrize("l0", [0, 4, 8])
@pytest.mark.parametrize("seed", range(3))
def test_partial_score_is_a_lower_bound(seed, l0, small):
    """剪枝与 beam 排序依赖的核心性质，直接验证：

        对任意可行的完整链与它的任意真前缀，score_partial(前缀) ≤ score_final(完整链)。

    暴力枚举 ≤3 台的全部链 × 全部切分。这条不成立，「用最好的完整解剪半成品」就会
    误杀最优解。"""
    from p2pmoe.planner.solver import _Partial, _scorers

    nm, net, spec = _instance(seed, 6, l0, small)
    obj = Objective(mu_ms=40.0, jitter_w=0.2)
    sp, sf, _ = _scorers(spec, sorted(nm), nm, net, obj)
    lo0, L = spec.layer_lo, spec.n_layers
    scale = spec.ms_per_layer_scale
    checked = 0
    for k in (2, 3):
        for chain in itertools.permutations(sorted(nm), k):
            if any(net.blocked(a, b) for a, b in zip(chain, chain[1:])):
                continue
            for cuts in itertools.combinations(range(1, L), k - 1):
                b = [0, *cuts, L]
                splits = [(lo0 + b[i], lo0 + b[i + 1] - 1) for i in range(k)]
                if not all(spec.resident_gb(lo, hi) <= nm[v].usable_gb + 1e-9
                           for v, (lo, hi) in zip(chain, splits)):
                    continue

                def part(j):
                    c = sum((hi - lo + 1) * nm[v].ms_per_layer * scale
                            for v, (lo, hi) in zip(chain[:j], splits[:j]))
                    h = sum(net.p50(x, y) for x, y in zip(chain[:j], chain[1:j]))
                    jt = sum(net.jitter(x, y) for x, y in zip(chain[:j], chain[1:j]))
                    return _Partial(tuple(chain[:j]), tuple(splits[:j]), b[j], c, h, jt)

                full = sf(part(k))
                for j in range(1, k):
                    assert sp(part(j)) <= full + 1e-9, (chain, splits, j)
                    checked += 1
    assert checked > 0


def test_beam_legacy_reproduces_the_deployed_plan(tmp_path):
    """--beam-legacy 与修复前的代码逐字节一致：拿它重出现在集群上用的方案，
    必须与 qwen3-30b/plan_es90_fill.json 一模一样（旧实验因此可以原样复现）。"""
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    want = root / "qwen3-30b" / "plan_es90_fill.json"
    es = root / "qwen3-30b" / "expert_sets_90" / "expert_sets_90_union.json"
    if not (want.exists() and es.exists() and (root / "task3").exists()):
        pytest.skip("缺 qwen3-30b/ 或 task3/ 数据")
    sys.path.insert(0, str(root))
    import examples.task_deploy as td

    out = tmp_path / "p.json"
    argv = sys.argv
    sys.argv = ["task_deploy", "--data", str(root / "task3"), "--config", "qwen3-30b-a3b",
                "--phase", "prefill", "--tasks", "gsm8k=1,mbpp=1,no_robots=1",
                "--expert-sets", str(es), "--l0", "8", "--front-full", "--reserve-gb", "2.5",
                "--exhaustive-short", "--fill-idle", "--beam-legacy", "--save-plan", str(out)]
    try:
        assert td.main() == 0
    finally:
        sys.argv = argv
    assert out.read_bytes() == want.read_bytes()
