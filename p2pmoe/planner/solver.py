"""II.1 核心求解器 deploy_path —— 跳数优先的 beam search。

前后段共用，差异全部收进 SegmentSpec。与集群版的三点差异（II.1）：

1. 单节点/少跳解显式优先 —— 每省一跳即省数十毫秒，远超任何计算优化。
   实现上体现为：目标函数里 hop_ms 与 compute_ms 同量纲相加，而分散环境下
   前者是后者的十倍以上；且「把剩余层全部放在当前节点」总是被显式枚举。
2. 抖动进硬屏蔽 —— p95 − p50 > J_cap 的链路不参与（MeasurementCache.blocked）。
3. 稳定性 μ 项升为一等 —— churn 频繁，可用率低的节点即使快也不值。

与文档伪码的一处结构差异（等价改写）：文档写 `p.finish_here()` 与
`p.extend(v′, seg)` 两类动作，本实现把「本节点承载多少层」在节点入链时就
决定，于是 finish_here 只是 seg_len == 剩余层数 的那个候选。两者枚举的解
空间相同，但状态定义更干净（层区间与节点一一对应）。

复杂度：单条段的最优解本可由 DP 多项式求得（命题 III.1.2，O(L²|V|²)），
但那需要「已占节点固定」且不含稳定性/租金这类非可加项。beam 是对含全部
评分项的版本的启发式，宽度 W 可调。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil
from typing import Mapping, Sequence

from .network import MeasurementCache
from .types import Node, Objective, Segment, SegmentSpec

__all__ = ["deploy_path", "prune_neighbors", "segment_from_nodes"]


@dataclass(frozen=True)
class _Partial:
    nodes: tuple[str, ...]
    splits: tuple[tuple[int, int], ...]
    placed: int
    compute_ms: float
    hop_ms: float
    jitter_ms: float

    @property
    def last(self) -> str:
        return self.nodes[-1]


def prune_neighbors(
    v: str,
    avail: Sequence[str],
    nodes: Mapping[str, Node],
    net: MeasurementCache,
    top_k: int,
    peers_for_offset: Sequence[str],
) -> list[str]:
    """预剪枝 N*(v)：按 (M_v, 接入偏移 δ̂_v, 可用率 r_v) 取前 K（II.1）。

    先剔除被抖动闸屏蔽的对端，再按「大内存优先、接入好优先、稳定优先」排序。
    大内存排在最前是因为分散环境下多装一层就可能少一跳。
    """
    cands = [u for u in avail if u != v and not net.blocked(v, u)]
    cands.sort(
        key=lambda u: (
            -nodes[u].usable_gb,
            net.p50(v, u),
            -nodes[u].avail,
        )
    )
    return cands[:top_k]


def _mem_ok(spec: SegmentSpec, lo: int, hi: int, node: Node) -> bool:
    """I.2.2 内存（含 KV）：Σ mem(l) + kv(S, ctx_max) ≤ M_v − 预留。"""
    return spec.resident_gb(lo, hi) <= node.usable_gb + 1e-9


def segment_from_nodes(
    spec: SegmentSpec,
    node_ids: Sequence[str],
    splits: Sequence[tuple[int, int]],
    nodes: Mapping[str, Node],
    net: MeasurementCache,
) -> Segment:
    """由给定的节点链与层切分重建一条 Segment（用于测试与手工方案导入）。"""
    compute = 0.0
    for v, (lo, hi) in zip(node_ids, splits):
        compute += (hi - lo + 1) * nodes[v].ms_per_layer * spec.ms_per_layer_scale
    hop = sum(net.p50(a, b) for a, b in zip(node_ids, node_ids[1:]))
    jit = sum(net.jitter(a, b) for a, b in zip(node_ids, node_ids[1:]))
    return Segment(
        kind=spec.kind,
        task=spec.task,
        nodes=tuple(node_ids),
        splits=tuple(splits),
        compute_ms=compute,
        hop_ms=hop,
        jitter_ms=jit,
    )


def _scorers(spec: SegmentSpec, avail: Sequence[str], nodes: Mapping[str, Node],
             net: MeasurementCache, objective: Objective):
    """→ (score_partial, score_final, completion_lb)。

    score_partial(p) 必须是 p 任何补全方式的 score_final 的**下界**：beam 用它给
    不同进度的半成品排序，分支定界用它剪掉「补完也赢不了现有最好完整解」的半成品。
    抽成模块级函数是为了让测试直接验证这条性质（tests/test_beam_complete_solutions.py）。
    """
    lo0, hi0 = spec.layer_lo, spec.layer_hi
    n_layers = spec.n_layers

    # --- 完成下界 -------------------------------------------------------- #
    # 部分解不能按「已累计代价」直接比较：那样「少放几层」永远分数更低，beam 会
    # 被一堆刚起步的前缀塞满而永远收敛不到完整解。加一个可容许的完成下界，
    # 让不同进度的前缀在同一尺度上比较（A* 式）。
    #
    # 剩余层必须落在**新**节点上（本实现里节点入链时其层数即已确定），故
    #   剩余计算 ≥ 剩余层数 × min ms_per_layer
    #   剩余跳数 ≥ ⌈剩余驻留量 / max(M_v − 预留)⌉      （命题 III.5.2 的同构式）
    # 两项都取可用节点上的极值，保证是下界。
    min_ms = min(nodes[v].ms_per_layer for v in avail) * spec.ms_per_layer_scale
    max_cap = max(nodes[v].usable_gb for v in avail)
    min_hop = net.min_hop_p50(avail)

    def completion_lb(placed: int) -> tuple[float, float]:
        rem = n_layers - placed
        if rem <= 0:
            return (0.0, 0.0)
        rem_gb = spec.resident_gb(lo0 + placed, hi0)
        from math import ceil as _ceil

        n_more = max(1, _ceil(rem_gb / max_cap - 1e-9))
        return (rem * min_ms, n_more * min_hop)

    def score_partial(p: _Partial) -> float:
        lb_c, lb_h = completion_lb(p.placed)
        return objective.score(
            p.compute_ms + lb_c, p.hop_ms + lb_h, p.jitter_ms, p.nodes, nodes, partial=True
        )

    def score_final(p: _Partial) -> float:
        return objective.score(
            p.compute_ms, p.hop_ms, p.jitter_ms, p.nodes, nodes, partial=False
        )


    return score_partial, score_final, completion_lb


def deploy_path(
    spec: SegmentSpec,
    avail: Sequence[str],
    nodes: Mapping[str, Node],
    net: MeasurementCache,
    objective: Objective,
    *,
    beam_width: int = 12,
    prune_topk: int = 10,
    exhaustive_short: bool = False,
    legacy_beam: bool = False,
    hop_slack: int | None = None,
) -> Segment | None:
    """在可用节点集上放置 spec 描述的层区间，返回最优段；不可行返回 None。

    hop_slack 给定时按段的大小限制链长：节点数 ≤ ⌈段驻留量 / 最大单机可用⌉ + hop_slack，
    即跳数 ≤ hops_min（命题 III.5.2）+ hop_slack。None 表示不限（见下方「链长上限」）。
    exhaustive_short=True 时，0 跳与 1 跳的解精确穷举后并进候选（见下方注释）。
    legacy_beam=True 时退回旧的 beam：完整解与半成品同榜截断、不剪枝 —— 只为复现
    修复前的方案（PlannerConfig.beam_legacy / task_deploy --beam-legacy）。
    默认关：开了它主流程的规划结果会变（实测 L₀=11 时多建一条通道，但一张 T4
    被选进前段，核心终审从 6.1% 变成 49.4%）—— 那是要单独决定的事，目前只给
    补位（fill.py）用。
    """
    lo0, hi0 = spec.layer_lo, spec.layer_hi
    n_layers = spec.n_layers
    if n_layers <= 0 or not avail:
        return None

    avail = list(avail)
    peers = avail  # δ̂ 的对端取全体可用节点

    score_partial, score_final, completion_lb = _scorers(spec, avail, nodes, net, objective)

    # --- 0 跳与 1 跳：精确穷举 --------------------------------------------- #
    # beam 只留前 W 个前缀，而完成下界假设「剩下的层能放进最大那台」—— 那台很可能
    # 已经在前缀里用掉了。于是明明存在的 1 跳解会在截断里丢掉，甚至整轮返回 None。
    # 实测（15 台真拓扑，L₀=11）：gsm8k 后段 24.1GB、候选 {N12,N13,N14,N15}，
    # N13+N14 两台就装得下；beam=12 返回 None，beam=50 给出一个白多一跳的三节点解，
    # beam=400 才找到那个两节点解。
    #
    # 本文件开头写的是「单节点/少跳解显式优先」，所以 0 跳与 1 跳不交给 beam，直接
    # 穷举（O(|V|²·L)，前缀和让每次内存判定 O(1)），并进候选集。只增不减：按同一个
    # 目标函数取最小，结果不会比原来差。≥2 跳仍然交给 beam。
    #
    # 先判内存、再碰网络：只对内存可行的节点对调 p50 / blocked，不额外触发探测。
    #
    # 默认不开（exhaustive_short=False）：它会改变主流程的规划结果，见 docstring。
    cum = [0.0]
    for l in range(lo0, hi0 + 1):
        cum.append(cum[-1] + spec.gb_per_layer(l) + spec.kv_gb_per_layer)

    def gb(a: int, b: int) -> float:  # 段内第 a..b 层（1-based，含两端）
        return (cum[b] - cum[a - 1] + (spec.head_extra_gb if a == 1 else 0.0)
                + (spec.tail_extra_gb if b == n_layers else 0.0))

    scale = spec.ms_per_layer_scale

    # --- 链长上限：按段的大小定（命题 III.5.2） ---------------------------- #
    # 一条段最少要 n_min = ⌈驻留量 / 最大单机可用⌉ 台节点；链长上限取 n_min + hop_slack。
    # 不设上限时，beam 的完成下界又过于乐观（最小跳 p50 可以接近 0、每加一台节点的
    # 稳定性罚 μ 不计入下界），「每台只放一两层」的短前缀会一直排在前面，最后拼成
    # 一条长链。实测（15 台真拓扑，L₀=8、前段装全部专家、预留 2.5GB、租金按最大取）：
    # mbpp 后段 31.8GB，N10+N11 两台 1 跳即可（评分 21.8），beam=12 交出的是
    # N09+…+N04 共 9 台 8 跳（评分 89.7），把剩余节点全部吃光，前段无处可放。
    #
    # 实现为**可行性剪枝**：前缀已用 k 台、剩余驻留 r GB 时，至少还要 ⌈r / 最大单机⌉ 台；
    # k + 它 > 上限就丢掉。这条下界对任何补全方式都成立，所以只剪掉违反上限的解，
    # 不会误剪合规解；分支定界的可容许性不受影响。
    # legacy_beam 下不生效：旧 beam 只用来逐字节复现旧方案。
    max_cap = max(nodes[v].usable_gb for v in avail)
    if hop_slack is not None and not legacy_beam and max_cap > 0:
        node_cap = max(1, ceil(gb(1, n_layers) / max_cap - 1e-9)) + max(0, hop_slack)
    else:
        node_cap = None

    def within_cap(n_nodes: int, placed: int) -> bool:
        if node_cap is None:
            return True
        rem = gb(placed + 1, n_layers) if placed < n_layers else 0.0
        need = ceil(rem / max_cap - 1e-9) if rem > 1e-9 else 0
        return n_nodes + need <= node_cap

    exact_best: _Partial | None = None
    exact_score = float("inf")

    def offer(p: _Partial) -> None:
        nonlocal exact_best, exact_score
        s = score_final(p)
        if s < exact_score:
            exact_best, exact_score = p, s

    for v in (avail if exhaustive_short else ()):
        if not spec.head_ok(v):
            continue
        nv = nodes[v]
        if gb(1, n_layers) <= nv.usable_gb + 1e-9 and spec.tail_ok(v):
            offer(_Partial((v,), ((lo0, hi0),), n_layers,
                           n_layers * nv.ms_per_layer * scale, 0.0, 0.0))
        if node_cap is not None and node_cap < 2:
            continue
        for k in range(1, n_layers):
            if gb(1, k) > nv.usable_gb + 1e-9:
                break
            rest = gb(k + 1, n_layers)
            for v2 in avail:
                if v2 == v or not spec.tail_ok(v2) or rest > nodes[v2].usable_gb + 1e-9:
                    continue
                if net.blocked(v, v2):
                    continue
                offer(_Partial(
                    (v, v2), ((lo0, lo0 + k - 1), (lo0 + k, hi0)), n_layers,
                    (k * nv.ms_per_layer + (n_layers - k) * nodes[v2].ms_per_layer) * scale,
                    net.p50(v, v2), net.jitter(v, v2),
                ))

    # --- 起点：每个候选起点 × 每种首段长度 ------------------------------- #
    starts = [v for v in avail if spec.head_ok(v)]
    starts.sort(key=lambda v: (-nodes[v].usable_gb, -nodes[v].avail))
    starts = starts[: max(prune_topk, beam_width)]

    beam: list[_Partial] = []
    for v in starts:
        node = nodes[v]
        for seg_len in range(1, n_layers + 1):
            lo, hi = lo0, lo0 + seg_len - 1
            if not _mem_ok(spec, lo, hi, node):
                break  # 内存单调：更长只会更不可行
            if seg_len == n_layers and not spec.tail_ok(v):
                continue
            if not within_cap(1, seg_len):
                continue
            beam.append(
                _Partial(
                    nodes=(v,),
                    splits=((lo, hi),),
                    placed=seg_len,
                    compute_ms=seg_len * node.ms_per_layer * spec.ms_per_layer_scale,
                    hop_ms=0.0,
                    jitter_ms=0.0,
                )
            )
    if not beam and exact_best is None:
        return None

    # --- 完整解不参与截断 -------------------------------------------------- #
    # 半成品的分数 = 已花代价 + 剩余部分的**乐观下界**；完整解的分数是真实代价。
    # 两者放进同一张榜截断，半成品永远排在「由它自己补完的完整解」前面 ——
    # 于是完整解被挤出 beam，半成品继续加节点，最后交出的是一条长链。
    # 实测（15 台真拓扑、L₀=8）：no_robots 后段第 1 轮就有 1 跳、24.2 分的
    # 完整解，排第 217 名被截；beam 一路扩到第 7 轮，交出 8 台 7 跳、107.5 分的
    # 链，再被跳数下限否决 —— 那一池整个建不出来。加宽 beam 到 800 也一样。
    # 所以：完整解一出现就收下，截断只在半成品之间做。
    if legacy_beam:
        # 旧行为（--beam-legacy）：完整解与半成品同榜截断、不剪枝。只为复现旧方案。
        beam.sort(key=score_partial)
        beam = beam[:beam_width]
        finished: list[_Partial] = [p for p in beam if p.placed == n_layers]
        frontier = [p for p in beam if p.placed < n_layers]
    else:
        finished = [p for p in beam if p.placed == n_layers]
        frontier = sorted((p for p in beam if p.placed < n_layers),
                          key=score_partial)[:beam_width]

    # --- 用当前最好的完整解剪掉没希望的半成品（分支定界） ------------------ #
    # 半成品的分数是它任何补全方式的下界，前提是下界真是下界：
    #   * min_hop_p50 在 ≤ 40 个节点时取全对最小值（更大的池是抽样估计，不保证）；
    #   * 目标函数各项权重非负（新增节点只会加分，不会减分）。
    # 满足时，乐观分已不低于现有最好完整解的半成品，补完只会更差，可以丢掉。
    prune_ok = (not legacy_beam and len(avail) <= 40 and objective.mu_ms >= 0 and objective.jitter_w >= 0
                and objective.rent_nu_ms >= 0 and objective.good_gamma_ms >= 0
                and objective.endpoint_w >= 0)

    def incumbent() -> float:
        best = min((score_final(p) for p in finished), default=float("inf"))
        return min(best, exact_score)

    def bound(front: list[_Partial]) -> list[_Partial]:
        if not prune_ok:
            return front
        inc = incumbent()
        return [p for p in front if score_partial(p) < inc]

    frontier = bound(frontier)

    # --- 逐轮扩展 --------------------------------------------------------- #
    while frontier:
        cand: list[_Partial] = []
        for p in frontier:
            remaining = n_layers - p.placed
            nxt_lo = lo0 + p.placed
            used = set(p.nodes)
            neigh = prune_neighbors(p.last, avail, nodes, net, prune_topk, peers)
            for v2 in neigh:
                if v2 in used:
                    continue
                node2 = nodes[v2]
                hop = net.p50(p.last, v2)
                jit = net.jitter(p.last, v2)
                for seg_len in range(1, remaining + 1):
                    lo, hi = nxt_lo, nxt_lo + seg_len - 1
                    if not _mem_ok(spec, lo, hi, node2):
                        break
                    is_last = seg_len == remaining
                    if is_last and not spec.tail_ok(v2):
                        continue
                    if not within_cap(len(p.nodes) + 1, p.placed + seg_len):
                        continue
                    cand.append(
                        _Partial(
                            nodes=p.nodes + (v2,),
                            splits=p.splits + ((lo, hi),),
                            placed=p.placed + seg_len,
                            compute_ms=p.compute_ms
                            + seg_len * node2.ms_per_layer * spec.ms_per_layer_scale,
                            hop_ms=p.hop_ms + hop,
                            jitter_ms=p.jitter_ms + jit,
                        )
                    )
        if not cand:
            break
        if legacy_beam:
            cand.sort(key=score_partial)
            cand = cand[:beam_width]
            finished.extend(p for p in cand if p.placed == n_layers)
            frontier = [p for p in cand if p.placed < n_layers]
            continue
        finished.extend(p for p in cand if p.placed == n_layers)   # 不参与截断
        frontier = bound(sorted((p for p in cand if p.placed < n_layers),
                                key=score_partial)[:beam_width])

    if exact_best is not None:
        finished.append(exact_best)
    if not finished:
        return None
    best = min(finished, key=score_final)
    return Segment(
        kind=spec.kind,
        task=spec.task,
        nodes=best.nodes,
        splits=best.splits,
        compute_ms=best.compute_ms,
        hop_ms=best.hop_ms,
        jitter_ms=best.jitter_ms,
    )
