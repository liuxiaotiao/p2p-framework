"""补位（--fill-idle）：六步流水线跑完之后，把还空着的节点全部用上。

为什么主流程会留空位
--------------------
六步流水线的目标是**组合矩阵均匀**：任意前段配任意同 task 后段，时延都在一个
窄带里，这样盲绑才是零后悔的（推论 III.3.2）。为此它有三道闸会主动放弃节点：

* Step 2 的 θ 折扣 —— 内存能建 5 条，只规划 4 条；
* Step 3 的跳数下限 —— 用剩余小节点拼出来的段跳数太多，当作建不出；
* Step 4 的公共带 —— 出口不在带内的节点做不了前段，后段条数随之被压回来。

被放弃的恰好是远端站点和小显存的卡。在 15 台真机上，默认参数只用 8 台；
L₀=11 时 15 台都有角色，但 4 台是备胎，协调器从不把备胎放进池子。

为什么可以放宽
--------------
这三道闸是为**网络主导**的环境设计的（代码注释原话：「每多一跳即数十毫秒/token」）。
实测这个集群是**计算主导**：逐对单向中位 3ms，而每 token 计算约 440ms。7–10ms 的
网络不均匀只占每 token 时延的 2% 左右，却让 4–7 台机器闲着。

这一轮做什么
------------
按顺序：

1. **加通道** —— 备胎先拆回空闲池，然后**一前一后成对地**建：后段给公平比最低的
   task（最堵的那个池），前段从剩下的节点里拼。一对都建成才提交 —— 并发上限是
   min(前段条数, 后段条数)，只加一边吞吐不涨。不过公共带、不设跳数下限。
   拆开也拼不出新通道的备胎，原样提拔成前段。
2. **分摊** —— 剩下拼不成整段的节点，挂到显存最紧的那台节点旁边，接走它的一部分
   层。每挂一台多一跳，但给 93% 显存的卡留出余量（真机上 N08 就压在那里）。
   **先挂补位段**，补位段都挂不上才动核心段 —— 核心段有终审担保，不该替补位买单。
3. 仍然一层都装不下的节点，**点名报出来**，不假装用上了。

代价要讲清楚：补出来的通道不在公共带内，盲绑不再严格零后悔。每条补位段都标了
origin，清单校验里涉及它们的「公共带 / w_cap / 抖动闸」降为警告，其余校验
（排他、内存、层覆盖、并集、驻留集、组合矩阵完整）照旧严格。

默认关闭：不开 --fill-idle 时规划结果与论文描述的算法逐字一致。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Mapping, Sequence

from .capacity import fair_ratios
from .memory import make_back_spec, make_front_spec
from .network import MeasurementCache
from .solver import deploy_path, segment_from_nodes
from .types import Node, Objective, PlannerConfig, Segment, SegmentSpec, TaskProfile

__all__ = ["FillResult", "fill_idle"]


@dataclass
class FillResult:
    fronts: list[Segment]
    standby: list[Segment]
    backs: dict[str, list[Segment]]
    origin: dict[Segment, str] = field(default_factory=dict)
    """非核心段的来历："promoted"（备胎提拔）/ "fill"（补位新建）。"""
    spread: list[tuple[str, str]] = field(default_factory=list)
    """(空闲节点, 它加入的段 label)。"""
    endpoint_moved: set[Segment] = field(default_factory=set)
    """分摊时首尾节点变了的段 —— 它们的组合可能不再落在公共带内。"""
    still_idle: list[str] = field(default_factory=list)
    log: list[str] = field(default_factory=list)

    @property
    def relaxed(self) -> set[Segment]:
        """清单校验里要放宽带内约束的段。"""
        return {s for s, o in self.origin.items() if o == "fill"} | self.endpoint_moved


def _n_back(backs: Mapping[str, Sequence[Segment]]) -> int:
    return sum(len(v) for v in backs.values())


def _used(fronts, standby, backs) -> set[str]:
    out: set[str] = set()
    for s in list(fronts) + list(standby) + [b for v in backs.values() for b in v]:
        out |= set(s.nodes)
    return out


def fill_idle(
    *,
    fronts: Sequence[Segment],
    standby: Sequence[Segment],
    backs: Mapping[str, Sequence[Segment]],
    tasks: Sequence[TaskProfile],
    model,
    union_experts,
    l0: int,
    node_map: Mapping[str, Node],
    net: MeasurementCache,
    cfg: PlannerConfig,
) -> FillResult:
    """在主流程结果上补位。不改核心段（分摊除外），只往池里加。"""
    fronts = list(fronts)
    standby_q = list(standby)
    backs = {u: list(v) for u, v in backs.items()}
    for t in tasks:
        backs.setdefault(t.name, [])
    res = FillResult(fronts=fronts, standby=[], backs=backs)
    log = res.log

    idle = set(node_map) - _used(fronts, standby_q, backs)
    core_delay = [s.delay_ms for s in fronts] + [b.delay_ms for v in backs.values() for b in v]
    ref = median(core_delay) if core_delay else 0.0
    log.append(
        f"[补位] 主流程用了 {len(node_map) - len(idle) - sum(len(s.nodes) for s in standby_q)} 台，"
        f"备胎 {len(standby_q)} 条，空闲 {len(idle)} 台 {sorted(idle)}"
    )

    front_spec = make_front_spec(model, union_experts, l0)  # 不限出口：补位不过公共带
    back_specs = {t.name: make_back_spec(model, t, l0) for t in tasks}
    lam = {t.name: t.lam for t in tasks}
    obj = Objective(mu_ms=cfg.mu_ms, jitter_w=cfg.jitter_w)

    def _note(seg: Segment, what: str) -> None:
        d = seg.delay_ms - ref
        log.append(
            f"[补位] {what} {seg.label()}：段时延 {seg.delay_ms:.1f}ms"
            f"（核心段中位 {ref:.1f}ms，{'+' if d >= 0 else ''}{d:.1f}），{seg.hops} 跳"
        )

    # ---------------- 1. 加通道：一前一后成对地加 ----------------------------
    # 备胎先拆回空闲池，而不是原样提拔。原样提拔只多一条前段 —— 而并发上限是
    # min(前段条数, 后段条数)，后段没加，吞吐一点不涨。拆开之后同样这几台机器
    # 往往能拼出「一条后段 + 一条更短的前段」，那才是多一条通道。拼不成的话，
    # 下面再把备胎原样提拔回来，不吃亏。
    for s in standby_q:
        idle |= set(s.nodes)

    def build_front(pool: set[str]) -> Segment | None:
        return deploy_path(front_spec, sorted(pool), node_map, net, obj,
                           beam_width=cfg.beam_width_front, prune_topk=cfg.prune_topk,
                           exhaustive_short=True)

    def build_back(u: str, pool: set[str]) -> Segment | None:
        return deploy_path(back_specs[u], sorted(pool), node_map, net, obj,
                           beam_width=cfg.beam_width, prune_topk=cfg.prune_topk,
                           exhaustive_short=True)

    while True:
        nf, nb = len(fronts), _n_back(backs)
        fair = fair_ratios({u: len(backs[u]) for u in lam}, lam)
        order = sorted(lam, key=lambda u: (fair[u], -lam[u]))   # 最堵的池排最前

        if nf < nb:                       # 后段多出来 → 只缺前段
            f = build_front(idle)
            if f is None:
                break
            fronts.append(f); idle -= set(f.nodes); res.origin[f] = "fill"
            _note(f, "新建前段")
            continue

        if nf > nb:                       # 前段多出来 → 只缺后段
            got = next(((u, b) for u in order if (b := build_back(u, idle))), None)
            if got is None:
                break
            u, b = got
            backs[u].append(b); idle -= set(b.nodes); res.origin[b] = "fill"
            _note(b, f"新建后段（{u} 公平比 {fair[u]:.2f}，当前最堵）")
            continue

        # 一样多 → 必须一对一起建成才提交，否则半条通道只是占着机器
        pair = None
        for u in order:
            b = build_back(u, idle)
            if b is None:
                continue
            f = build_front(idle - set(b.nodes))
            if f is not None:
                pair = (u, b, f)
                break
        if pair is None:
            log.append(f"[补位] 空闲 {sorted(idle)} 凑不出一整条通道（一前一后）—— 停止加通道")
            break
        u, b, f = pair
        backs[u].append(b); fronts.append(f)
        idle -= set(b.nodes) | set(f.nodes)
        res.origin[b] = res.origin[f] = "fill"
        _note(b, f"新建后段（{u} 公平比 {fair[u]:.2f}，当前最堵）")
        _note(f, "新建前段（与上面那条后段配成一条通道）")

    # 节点原封未动的备胎，原样提拔成前段 —— 它本来就在公共带内，比拆散了分摊好
    for s in standby_q:
        if set(s.nodes) <= idle:
            fronts.append(s)
            idle -= set(s.nodes)
            res.origin[s] = "promoted"
            _note(s, "备胎提拔为前段（拆开也拼不出新通道）")
    standby_q = []

    # ---------------- 2. 分摊：挂到显存最紧的节点旁边 ------------------------
    def spec_of(seg: Segment) -> SegmentSpec:
        return front_spec if seg.kind == "front" else back_specs[seg.task]

    def util(seg: Segment, i: int) -> float:
        lo, hi = seg.splits[i]
        return spec_of(seg).resident_gb(lo, hi) / node_map[seg.nodes[i]].usable_gb

    def replace(old: Segment, new: Segment) -> None:
        for lst in [fronts] + list(backs.values()):
            for k, s in enumerate(lst):
                if s is old:
                    lst[k] = new
                    if old in res.origin:
                        res.origin[new] = res.origin.pop(old)
                    if old in res.endpoint_moved:
                        res.endpoint_moved.discard(old)
                        res.endpoint_moved.add(new)
                    return

    def is_core(seg: Segment) -> bool:
        # 主流程建的段与原样提拔的备胎都在公共带内，终审替它们作过保 ——
        # 往里插一跳会让这些「有保证」的通道变慢。补位新建的段本来就在带外。
        return res.origin.get(seg) != "fill"

    for r in sorted(idle, key=lambda v: -node_map[v].usable_gb):
        cap_r = node_map[r].usable_gb
        cands = []
        for seg in fronts + [b for v in backs.values() for b in v]:
            for i, (lo, hi) in enumerate(seg.splits):
                if hi > lo:
                    cands.append((util(seg, i), seg, i))
        # 先挂补位段，补位段都挂不上才动核心段；同一类里先救显存最紧的那台。
        # 实测：不分先后时 N15 被挂进核心段 mbpp[N06+N08]，那条有终审担保的
        # 通道凭空慢了 30ms。
        cands.sort(key=lambda c: (is_core(c[1]), -c[0]))

        placed = False
        for u0, seg, i in cands:
            spec = spec_of(seg)
            lo, hi = seg.splits[i]
            v = seg.nodes[i]
            last = len(seg.nodes) - 1
            # 尽量插在段中间，不动首尾 —— 首尾变了，组合就可能出公共带
            move_head = (i == last and i > 0)
            if move_head:
                nb = seg.nodes[i - 1]
                if net.blocked(nb, r) or net.blocked(r, v):
                    continue
            else:
                nb = seg.nodes[i + 1] if i < last else None
                if net.blocked(v, r) or (nb is not None and net.blocked(r, nb)):
                    continue

            best = None
            for k in range(1, hi - lo + 1):
                if move_head:
                    mv, kp = (lo, lo + k - 1), (lo + k, hi)
                else:
                    mv, kp = (hi - k + 1, hi), (lo, hi - k)
                if spec.resident_gb(*mv) > cap_r + 1e-9:
                    continue
                peak = max(spec.resident_gb(*kp) / node_map[v].usable_gb,
                           spec.resident_gb(*mv) / cap_r)
                if best is None or peak < best[0]:
                    best = (peak, mv, kp)
            if best is None:
                continue

            peak, mv, kp = best
            nodes = list(seg.nodes)
            splits = list(seg.splits)
            if move_head:
                nodes[i:i + 1] = [r, v]
                splits[i:i + 1] = [mv, kp]
            else:
                nodes[i:i + 1] = [v, r]
                splits[i:i + 1] = [kp, mv]
            new = segment_from_nodes(spec, nodes, splits, node_map, net)
            moved_ep = (new.head != seg.head) or (new.tail != seg.tail)
            core = is_core(seg)
            replace(seg, new)
            if moved_ep:
                res.endpoint_moved.add(new)
            res.spread.append((r, new.label()))
            log.append(
                f"[补位·分摊] {r} 接走 {v} 的层 {mv[0]}–{mv[1]}"
                f"（{v} 显存 {u0:.0%} → {spec.resident_gb(*kp) / node_map[v].usable_gb:.0%}，"
                f"{r} {spec.resident_gb(*mv) / cap_r:.0%}）；{new.label()} "
                f"段时延 {seg.delay_ms:.1f} → {new.delay_ms:.1f}ms"
                + ("；首尾变了，组合不再保证在带内" if moved_ep else "")
                + ("；⚠ 动了核心段（补位段都挂不上）" if core else "")
            )
            placed = True
            break
        if not placed:
            res.still_idle.append(r)

    if res.still_idle:
        log.append(
            f"[补位] 仍空着 {sorted(res.still_idle)} —— 装不下任何一段里的任何一层，"
            f"或到所有候选邻居的链路都被抖动闸屏蔽"
        )
    else:
        log.append(f"[补位] 全部 {len(node_map)} 台都已上岗")
    res.standby = standby_q
    return res
