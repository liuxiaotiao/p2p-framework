"""原始模型路径（--raw-fill）：六步流水线跑完之后，用剩下的节点建「全部专家」的完整路径。

和 --quota-equal 一起用：Step 2 先按 task 均分 ⌊N_work / task 数⌋ 条，Step 3–6 照原算法建
（前后段盲绑、任意组合、公共带、回环裁剪、终审都不变）；均分剩下的那部分显存（不够每个
task 再各分一条）在这里变成原始模型路径：

1. 剩下的节点 = 全部 − 主流程的前段 / 后段（Step 5 超建、Step 6 裁掉的前段也算剩下）。
   备胎前段也拆回来 —— 协调器从不把备胎放进池子，它们和空着没区别；拆了也没用上的
   备胎原样留作备胎；
2. 后段：第 L₀+1..L 层装**全部**专家（不裁剪），放进共享池 SHARED_POOL（"any"）——
   识别成哪个 task 都能接；在线时某个 task 的专属后段全忙，就溢出到这里
   （runtime/coordinator.py 的 _bind）；
3. 前段：与主流程的前段内容相同，并入前段池一起盲绑；
4. 一前一后成对地建，都建成才提交，直到放不下为止；仍然空着的节点点名报出来。

放置：后段用 planner/exact.py（枚举节点链 + 逐链 DP 切层，目标与 deploy_path 相同）——
全专家后段 50GB 要 3 台以上，剩下的又多是小卡，beam 在这种情况下不可靠（beam=12 交出
一台 T4 只放 1 层的四节点链，beam=50 返回 None）；前段与补位一样用 deploy_path（短段 0/1
跳精确穷举）。

原始模型路径不过公共带：涉及它们的组合的带内 / w_cap / 抖动检查在清单里记为警告。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .exact import exact_path
from .experts import ExpertPlacement
from .memory import make_back_spec, make_front_spec
from .network import MeasurementCache
from .solver import deploy_path
from .types import Node, Objective, PlannerConfig, Segment, TaskProfile

__all__ = ["RAW_POOL", "RawFillResult", "raw_placement", "fill_raw"]

RAW_POOL = "any"
"""= runtime.coordinator.SHARED_POOL（这里不 import runtime，免得规划器依赖在线代码）。"""


@dataclass
class RawFillResult:
    fronts: list[Segment]
    standby: list[Segment]
    backs: dict[str, list[Segment]]
    placement: ExpertPlacement
    origin: dict[Segment, str] = field(default_factory=dict)
    still_idle: list[str] = field(default_factory=list)
    log: list[str] = field(default_factory=list)

    @property
    def relaxed(self) -> set[Segment]:
        return set(self.origin)


def raw_placement(front_experts, n_layers: int, n_experts: int, l0: int) -> ExpertPlacement:
    """原始模型后段的驻留集：第 L₀+1 层起全部专家。

    前 L₀ 层后段不装，这里照抄前段的集合 —— 清单校验的「前段 ⊇ 每个后段」只看前 L₀ 层，
    抄前段就不会因为前段是并集（没开 --front-full）而误报。"""
    full = frozenset(range(n_experts))
    sets = tuple(
        (frozenset(front_experts.at(l)) if isinstance(front_experts, ExpertPlacement)
         else full) if l <= l0 else full
        for l in range(1, n_layers + 1))
    return ExpertPlacement(name=RAW_POOL, n_layers=n_layers, sets=sets,
                           achieved_coverage=tuple(1.0 for _ in range(n_layers)))


def fill_raw(
    *,
    fronts: Sequence[Segment],
    standby: Sequence[Segment],
    backs: Mapping[str, Sequence[Segment]],
    model,
    front_experts,
    l0: int,
    node_map: Mapping[str, Node],
    net: MeasurementCache,
    cfg: PlannerConfig,
) -> RawFillResult:
    fronts = list(fronts)
    backs = {u: list(v) for u, v in backs.items()}
    used = {v for s in list(fronts) + [b for x in backs.values() for b in x] for v in s.nodes}
    idle = set(node_map) - used          # 含备胎的节点
    plc = raw_placement(front_experts, model.n_layers, model.n_experts, l0)
    spec = make_back_spec(model, TaskProfile(name=RAW_POOL, lam=0.0,
                                             experts_per_layer=model.n_experts,
                                             placement=plc), l0)
    front_spec = make_front_spec(model, front_experts, l0)
    obj = Objective(mu_ms=cfg.mu_ms, jitter_w=cfg.jitter_w)
    res = RawFillResult(fronts=fronts, standby=[], backs=backs, placement=plc)
    log = res.log
    log.append(
        f"[原始模型路径] 主流程用了 {len(used)} 台"
        + (f"（另有备胎 {[s.label() for s in standby]}，拆回来一起用）" if standby else "")
        + f"，剩 {len(idle)} 台 {sorted(idle)}，"
        f"可用显存合计 {sum(node_map[v].usable_gb for v in idle):.1f}GB；"
        f"一条 = 全专家后段 {spec.total_gb():.1f}GB + 前段 {front_spec.total_gb():.1f}GB"
    )
    raws: list[Segment] = []
    while True:
        b = exact_path(spec, sorted(idle), node_map, net, obj, hop_slack=cfg.max_hops_slack)
        if b is None:
            log.append(f"[原始模型路径] 剩下 {sorted(idle)} 放不下一条全专家后段 —— 停")
            break
        f = deploy_path(front_spec, sorted(idle - set(b.nodes)), node_map, net, obj,
                        beam_width=cfg.beam_width_front, prune_topk=cfg.prune_topk,
                        exhaustive_short=True, legacy_beam=cfg.beam_legacy,
                        hop_slack=cfg.max_hops_slack)
        if f is None:
            log.append(f"[原始模型路径] 后段 {b.label()} 放得下，但剩下的节点放不了前段 —— 停")
            break
        idle -= set(b.nodes) | set(f.nodes)
        raws.append(b)
        fronts.append(f)
        res.origin[b] = res.origin[f] = "raw"
        log.append(
            f"[原始模型路径] 后段 {b.label()}：{b.delay_ms:.1f}ms（{b.hops} 跳，"
            + "、".join(f"{v} {lo}–{hi}" for v, (lo, hi) in zip(b.nodes, b.splits))
            + f"）；前段 {f.label()}：{f.delay_ms:.1f}ms —— 前段并入前段池，后段进共享池 {RAW_POOL}"
        )
    if raws:
        backs[RAW_POOL] = raws
    for s in standby:                    # 节点原封未动的备胎照旧当备胎
        if set(s.nodes) <= idle:
            res.standby.append(s)
            idle -= set(s.nodes)
    res.still_idle = sorted(idle)
    log.append(
        f"[原始模型路径] 建出 {len(raws)} 条"
        + (f"；仍空着 {res.still_idle}" if res.still_idle else f"；全部 {len(node_map)} 台上岗")
    )
    return res
