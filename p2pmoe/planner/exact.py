"""小池子上的精确段放置：枚举节点链 + 逐链 DP 切层。

deploy_path 的 beam 对「要 3–4 台才放得下、剩下的又多是小卡」的段不可靠：
半成品的完成下界假设剩余层都能放进最大那台，而那台往往已在前缀里。实测
（vLLM 输入，剩 N10–N15 放 50.5GB 的全专家后段）：beam=12 交出一台 T4 只放
1 层的四节点链（87.3ms），beam=50 返回 None，beam=200 才给出 N10+N11+N13（85.9ms）。

这里换成精确解：

* 枚举所有长度 ≤ 链长上限的有序节点链（上限 = ⌈驻留量 / 最大单机⌉ + hop_slack，
  与 deploy_path 同一条规则）；链上节点的可用量之和装不下整段的直接跳过；
* 链固定后，跳的代价、端点项、稳定性项都定了，只剩「每台放几层」—— 计算时延
  对层数是线性的，内存对区间是单调的，DP 精确求最小计算时延；
* 按与 deploy_path 相同的 Objective.score 取最小（并列取字典序最小的链，结果确定）。

规模：15 台、上限 3 台时约 2700 条链，每条 DP O(n·L²)，一次几秒内。只给
planner/split.py 用；六步流水线仍用 deploy_path（改了会让已有方案变样）。
"""

from __future__ import annotations

from itertools import permutations
from math import ceil
from typing import Mapping, Sequence

from .network import MeasurementCache
from .types import Node, Objective, Segment, SegmentSpec

__all__ = ["exact_path"]


def exact_path(
    spec: SegmentSpec,
    avail: Sequence[str],
    nodes: Mapping[str, Node],
    net: MeasurementCache,
    objective: Objective,
    *,
    hop_slack: int | None = 1,
    max_nodes: int | None = None,
) -> Segment | None:
    lo0, n = spec.layer_lo, spec.n_layers
    avail = sorted(avail)
    if n <= 0 or not avail:
        return None

    cum = [0.0]
    for l in range(lo0, spec.layer_hi + 1):
        cum.append(cum[-1] + spec.gb_per_layer(l) + spec.kv_gb_per_layer)

    def gb(a: int, b: int) -> float:  # 段内第 a..b 层（1-based，含两端）
        return (cum[b] - cum[a - 1] + (spec.head_extra_gb if a == 1 else 0.0)
                + (spec.tail_extra_gb if b == n else 0.0))

    total = gb(1, n)
    max_cap = max(nodes[v].usable_gb for v in avail)
    if max_cap <= 0:
        return None
    n_min = max(1, ceil(total / max_cap - 1e-9))
    cap_n = n_min + (2 if hop_slack is None else max(0, hop_slack))
    if max_nodes is not None:
        cap_n = min(cap_n, max_nodes)
    cap_n = min(cap_n, len(avail), n)
    scale = spec.ms_per_layer_scale
    INF = float("inf")

    def split_dp(seq: tuple[str, ...]) -> tuple[float, list[tuple[int, int]]] | None:
        k = len(seq)
        # best[i][j]：前 i 台放完段内前 j 层的最小计算时延
        best = [[INF] * (n + 1) for _ in range(k + 1)]
        arg = [[-1] * (n + 1) for _ in range(k + 1)]
        best[0][0] = 0.0
        for i, v in enumerate(seq, start=1):
            cap, ms = nodes[v].usable_gb + 1e-9, nodes[v].ms_per_layer * scale
            prev, cur, a = best[i - 1], best[i], arg[i]
            for j in range(i, n - (k - i) + 1):
                for s in range(j - 1, i - 2, -1):   # 本台放 s+1..j；s 越小区间越大
                    if gb(s + 1, j) > cap:
                        break
                    if prev[s] < INF:
                        c = prev[s] + (j - s) * ms
                        if c < cur[j] - 1e-12:
                            cur[j], a[j] = c, s
        if best[k][n] == INF:
            return None
        splits: list[tuple[int, int]] = []
        j = n
        for i in range(k, 0, -1):
            s = arg[i][j]
            splits.append((lo0 + s, lo0 + j - 1))
            j = s
        return best[k][n], splits[::-1]

    caps = {v: nodes[v].usable_gb for v in avail}
    best_seg: tuple[float, tuple[str, ...], list[tuple[int, int]], float, float, float] | None = None
    for m in range(1, cap_n + 1):
        for seq in permutations(avail, m):
            if sum(caps[v] for v in seq) < total - 1e-9:
                continue
            if not spec.head_ok(seq[0]) or not spec.tail_ok(seq[-1]):
                continue
            if any(net.blocked(a, b) for a, b in zip(seq, seq[1:])):
                continue
            got = split_dp(seq)
            if got is None:
                continue
            comp, splits = got
            hop = sum(net.p50(a, b) for a, b in zip(seq, seq[1:]))
            jit = sum(net.jitter(a, b) for a, b in zip(seq, seq[1:]))
            sc = objective.score(comp, hop, jit, seq, nodes, partial=False)
            if best_seg is None or sc < best_seg[0] - 1e-9:
                best_seg = (sc, seq, splits, comp, hop, jit)
    if best_seg is None:
        return None
    _, seq, splits, comp, hop, jit = best_seg
    return Segment(kind=spec.kind, task=spec.task, nodes=tuple(seq),
                   splits=tuple(splits), compute_ms=comp, hop_ms=hop, jitter_ms=jit)
