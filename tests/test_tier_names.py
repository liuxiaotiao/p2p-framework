"""同一型号因可用量不同落进两个档时，档名必须区分开（capacity.tier_names）。

回归：实测显存下 7 台 A30 可用 24.43GB、N01 因 CUDA context 大一点只有 24.13GB，
按 0.5GB 取整分成 24.5 / 24.0 两档。旧代码两档同名 "A30"，配平的供给表被后写的
那档（1 台）覆盖 —— 8 台 A30 当成 1 台，可建条数从 5 掉到 4。
"""

from p2pmoe.planner.capacity import classify_tiers, estimate_capacity_by_tier, tier_names
from p2pmoe.planner.types import Node, SegmentSpec, TaskProfile


def _n(i, mem, tier="A30"):
    return Node(id=f"N{i}", tier=tier, mem_gb=mem, ms_per_layer=1.0, reserve_gb=0.0)


def test_split_type_gets_distinct_names():
    nodes = [_n(1, 24.13)] + [_n(i, 24.43) for i in range(2, 9)] + [_n(9, 15.0, "T4")]
    names = tier_names(nodes)
    assert names["N1"] != names["N2"] and names["N2"] == names["N8"]
    assert names["N9"] == "T4"                         # 没分档的型号保持原名
    tiers = {t.name: t.count for t in classify_tiers(nodes)}
    assert sum(tiers.values()) == 9 and len(tiers) == 3


def test_capacity_counts_every_node_of_a_split_type():
    front = SegmentSpec(kind="front", task=None, layer_lo=1, layer_hi=2,
                        gb_per_layer=lambda l: 5.0, kv_gb_per_layer=0.0)
    back = SegmentSpec(kind="back", task="t", layer_lo=3, layer_hi=6,
                       gb_per_layer=lambda l: 8.0, kv_gb_per_layer=0.0)      # 32GB → 两台
    tasks = [TaskProfile("t", 1.0, 1)]
    same = [_n(i, 24.4) for i in range(1, 10)]
    split = [_n(1, 24.1)] + [_n(i, 24.4) for i in range(2, 10)]
    a = estimate_capacity_by_tier(same, front, {"t": back}, tasks)
    b = estimate_capacity_by_tier(split, front, {"t": back}, tasks)
    assert a.n_max_slots == b.n_max_slots == 3            # 9 台：3 前 + 3×2 后
