"""影子租金 R_F 的三种口径（pipeline.select_rent_nodes）。

none（真机规划默认）：前段层数压到每台设备都放得下，不再给前段预留节点。
"""

from p2pmoe.planner.pipeline import select_rent_nodes
from p2pmoe.planner.types import Node, SegmentSpec

NODES = [Node(id="big1", tier="48", mem_gb=48, ms_per_layer=1.0, reserve_gb=0),
         Node(id="big2", tier="48", mem_gb=48, ms_per_layer=1.0, reserve_gb=0),
         Node(id="mid", tier="24", mem_gb=24, ms_per_layer=1.0, reserve_gb=0),
         Node(id="small", tier="16", mem_gb=16, ms_per_layer=1.0, reserve_gb=0)]
FRONT = SegmentSpec(kind="front", task=None, layer_lo=1, layer_hi=2,
                    gb_per_layer=lambda l: 10.0, kv_gb_per_layer=0.0)     # 前段 20GB


def test_none_reserves_nothing():
    assert select_rent_nodes(NODES, FRONT, 2, policy="none") == frozenset()


def test_largest_takes_the_biggest():
    assert select_rent_nodes(NODES, FRONT, 2, policy="largest") == {"big1", "big2"}


def test_smallest_sufficient_skips_nodes_that_cannot_hold_the_front():
    assert select_rent_nodes(NODES, FRONT, 2, policy="smallest_sufficient") == {"mid", "big1"}
