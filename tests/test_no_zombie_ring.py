"""静态模式下 max_tokens 结束的请求不能留下「僵尸环」。

环是自转的（tail(b) 采样完直接回环给 head(f)），协调器发 release 时环上总还有
一步在途。早先节点收到这一步就重建状态和 KV 接着算，前段出口又有 static_peer
可送 —— 请求永远转下去。sel_static2 实测：同一通道逐条变慢（no_robots
281→1111ms/token），N13 最后 CUDA OOM。
"""

from __future__ import annotations

import time

import pytest

from p2pmoe.runtime.wire import rpc

from test_lr_classifier import toy_plan  # noqa: F401  （复用 fixture）


def _active(cl) -> dict[str, int]:
    out = {}
    for nid, port in cl.ports.items():
        r = rpc(("127.0.0.1", port), {"type": "capabilities"}, timeout=8.0)
        out[nid] = int(r.get("active_reqs", 0))
    return out


@pytest.mark.slow
def test_static_max_tokens_leaves_no_ring_running(toy_plan) -> None:  # noqa: F811
    from p2pmoe.runtime.coordinator import LocalCluster

    t = toy_plan
    man = t["res"].manifest
    fronts = sorted(s for s, d in man.segments.items() if d["role"] == "front")
    backs = sorted((s, d["task"]) for s, d in man.segments.items()
                   if d["role"].startswith("back"))
    wiring = {f: b for f, b in zip(fronts, backs)}
    with LocalCluster(man, t["cfg"], t["links"], None, static_wiring=wiring,
                      priors=t["tasks"]) as cl:
        cl.coord.max_tokens = 4
        u = wiring[fronts[0]][1]
        recs = [cl.coord.submit(f"z{i}", t["sample"](t["corpus"], u, 12, seed=8200 + i),
                                true_task=u, task=u) for i in range(2)]
        for r in recs:
            assert r.done.wait(timeout=120)
            assert r.stop_reason == "max_tokens"
        time.sleep(1.5)                    # 让在途的那一步走完
        a1 = _active(cl)
        time.sleep(1.0)
        a2 = _active(cl)
        assert not cl.coord.errors, cl.coord.errors[:2]
    assert sum(a1.values()) == 0 and sum(a2.values()) == 0, \
        f"请求结束后节点仍持有 KV（环还在转）：{a1} → {a2}"


def test_release_racing_an_in_flight_step_leaves_no_kv() -> None:
    """release 落在「入口检查已过、forward 正在算」的那一步中间。

    sel_dyn 实测：跑完 N11 还持有 1 条 KV。入口的 _stale 拦不住已经开算的这一步，
    它的 forward 会把 KV 重建出来。_advance 入口必须再查一次并清掉。"""
    from p2pmoe.runtime.model import ToyMoEConfig
    from p2pmoe.runtime.node import NodeConfig, NodeServer
    from p2pmoe.runtime.wire import LinkTable

    mc = ToyMoEConfig()
    allx = list(range(mc.n_experts))
    s = NodeServer("f", host="127.0.0.1", port=0)
    s.apply_config(NodeConfig(
        node_id="f", role="front", segment="F0", layer_experts={1: allx, 2: allx},
        next_hop=None, seg_head="f", is_head=True, is_tail=True, peers={},
        links=LinkTable().to_dict(), coordinator=("127.0.0.1", 1), model={},
        static_peer="b", static_task="t"))
    sent = []
    s.pool.send = lambda dest, h, arr=None, **k: sent.append((dest, h))
    s._report = lambda m: None

    s._on_prefill({"req": "r", "ids": list(range(6))})
    assert s.model.active_requests() == ["r"]

    # 模拟竞态：decode 这一步已经过了入口检查、算完了（KV 在），release 才到
    from p2pmoe.runtime.model import embed_tokens
    x = embed_tokens(s.mcfg, [3])
    y, st = s._forward(x, "r", decode=True)
    s._dispatch({"type": "release", "req": "r"}, None)
    s.model._kv.setdefault("r", {})          # forward 在 release 之后重建的 KV
    sent.clear()
    s._advance("r", "decode", y, st, "f")
    assert sent == [], "已释放的请求不该再往后段送"
    assert s.model.active_requests() == [], "竞态重建的 KV 没被清掉"

    # 后续在途消息直接在入口被丢
    s._dispatch({"type": "loop", "req": "r", "token": 4}, None)
    assert s.model.active_requests() == [] and sent == []
    assert s.n_dropped_stale >= 2

    # 同名新请求（下一轮 measure 又从 req0 起）照常进来
    s._dispatch({"type": "prefill", "req": "r", "ids": [1, 2, 3]}, None)
    assert s.model.active_requests() == ["r"] and sent
