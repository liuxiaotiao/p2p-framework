"""连续测试里暴露的两件事：

1. 通道二误报换绑（选择器 150/150 判对，仍换绑 11 次）。中途换绑只给新后段重做
   prompt 的 prefill，已生成 token 不在新后段的 KV 里 → 6 条退化成 2048 token 的
   复读。`--no-rebind`：只报警、不换。
2. 「后段排队」用最后一次拿到后段的时刻算，换绑会把 decode 时间也记成排队，
   出现「后段排队 > TTFT」。改为只算第一次派发前的等待。
"""

from __future__ import annotations

import time

from test_serving_loop import classify, make_coord


def miss_token(c, req, miss):
    c._on({"type": "token", "req": req, "node": "b", "segment": "B", "phase": "decode",
           "token": 1, "back_stats": {"hist": [], "ntl": 10, "miss": miss, "mass": 0.0}})


def _armed(allow: bool):
    c, pool = make_coord()
    c.alarm_factor = 2.0
    c.min_window = 3
    c.max_tokens = 50
    c.allow_rebind = allow
    r = c.submit("r0", [1], true_task="X")
    classify(c, "r0", "X")
    r.scores = {"X": 0.9, "Y": 0.1}
    for _ in range(4):
        miss_token(c, "r0", 9)            # miss 率 90% ≫ 基线 10% × 2
    return c, pool, r


def test_rebind_happens_when_allowed():
    c, pool, r = _armed(True)
    assert r.rebinds == 1 and r.task == "Y" and r.alarms >= 1
    assert r.rebind_log and r.rebind_log[0]["to"] == "BY0"


def test_no_rebind_only_counts_alarms():
    c, pool, r = _armed(False)
    assert r.rebinds == 0 and r.task == "X" and r.back.startswith("BX")
    assert r.alarms >= 1
    assert not pool.of_type("drop_kv"), "关了换绑就不该丢后段 KV"
    for _ in range(4):
        miss_token(c, "r0", 9)
    assert r.rebinds == 0 and r.alarms >= 2


def test_back_queue_counts_only_the_first_dispatch():
    c, pool, r = _armed(True)
    # 换绑发生在 decode 中途：t_back 更新了，但后段排队只算第一次派发
    assert r.t_back > r.t_back_first > 0
    assert r.wait_back_ms == (r.t_back_first - r._t_classified) * 1000
    time.sleep(0.01)
    ttft = (r.t_first - r.t0) * 1000
    assert r.wait_back_ms <= ttft, "后段排队不可能比 TTFT 还长"
