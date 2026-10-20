"""公共带的右端取 min(w_lo + W, w_cap)：W 比 w_cap 大时也要能找到带。

回归：计算项按逐层实测放大后，W 上限 η·median T 从 3.5ms 变成约 19ms，而 w_cap 只看
网络一跳（约 4.6ms）。旧写法「w_lo + W > w_cap 就停」一个窗都不试，人口恒为 0，
整份规划失败。
"""

from p2pmoe.planner.common_band import ExitProbe, common_band


def _probes(ivs):
    return {t: ExitProbe(exit_id=t, p50={"h1": lo, "h2": hi}, p95={"h1": lo, "h2": hi})
            for t, (lo, hi) in ivs.items()}


def test_band_wider_than_w_cap_still_finds_exits():
    pr = _probes({"a": (0.1, 3.0), "b": (1.0, 4.0), "c": (5.0, 6.0)})
    r = common_band(pr, ["h1", "h2"], W=10.0, w_cap=4.6)
    assert r.exits == ["a", "b"] and r.w_hi <= 4.6 + 1e-9


def test_exits_above_w_cap_never_enter():
    pr = _probes({"a": (5.0, 6.0)})
    assert common_band(pr, ["h1", "h2"], W=10.0, w_cap=4.6).population == 0


def test_narrow_band_unchanged():
    pr = _probes({"a": (0.1, 1.0), "b": (0.5, 1.4), "c": (2.0, 2.5)})
    r = common_band(pr, ["h1", "h2"], W=1.5, w_cap=10.0)
    assert r.exits == ["a", "b"] and r.w_lo == 0.1 and abs(r.w_hi - 1.6) < 1e-9
