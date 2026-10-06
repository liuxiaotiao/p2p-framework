"""时延构成那张表必须诚实：不把别的东西叫作「网络」。

真机上量出来的两处失真，都让「网络」虚高：

1. **后段排队被并进了残差。** `queue_ms` 只记等前段的时间，等后段的
   `wait_back_ms` 没进去，于是它落进 `总时延 − 计算 − 排队` 里，被打印成
   「网络+协议+调度」。动态那轮 req16 等了 29 秒后段，报表里排队显示 0。

2. **埋点缺席节点的计算也被并进了残差。** `control.py` 从来没调用过
   `wait_trace`（只有 run.py 调），节点忙着算下一条时埋点还在路上，总结就
   做完了。漏的偏偏是 N08 —— 一台占 48% 的计算。

两者叠加的结果：报表说「网络占 30.6%（动态）/ 51.8%（静态）」，而 `check`
量到的两两延迟中位是 3ms，一个 token 绕四跳约 12ms，对 437ms 的逐 token
是 **3% 左右**。差了一个数量级，而这张图是要进论文的。
"""

from __future__ import annotations

import pytest

from p2pmoe.runtime.timing import NodeTiming, RequestTiming


def _node(name: str, ms: float) -> NodeTiming:
    return NodeTiming(node=name, role="front", segment="F0", layers=(1,),
                      n_experts=1, compute_ms=ms, n_forward=1, bytes_out=0)


def _t(**kw) -> RequestTiming:
    base = dict(req="r0", front="F0", back="B0", n_prompt=8, n_generated=32,
                total_ms=1000.0, queue_ms=0.0, prefill_ms=100.0, decode_ms=900.0)
    base.update(kw)
    return RequestTiming(**base)


def test_back_queue_is_not_called_network() -> None:
    """核心断言：等后段的时间不能落进残差。"""
    t = _t(total_ms=1000.0, queue_ms=0.0, queue_back_ms=600.0,
           nodes=[_node("a", 300.0)])
    assert t.other_ms == pytest.approx(100.0), \
        "后段排队被算成了网络 —— 这正是真机上「网络 30.6%」的来源"


def test_the_four_parts_still_sum_to_the_total() -> None:
    """拆出一项之后，四段相加仍须恒等于总时延 —— 否则这张表就会骗人。"""
    t = _t(total_ms=1000.0, queue_ms=120.0, queue_back_ms=300.0,
           nodes=[_node("a", 400.0), _node("b", 80.0)])
    assert (t.compute_ms + t.queue_ms + t.queue_back_ms + t.other_ms
            == pytest.approx(t.total_ms))


def test_the_residual_never_goes_negative() -> None:
    """埋点之和超过总时延时（时钟抖动、重复上报）只能夹到 0，不能出负数。"""
    t = _t(total_ms=100.0, queue_ms=50.0, queue_back_ms=50.0,
           nodes=[_node("a", 80.0)])
    assert t.other_ms == 0.0


def test_a_request_with_missing_traces_is_marked_incomplete() -> None:
    """缺席节点的计算在残差里，所以这条请求的「网络」没有意义。

    标出来是为了让调用方能把它剔出均值 —— 而不是让它悄悄把结论带偏。
    """
    good = _t(nodes=[_node("a", 300.0)])
    assert good.complete is True

    bad = _t(nodes=[_node("a", 300.0)], missing_traces=["N08"])
    assert bad.complete is False


def test_a_missing_heavy_node_inflates_the_residual() -> None:
    """把真机那组数复现一遍，说明为什么不能混进均值。

    N08 一台占 48% 的计算。它的埋点丢了，那 480ms 就整个变成「网络」。
    """
    with_n08 = _t(total_ms=1000.0, nodes=[_node("a", 300.0), _node("N08", 480.0)])
    without = _t(total_ms=1000.0, nodes=[_node("a", 300.0)],
                 missing_traces=["N08"])

    assert with_n08.other_ms == pytest.approx(220.0)
    assert without.other_ms == pytest.approx(700.0)
    assert without.other_ms > with_n08.other_ms * 3
    assert without.complete is False, "至少要标出来，否则 700ms 会被当成网络"


def test_the_control_path_waits_for_telemetry() -> None:
    """`control.py` 必须在总结前等埋点。

    以前只有 `run.py` 调 `wait_trace`，而真机走的是 `control.py` —— 于是
    整条部署路径上从来没等过。埋点齐了 `wait_trace` 立刻返回，所以这行在
    正常情况下不花时间，只在真的缺席时才等。
    """
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1]
           / "p2pmoe/deploy/control.py").read_text(encoding="utf-8")
    i_wait = src.find("coord.wait_trace(")
    i_sum = src.find("summarise_request(rec, coord)")
    assert i_wait != -1, "control.py 没有等埋点"
    assert i_wait < i_sum, "等埋点必须发生在总结之前，否则等了也没用"


def test_the_report_excludes_incomplete_requests_from_the_means() -> None:
    """报表必须按 `traces_complete` 过滤，而不是闷头平均。"""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1]
           / "deploy_15.sh").read_text(encoding="utf-8")
    assert 'r.get("traces_complete", not r.get("missing_traces"))' in src, (
        "报表没有按埋点齐全与否过滤。回落到 missing_traces 是有意的 —— "
        "旧的 results/*.json 没有 traces_complete 字段，但有 missing_traces，"
        "这样过去跑的结果不用重跑集群就能拿到修正后的构成表。")
    assert 'queue_back_ms' in src, "报表没有后段排队这一列"


def test_static_mode_has_no_back_queue() -> None:
    """静态模式前后段一起拿，没有「识别完再等后段」这一段。

    以前 `wait_back_ms = t_back − _t_classified`，而静态模式从不识别，
    `_t_classified` 恒为 0 —— 减出来的是 perf_counter 的绝对值，toy 冒烟里
    「后排」一栏显示 1218224ms。真机静态报表的时延构成也被它带歪。
    """
    from p2pmoe.runtime.coordinator import RequestRecord

    rec = RequestRecord(req="s0", true_task="X")
    rec.t0 = 1000.0
    rec.t_front = rec.t_back = 1000.5          # 静态：一起拿到
    assert rec.wait_back_ms == 0.0

    dyn = RequestRecord(req="d0", true_task="X")
    dyn._t_classified, dyn.t_back = 2000.0, 2000.25
    assert dyn.wait_back_ms == pytest.approx(250.0)


def test_the_report_repairs_old_static_files() -> None:
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1]
           / "deploy_15.sh").read_text(encoding="utf-8")
    assert 'if r.get("queue_back_ms", 0) > r["total_ms"]:' in src, \
        "旧的静态结果文件里 queue_back_ms 是绝对时钟，报表要在读的时候改正"
