"""连续性测试：闭环压测 + 时延 / 吞吐 / 资源占用指标。

    请求池（一个，所有前段共享）──▶ 同时在途 N 条（默认 = 后段通道总数）
                                      │ 任意一条完成
                                      ▼
                               立刻从池里放出下一条

* **池**：测试集按 task 轮转排好（gsm8k, mbpp, no_robots, gsm8k, …），三个 task
  均匀地进入系统。哪个前段空闲就由哪个前段接 —— 前段之间没有各自的队列。
* **闭环**：「每个后段完成一个 request，再释放一个 request 出来」。在途数恒定，
  所以测到的是系统在持续满载下的稳态，而不是一阵突发。
* **指标**（每条请求都落盘，汇总在 `run` 里）：
    - 生成时延：提交 → 最后一个 token（端到端，含排队），均值 / p50 / p90 / p99
    - TTFT：提交 → 第一个 token（含排队与 prefill）
    - TPOT：decode 阶段每 token 的平均间隔
    - 吞吐：输出 token/s、请求/分钟（墙钟 = 第一条提交 → 最后一条完成）
    - 资源：各节点的计算占用（forward 时间 / 墙钟）、GPU 利用率（nvidia-smi 采样）、
      显存（nvidia-smi 整卡峰值 + torch 分配器峰值）
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from typing import Callable, Sequence

import numpy as np

log = logging.getLogger("p2pmoe.continuous")


# --------------------------------------------------------------------------- #
# 请求池的顺序
# --------------------------------------------------------------------------- #
def interleave(cases: Sequence[tuple], key: Callable = lambda c: c[0]) -> list:
    """按 task 轮转：每个 task 内部保持原顺序，task 之间交替。

    task 出现的先后决定轮转顺序。某个 task 的样本先用完，剩下的照样轮转。"""
    groups: dict = {}
    for c in cases:
        groups.setdefault(key(c), []).append(c)
    out = []
    for i in range(max((len(g) for g in groups.values()), default=0)):
        for g in groups.values():
            if i < len(g):
                out.append(g[i])
    return out


# --------------------------------------------------------------------------- #
# 闭环驱动
# --------------------------------------------------------------------------- #
def run_closed_loop(
    n: int,
    submit: Callable[[int], object],
    *,
    inflight: int,
    timeout_s: float,
    on_done: Callable[[object, int, int], None],
    on_timeout: Callable[[object], None] | None = None,
    poll_s: float = 0.02,
) -> tuple[list, list, float, float]:
    """提交 n 条请求，任何时刻最多 `inflight` 条在途；一条完成就补一条。

    `submit(i)` 返回带 `.done`（threading.Event）与 `.t0` 的记录。
    返回 (完成的记录, 超时的记录, 墙钟起点, 墙钟终点)，墙钟用 perf_counter。

    轮询而不是回调：补位的时刻晚 ≤ poll_s，但请求自己的 t0 在 submit 时才记，
    所以这点延迟不会算进任何一条请求的时延，只会让在途数短暂少一条。"""
    nxt, live, done, dead = 0, [], [], []
    t_start = time.perf_counter()
    while nxt < n or live:
        while nxt < n and len(live) < inflight:
            live.append(submit(nxt))
            nxt += 1
        time.sleep(poll_s)
        still = []
        now = time.perf_counter()
        for r in live:
            if r.done.is_set():
                done.append(r)
                on_done(r, len(done), n)
            elif now - r.t0 > timeout_s:
                dead.append(r)
                if on_timeout:
                    on_timeout(r)
            else:
                still.append(r)
        live = still
    return done, dead, t_start, time.perf_counter()


# --------------------------------------------------------------------------- #
# 资源采样
# --------------------------------------------------------------------------- #
class UsageMonitor:
    """后台线程按固定间隔问每台节点要 `usage`（累计计算时间 + GPU 快照）。"""

    def __init__(self, nodes: Sequence[str], ask: Callable[[str, bool], dict],
                 interval_s: float = 2.0):
        self.nodes = list(nodes)
        self.ask = ask
        self.interval = interval_s
        self.samples: dict[str, list[tuple[float, dict]]] = defaultdict(list)
        self.errors: dict[str, str] = {}
        self._stop = threading.Event()
        self._th: threading.Thread | None = None

    def _sample_all(self, reset_peak: bool = False) -> None:
        now = time.perf_counter()
        ths = []

        def one(nd):
            try:
                self.samples[nd].append((now, self.ask(nd, reset_peak)))
            except Exception as e:                 # 采样失败不能拖垮测试
                self.errors[nd] = f"{type(e).__name__}: {e}"[:200]
        for nd in self.nodes:
            t = threading.Thread(target=one, args=(nd,), daemon=True)
            t.start()
            ths.append(t)
        for t in ths:
            t.join(timeout=10)

    def start(self) -> None:
        self._sample_all(reset_peak=True)          # 起点：清峰值，记累计计算时间

        def loop():
            while not self._stop.wait(self.interval):
                self._sample_all()
        self._th = threading.Thread(target=loop, daemon=True)
        self._th.start()

    def stop(self) -> None:
        self._stop.set()
        if self._th:
            self._th.join(timeout=self.interval + 10)
        self._sample_all()                         # 终点

    def summary(self, t0: float, t1: float) -> dict[str, dict]:
        """只用 [t0, t1] 墙钟窗口内（含两端最近的样本）算占用。"""
        out = {}
        for nd in self.nodes:
            ss = self.samples.get(nd) or []
            if not ss:
                out[nd] = {"error": self.errors.get(nd, "没有样本")}
                continue
            first = max((s for s in ss if s[0] <= t0), key=lambda s: s[0], default=ss[0])
            last = min((s for s in ss if s[0] >= t1), key=lambda s: s[0], default=ss[-1])
            win = [s for s in ss if first[0] <= s[0] <= last[0]]
            span = max(last[0] - first[0], 1e-9)
            comp = (last[1].get("compute_ms", 0.0) - first[1].get("compute_ms", 0.0))
            smi = [s[1].get("gpu", {}).get("smi") for s in win]
            smi = [x for x in smi if x]
            tor = [s[1].get("gpu", {}).get("torch") for s in win]
            tor = [x for x in tor if x]

            def col(xs, k):
                v = [x[k] for x in xs if x.get(k) is not None]
                return v
            util = col(smi, "util_pct")
            mem = col(smi, "mem_used_mb")
            tot = col(smi, "mem_total_mb")
            pw = col(smi, "power_w")
            out[nd] = {
                "compute_util": round(comp / (span * 1000.0), 4),
                "gpu_util_mean": round(float(np.mean(util)), 1) if util else None,
                "gpu_util_p95": round(float(np.percentile(util, 95)), 1) if util else None,
                "mem_used_max_mb": max(mem) if mem else None,
                "mem_used_mean_mb": round(float(np.mean(mem)), 1) if mem else None,
                "mem_total_mb": tot[0] if tot else None,
                "torch_allocated_max_mb": max(col(tor, "max_allocated_mb"), default=None),
                "torch_reserved_max_mb": max(col(tor, "reserved_mb"), default=None),
                "power_mean_w": round(float(np.mean(pw)), 1) if pw else None,
                "n_samples": len(win),
            }
        return out


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
def _stats(xs: Sequence[float]) -> dict:
    xs = [float(x) for x in xs if x is not None]
    if not xs:
        return {"n": 0}
    a = np.asarray(xs)
    return {"n": len(xs), "mean": round(float(a.mean()), 2),
            "p50": round(float(np.percentile(a, 50)), 2),
            "p90": round(float(np.percentile(a, 90)), 2),
            "p99": round(float(np.percentile(a, 99)), 2),
            "max": round(float(a.max()), 2)}


def _label(r: dict) -> str | None:
    """选择器自己判的类（原始概率 argmax）；没有原始概率时用路由结果。"""
    raw = r.get("raw_scores") or (r.get("shadow") or {}).get("raw")
    if raw:
        return str(max(raw, key=raw.get)).split("#")[0]
    return r.get("task")


def _acc(rows: Sequence[dict]) -> dict | None:
    xs = [(_label(r) == r["true_task"]) for r in rows if r.get("true_task")]
    if not xs:
        return None
    return {"correct": int(sum(xs)), "n": len(xs), "rate": round(sum(xs) / len(xs), 4)}


def ttft_service_ms(r: dict) -> float:
    """去掉排队的 TTFT：拿到前段与后段之后，prefill + 传输 + 出首 token 的时间。"""
    return r["prefill_ms"] - r.get("queue_ms", 0.0) - r.get("queue_back_ms", 0.0)


def tpot_ms(r: dict) -> float | None:
    """decode 阶段每 token 的平均间隔：首 token 之后的时间 / 之后的 token 数。"""
    n = int(r.get("n_generated", 0))
    if n < 2:
        return None
    return (r["total_ms"] - r["prefill_ms"]) / (n - 1)


def run_metrics(rows: Sequence[dict], wall_s: float, *,
                usage: dict | None = None, inflight: int | None = None,
                n_timeout: int = 0) -> dict:
    """rows：控制面落盘的逐请求字典（total_ms、prefill_ms=TTFT、n_generated …）。"""
    out_tok = sum(int(r["n_generated"]) for r in rows)
    in_tok = sum(int(r["n_prompt"]) for r in rows)
    m = {
        "n_requests": len(rows),
        "n_timeout": n_timeout,
        "inflight": inflight,
        "wall_s": round(wall_s, 2),
        "latency_ms": _stats([r["total_ms"] for r in rows]),
        "ttft_ms": _stats([r["prefill_ms"] for r in rows]),
        "ttft_service_ms": _stats([ttft_service_ms(r) for r in rows]),
        "tpot_ms": _stats([tpot_ms(r) for r in rows]),
        "queue_front_ms": _stats([r.get("queue_ms", 0.0) for r in rows]),
        "queue_back_ms": _stats([r.get("queue_back_ms", 0.0) for r in rows]),
        "output_tokens": out_tok,
        "prompt_tokens": in_tok,
        "gen_len": _stats([r["n_generated"] for r in rows]),
        "throughput": {
            "output_tok_per_s": round(out_tok / wall_s, 3) if wall_s else None,
            "total_tok_per_s": round((out_tok + in_tok) / wall_s, 3) if wall_s else None,
            "req_per_min": round(len(rows) / wall_s * 60, 3) if wall_s else None,
        },
        "rebinds": sum(int(r.get("rebinds", 0)) for r in rows),
        "alarms": sum(int(r.get("alarms", 0)) for r in rows),
        "accuracy": _acc(rows),
        "stop_reasons": dict(sorted(
            {k: sum(1 for r in rows if r.get("stop_reason") == k)
             for k in {r.get("stop_reason") for r in rows}}.items(), key=str)),
    }
    by = defaultdict(list)
    for r in rows:
        by[r.get("true_task") or "?"].append(r)
    m["by_task"] = {u: {
        "n": len(rs),
        "latency_ms": _stats([r["total_ms"] for r in rs]),
        "ttft_ms": _stats([r["prefill_ms"] for r in rs]),
        "ttft_service_ms": _stats([ttft_service_ms(r) for r in rs]),
        "tpot_ms": _stats([tpot_ms(r) for r in rs]),
        "queue_back_ms": _stats([r.get("queue_back_ms", 0.0) for r in rs]),
        "output_tokens": sum(int(r["n_generated"]) for r in rs),
        "gen_len": _stats([r["n_generated"] for r in rs]),
        "accuracy": _acc(rs),
        "rebinds": sum(int(r.get("rebinds", 0)) for r in rs),
    } for u, rs in sorted(by.items())}
    if usage:
        m["nodes"] = usage
        cu = [v["compute_util"] for v in usage.values() if "compute_util" in v]
        gu = [v["gpu_util_mean"] for v in usage.values() if v.get("gpu_util_mean") is not None]
        mu = [v["mem_used_max_mb"] for v in usage.values() if v.get("mem_used_max_mb")]
        mt = [v["mem_total_mb"] for v in usage.values() if v.get("mem_total_mb")]
        m["resources"] = {
            "compute_util_mean": round(float(np.mean(cu)), 4) if cu else None,
            "gpu_util_mean": round(float(np.mean(gu)), 1) if gu else None,
            "mem_used_max_total_mb": round(float(sum(mu)), 1) if mu else None,
            "mem_total_mb": round(float(sum(mt)), 1) if mt else None,
        }
    return m
