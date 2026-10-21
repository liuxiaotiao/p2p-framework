"""连续性测试结果（--save-results 落的 JSON）的离线分析。

    python3 examples/analyze_run.py results/run2.json                 # 全部
    python3 examples/analyze_run.py results/run2.json latency ttft    # 只看某几节
    python3 examples/analyze_run.py results/run2.json --csv results/run2_latency.csv

各节：
  latency     端到端时延：含排队 vs 去排队，TTFT、TPOT、单请求 token/s（总体 + 分 task，Markdown 表）
  ttft        TTFT：原始 / 去排队 / 排队本身（总体 + 分 task）
  throughput  吞吐：整体 / 稳态（在途数保持满）/ 收尾，以及每 5 分钟一段
  slowest     TTFT 最高的几条：链路、TTFT 拆分、排队时占着那条后段的是谁
  accuracy    选择器：按 task 的正确率、混淆、置信不到 commit 区的
  rebind      换绑 / 通道二报警，以及「后段排队 > TTFT」这类异常
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import sys

import numpy as np

SECTIONS = ("latency", "ttft", "throughput", "slowest", "accuracy", "rebind")


def queue(r):
    return r.get("queue_ms", 0.0) + r.get("queue_back_ms", 0.0)


def tpot(r):
    n = r["n_generated"]
    return (r["total_ms"] - r["prefill_ms"]) / (n - 1) if n > 1 else float("nan")


def label(r):
    raw = r.get("raw_scores") or (r.get("shadow") or {}).get("raw")
    return str(max(raw, key=raw.get)).split("#")[0] if raw else r.get("task")


def tok_s(n, ms):
    return n / ms * 1000 if ms and ms > 0 else float("nan")


def groups(rs):
    return [("全部", rs)] + [(u, [r for r in rs if r["true_task"] == u])
                             for u in sorted({r["true_task"] for r in rs})]


def table(rows, unit, fmt="{:.2f}"):
    print("| 指标 | mean | std | p50 | p90 | p99 | max |")
    print("|---|---|---|---|---|---|---|")
    for name, xs in rows:
        a = np.asarray(xs, float)
        a = a[~np.isnan(a)] / unit
        if not len(a):
            print(f"| {name} | — | — | — | — | — | — |")
            continue
        print(f"| {name} | " + " | ".join(fmt.format(v) for v in (
            a.mean(), a.std(ddof=1) if len(a) > 1 else 0.0,
            np.percentile(a, 50), np.percentile(a, 90),
            np.percentile(a, 99), a.max())) + " |")


# --------------------------------------------------------------------------- #
def sec_latency(rs, _d):
    print("## 端到端时延：含排队 vs 去排队（秒；TPOT 毫秒；std 为样本标准差）")
    print("单请求 token/s：每条请求的生成 token ÷ 用时，再对请求取均值 / 标准差 —— "
          "含排队从提交算起，去排队从拿到前后段算起，解码 = 1000 ÷ TPOT")
    for name, s in groups(rs):
        share = sum(queue(r) for r in s) / max(sum(r["total_ms"] for r in s), 1e-9)
        print(f"\n### {name}（{len(s)} 条，排队占时延 {share:.1%}，平均生成 "
              f"{np.mean([r['n_generated'] for r in s]):.0f} token）")
        table([("E2E（含排队）", [r["total_ms"] for r in s]),
               ("E2E（去排队）", [r["total_ms"] - queue(r) for r in s]),
               ("排队", [queue(r) for r in s]),
               ("TTFT（含排队）", [r["prefill_ms"] for r in s]),
               ("TTFT（去排队）", [r["prefill_ms"] - queue(r) for r in s])], 1000)
        table([("TPOT ms/token", [tpot(r) for r in s])], 1, "{:.1f}")
        table([("单请求 token/s（含排队）", [tok_s(r["n_generated"], r["total_ms"]) for r in s]),
               ("单请求 token/s（去排队）",
                [tok_s(r["n_generated"], r["total_ms"] - queue(r)) for r in s]),
               ("解码 token/s（1000/TPOT）", [tok_s(1, tpot(r)) for r in s])], 1)


def sec_ttft(rs, _d):
    print("## TTFT（秒）")
    for name, s in groups(rs):
        print(f"\n### {name}（{len(s)} 条）")
        table([("TTFT（原始）", [r["prefill_ms"] for r in s]),
               ("TTFT（去排队）", [r["prefill_ms"] - queue(r) for r in s]),
               ("排队", [queue(r) for r in s])], 1000)


def sec_throughput(rs, d):
    print("## 吞吐")
    if not all("t_submit_s" in r for r in rs):
        print("  （结果里没有 t_submit_s —— 不是 --closed-loop 跑的，算不了时间线）")
        return
    wall = d.get("run", {}).get("wall_s") or max(r["t_done_s"] for r in rs)
    t_last = max(r["t_submit_s"] for r in rs)          # 池子放空：之后在途数开始下降
    ts = []
    for r in rs:
        t = r["t_submit_s"] + r["prefill_ms"] / 1000
        ts.append(t)
        for dt in r.get("token_ms", []):
            t += dt / 1000
            ts.append(t)
    ts = np.asarray(ts)
    win = ts[ts <= t_last]
    print(f"  整体    {len(ts)} token / {wall:.0f}s = {len(ts) / wall:.2f} token/s，"
          f"{len(rs) / wall * 60:.2f} 请求/分钟")
    if t_last > 0:
        print(f"  稳态    0 → {t_last:.0f}s（最后一条提交）：{len(win) / t_last:.2f} token/s")
    if wall > t_last:
        print(f"  收尾    {t_last:.0f} → {wall:.0f}s：{(len(ts) - len(win)) / (wall - t_last):.2f} token/s")
    print("  每 5 分钟：")
    for a in range(0, int(np.ceil(wall)), 300):
        n = ((ts >= a) & (ts < a + 300)).sum()
        span = min(300, wall - a)
        print(f"    {a:>5}–{a + 300:<5}s  {n / span:6.2f} token/s")


def sec_slowest(rs, _d, k=5):
    print(f"## TTFT 最高的 {k} 条")
    order = sorted(rs, key=lambda r: r["prefill_ms"])
    q = 0.99 * (len(rs) - 1)
    p99 = np.percentile([r["prefill_ms"] for r in rs], 99)
    print(f"  TTFT p99 = {p99 / 1000:.1f}s：第 {int(q) + 1} 与第 {int(q) + 2} 名（从小到大）插值")

    def chain(r):
        ns = sorted(r.get("nodes", []),
                    key=lambda n: int((re.match(r"\d+", str(n["layers"])) or [0])[0]))
        return " → ".join(f"{n['node']}[{n['segment']} 层{n['layers']}]" for n in ns)

    for r in order[-k:][::-1]:
        comp = r["prefill_ms"] - queue(r)
        print(f"\n  {r['req']:<7} {r.get('case_id', '?'):<16} 真实 {r['true_task']:<9} "
              f"路由 {r['task']:<9} c={r.get('confidence', 0):.2f}")
        print(f"    TTFT {r['prefill_ms'] / 1000:.1f}s = 前段排队 {r['queue_ms'] / 1000:.1f}"
              f" + 后段排队 {r['queue_back_ms'] / 1000:.1f} + 计算/传输 {comp / 1000:.1f}"
              f"（prompt {r['n_prompt']}，生成 {r['n_generated']}，总时延 {r['total_ms'] / 1000:.0f}s）")
        print(f"    链路 {r['front']} × {r['back']}：{chain(r)}")
        if "t_submit_s" in r:
            w0, w1 = r["t_submit_s"], r["t_submit_s"] + r["prefill_ms"] / 1000
            busy = [o for o in rs if o is not r and o["back"] == r["back"]
                    and o["t_submit_s"] < w1 and o["t_done_s"] > w0]
            if busy:
                print(f"    排队期间占着 {r['back']} 的：" + "，".join(
                    f"{o['req']}({o['t_submit_s']:.0f}–{o['t_done_s']:.0f}s，生成 {o['n_generated']})"
                    for o in sorted(busy, key=lambda o: o["t_submit_s"])))


def sec_accuracy(rs, _d):
    print("## 选择器")
    by = collections.defaultdict(lambda: [0, 0])
    conf = collections.Counter()
    for r in rs:
        by[r["true_task"]][0] += label(r) == r["true_task"]
        by[r["true_task"]][1] += 1
        conf[(r["true_task"], label(r))] += 1
    ok, n = sum(v[0] for v in by.values()), sum(v[1] for v in by.values())
    print(f"  总计 {ok}/{n} = {ok / n:.1%}（选择器自己判的类）")
    for u, (a, b) in sorted(by.items()):
        print(f"    {u:<10} {a}/{b} = {a / b:.1%}")
    ts = sorted({t for t, _ in conf} | {p for _, p in conf})
    print("  混淆（行=真实，列=判成）：")
    print("  " + " " * 12 + "".join(f"{t[:10]:>11}" for t in ts))
    for t in ts:
        print(f"  {t[:10]:<12}" + "".join(f"{conf[(t, p)]:>11}" for p in ts))
    low = [r for r in rs if r.get("zone") not in (None, "commit")]
    print(f"  不在 commit 区（置信 < τ_hi）的 {len(low)} 条" + (
        "：" + "，".join(f"{r['req']}({r['true_task']}→{r['task']} c={r.get('confidence', 0):.2f})"
                        for r in low[:10]) if low else ""))


def sec_rebind(rs, _d):
    print("## 换绑 / 通道二")
    print("  换绑：", {u: sum(r.get("rebinds", 0) for r in s) for u, s in groups(rs)[1:]},
          "  报警：", {u: sum(r.get("alarms", 0) for r in s) for u, s in groups(rs)[1:]})
    bad = [r for r in rs if r["queue_back_ms"] > r["prefill_ms"]]
    print(f"  后段排队 > TTFT 的 {len(bad)} 条（关了换绑、且用新版协调器跑的结果应为 0）")
    for r in bad[:10]:
        print(f"    {r['req']:<7} {r['true_task']:<9} 换绑 {r.get('rebinds', 0)}  最终后段 {r['back']}"
              f"  生成 {r['n_generated']}  «{r.get('text', '')[-60:]!r}»")
    long_ = [r for r in rs if r.get("stop_reason") == "max_tokens"]
    print(f"  写满 max_tokens 的 {len(long_)} 条：" + "，".join(
        f"{r['req']}({r['true_task']}，换绑 {r.get('rebinds', 0)})" for r in long_[:15]))


def write_csv(rs, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["req", "case_id", "task", "routed", "front", "back", "n_prompt",
                    "n_generated", "stop_reason", "e2e_ms", "e2e_noqueue_ms", "queue_front_ms",
                    "queue_back_ms", "ttft_ms", "ttft_noqueue_ms", "tpot_ms",
                    "tok_per_s", "tok_per_s_noqueue", "rebinds",
                    "t_submit_s", "t_done_s"])
        for r in sorted(rs, key=lambda r: int(re.sub(r"\D", "", r["req"]) or 0)):
            w.writerow([r["req"], r.get("case_id"), r["true_task"], r["task"], r["front"],
                        r["back"], r["n_prompt"], r["n_generated"], r.get("stop_reason"),
                        round(r["total_ms"], 1), round(r["total_ms"] - queue(r), 1),
                        r["queue_ms"], r["queue_back_ms"], round(r["prefill_ms"], 1),
                        round(r["prefill_ms"] - queue(r), 1), round(tpot(r), 2),
                        round(tok_s(r["n_generated"], r["total_ms"]), 3),
                        round(tok_s(r["n_generated"], r["total_ms"] - queue(r)), 3),
                        r.get("rebinds", 0), r.get("t_submit_s"), r.get("t_done_s")])
    print(f"\n逐条数据 → {path}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("results")
    ap.add_argument("sections", nargs="*", default=[],
                    help="要看的节（默认全部）：" + " ".join(SECTIONS))
    ap.add_argument("--csv", default=None, help="逐条导出（画 CDF / 散点用）")
    a = ap.parse_args(argv)
    bad = [x for x in a.sections if x not in SECTIONS]
    if bad:
        ap.error(f"不认识的节 {bad}，可选 {SECTIONS}")
    d = json.load(open(a.results, encoding="utf-8"))
    rs = d["requests"]
    print(f"{a.results}：{len(rs)} 条，模式 {d.get('mode')}，miss={d.get('miss_policy')}")
    fns = {"latency": sec_latency, "ttft": sec_ttft, "throughput": sec_throughput,
           "slowest": sec_slowest, "accuracy": sec_accuracy, "rebind": sec_rebind}
    for s in (a.sections or SECTIONS):
        print()
        fns[s](rs, d)
    if a.csv:
        write_csv(rs, a.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
