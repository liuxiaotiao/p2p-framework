"""连续性测试（闭环）：共享池、task 轮转、在途数恒定、指标齐全。"""

from __future__ import annotations

import json
import threading
import time

import numpy as np
import pytest

from p2pmoe.deploy.continuous import interleave, run_metrics, tpot_ms


def test_interleave_rotates_tasks_evenly():
    cases = [("a", 1), ("a", 2), ("a", 3), ("b", 1), ("b", 2), ("c", 1)]
    assert [c[0] for c in interleave(cases)] == ["a", "b", "c", "a", "b", "a"]
    assert [c for c in interleave(cases) if c[0] == "a"] == [("a", 1), ("a", 2), ("a", 3)]


def test_metrics_definitions():
    rows = [
        {"total_ms": 1000, "prefill_ms": 100, "n_generated": 10, "n_prompt": 5,
         "true_task": "x", "stop_reason": "eos", "queue_ms": 0, "queue_back_ms": 50},
        {"total_ms": 3000, "prefill_ms": 300, "n_generated": 28, "n_prompt": 7,
         "true_task": "y", "stop_reason": "max_tokens", "queue_ms": 0, "queue_back_ms": 0},
    ]
    assert tpot_ms(rows[0]) == pytest.approx(900 / 9)
    m = run_metrics(rows, wall_s=4.0, inflight=2)
    assert m["latency_ms"]["mean"] == 2000 and m["ttft_ms"]["mean"] == 200
    assert m["throughput"]["output_tok_per_s"] == pytest.approx(38 / 4)
    assert m["throughput"]["req_per_min"] == pytest.approx(30)
    assert m["latency_ms"]["p99"] == pytest.approx(np.percentile([1000, 3000], 99))
    assert set(m["by_task"]) == {"x", "y"}
    assert m["stop_reasons"] == {"eos": 1, "max_tokens": 1}


@pytest.mark.slow
def test_closed_loop_end_to_end(tmp_path) -> None:
    from test_selector import CASES, _tiny_qwen3

    from p2pmoe.deploy.control import main
    from p2pmoe.runtime.node import NodeServer

    ck = tmp_path / "ckpt"
    _tiny_qwen3(ck)
    rng = np.random.default_rng(0)
    sel = tmp_path / "sel_params.npz"
    np.savez(sel, mean=rng.random(16), scale=rng.random(16) + .5,
             coef=rng.normal(size=(3, 16)), intercept=np.zeros(3),
             classes=np.array(["gsm8k", "mbpp", "no_robots"]),
             n_layers=np.array(2), n_experts=np.array(8))
    cases = tmp_path / "cases.jsonl"
    rows = []
    for i in range(3):
        rows += [{"id": f"gsm8k:{i}", "task": "gsm8k",
                  "segments": [CASES[0][0], [f"Tom has {i + 3} apples. How many?", False]]},
                 {"id": f"mbpp:{i}", "task": "mbpp", "text": f"Write a function f{i}."},
                 {"id": f"no_robots:{i}", "task": "no_robots", "text": f"Write poem {i}."}]
    rows.sort(key=lambda r: r["task"])               # 文件里按 task 分组 —— 要靠 interleave 打散
    cases.write_text("\n".join(json.dumps(x) for x in rows))
    srv = [NodeServer(f"v{i}", host="127.0.0.1", port=0) for i in range(13)]
    for s in srv:
        threading.Thread(target=s.serve_forever, daemon=True).start()
    time.sleep(0.3)
    try:
        res = tmp_path / "res.json"
        args = ["--agents", ",".join(f"{s.me}=127.0.0.1:{s.port}" for s in srv),
                "--advertise", "127.0.0.1", "--model-dir", str(ck), "--device", "cpu",
                "--chat", "--tasks", "gsm8k=1,mbpp=1", "--requests", "0", "--tokens", "3",
                "--k-probe", "2", "--skip-model-check", "--lr-model", str(sel),
                "--prompts-file", str(cases), "--save-results", str(res), "--once",
                "--closed-loop", "--inflight", "2", "--usage-interval", "0.2",
                "--warmup", "1"]
        assert main(args) == 0
    finally:
        for s in srv:
            s._stop.set()
    out = json.loads(res.read_text())
    rq, run = out["requests"], out["run"]
    assert run["n_requests"] == 9 and run["n_timeout"] == 0 and run["inflight"] == 2
    # 池按 task 轮转
    order = sorted(rq, key=lambda r: int(r["req"][3:]))
    assert [r["true_task"] for r in order[:3]] == ["gsm8k", "mbpp", "no_robots"]
    # 在途数从不超过 2；第 k 条（k ≥ 2）一定在某条完成之后才提交
    ev = sorted([(r["t_submit_s"], 1) for r in rq] + [(r["t_done_s"], -1) for r in rq],
                key=lambda e: (e[0], e[1]))
    live = peak = 0
    for _, d in ev:
        live += d
        peak = max(peak, live)
    assert peak <= 2
    done_t = sorted(r["t_done_s"] for r in rq)
    for k, r in enumerate(order[2:]):
        assert r["t_submit_s"] >= done_t[k] - 1e-3
    for key in ("latency_ms", "ttft_ms", "tpot_ms"):
        assert {"mean", "p50", "p99"} <= set(run[key])
    assert run["throughput"]["output_tok_per_s"] > 0
    assert set(run["by_task"]) == {"gsm8k", "mbpp", "no_robots"}
    assert len(run["nodes"]) == 13
    used = [v for v in run["nodes"].values() if v.get("compute_util", 0) > 0]
    assert used, "资源采样没拿到任何节点的计算时间"
    assert all(v["n_samples"] >= 2 for v in run["nodes"].values())
    _report_renders(res)


def _report_renders(res) -> None:
    """deploy_15.sh report 的 Python 段能把连续测试的结果画出来。"""
    import subprocess
    import sys
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "deploy_15.sh").read_text(encoding="utf-8")
    body = src.split("cmd_report() {", 1)[1].split("<<'PYEOF'\n", 1)[1].split("\nPYEOF", 1)[0]
    p = subprocess.run([sys.executable, "-c", body, str(res)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-2000:]
    for want in ("连续性测试", "生成时延 ms", "TTFT ms", "TPOT ms/token", "吞吐", "资源占用"):
        assert want in p.stdout, (want, p.stdout[-1500:])


def test_gpu_snapshot_degrades_gracefully_without_nvml():
    from p2pmoe.runtime import gpu_stats
    from p2pmoe.runtime.weights import gpu_usage
    v = gpu_stats.snapshot()               # 沙箱/CPU 节点没有驱动 → None，不抛
    assert v is None or "mem_total_mb" in v
    assert "smi" in gpu_usage()
