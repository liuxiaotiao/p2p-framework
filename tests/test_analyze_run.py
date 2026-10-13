"""examples/analyze_run.py 能在连续测试的结果文件上跑通每一节。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "analyze_run", Path(__file__).resolve().parent.parent / "examples" / "analyze_run.py")
ar = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ar)


def _fake(tmp: Path) -> Path:
    rs = []
    for i in range(12):
        u = ("gsm8k", "mbpp", "no_robots")[i % 3]
        qb = 5000.0 * (u == "no_robots")
        rs.append({
            "req": f"req{i}", "case_id": f"{u}:{i}", "true_task": u, "task": u,
            "raw_scores": {u: 0.99, "x": 0.01}, "zone": "commit", "confidence": 0.99,
            "front": f"F{i % 5}", "back": f"B{u}0", "n_prompt": 50, "n_generated": 20,
            "stop_reason": "eos", "total_ms": 4000 + qb, "prefill_ms": 1200 + qb,
            "queue_ms": 0.0, "queue_back_ms": qb, "token_ms": [140.0] * 19,
            "t_submit_s": i * 2.0, "t_done_s": i * 2.0 + (4000 + qb) / 1000,
            "nodes": [{"node": "N01", "segment": f"F{i % 5}", "layers": "1–8"},
                      {"node": "N03", "segment": f"B{u}0", "layers": "9–48"}],
            "rebinds": 0, "alarms": 0, "text": "ok"})
    p = tmp / "run.json"
    p.write_text(json.dumps({"mode": "dynamic", "miss_policy": "local_topk",
                             "requests": rs, "run": {"wall_s": 40.0}}))
    return p


def test_every_section_runs(tmp_path, capsys):
    p = _fake(tmp_path)
    assert ar.main([str(p), "--csv", str(tmp_path / "o.csv")]) == 0
    out = capsys.readouterr().out
    for want in ("E2E（去排队）", "TTFT（去排队）", "稳态", "TTFT 最高", "总计 12/12",
                 "后段排队 > TTFT 的 0 条"):
        assert want in out, want
    assert (tmp_path / "o.csv").read_text().count("\n") == 13


def test_section_filter(tmp_path, capsys):
    assert ar.main([str(_fake(tmp_path)), "ttft"]) == 0
    out = capsys.readouterr().out
    assert "TTFT（原始）" in out and "选择器" not in out
