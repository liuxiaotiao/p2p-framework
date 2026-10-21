"""共享后段池（`back:any`）与手写分配 → 清单（examples/manual_plan.py）。

完整模型（不裁专家）的部署里每条后段都能服务任何 task。清单把这些后段标成 `back:any`，
识别出来的 task 没有专属后段时就派到这个池；有专属后段的照旧走自己的池。
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _layout(path: Path, chans) -> Path:
    path.write_text(json.dumps({"channels": [
        {"task": t, "front": [{"node": f, "layers": [1, l0]}],
         "back": [{"node": n, "layers": list(r)} for n, r in back]}
        for t, f, l0, back in chans]}))
    return path


def test_manual_plan_cli_full_model(tmp_path, capsys):
    import importlib.util

    spec = importlib.util.spec_from_file_location("manual_plan", ROOT / "examples" / "manual_plan.py")
    mp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mp)
    lay = _layout(tmp_path / "l.json", [
        ("any", "N01", 8, [("N05", (9, 28)), ("N06", (29, 48))]),
        ("any", "N11", 8, [("N08", (9, 28)), ("N04", (29, 48))])])
    out = tmp_path / "p.json"
    lb = ROOT / "qwen3-30b" / "layer_bench_wash1.json"
    args = [str(lay), "--out", str(out)]
    if lb.exists():
        args += ["--layer-times", str(lb), "--topo", str(ROOT / "task3" / "topo_fabric_mbps.json")]
    rc = mp.main(args)
    text = capsys.readouterr().out
    man = json.loads(out.read_text())
    assert man["l0"] == 8
    roles = {n["node"]: n["role"] for n in man["nodes"]}
    assert roles["N01"] == "front" and roles["N06"] == "back:any"
    assert all(len(l["experts"]) == 128 for n in man["nodes"] for l in n["layers"])
    if lb.exists():
        # 20 层全专家 ≈ 24.9GB，单台 A30 留 2GB KV 后装不下 → 报出来、返回 1
        assert rc == 1 and "✗ 装不下" in text and "每 token 时延估算" in text


def test_manual_plan_rejects_broken_layout(tmp_path, capsys):
    import importlib.util

    spec = importlib.util.spec_from_file_location("manual_plan", ROOT / "examples" / "manual_plan.py")
    mp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mp)
    lay = _layout(tmp_path / "l.json", [("any", "N01", 8, [("N05", (9, 30)), ("N06", (32, 48))])])
    with pytest.raises(ValueError, match="层区间断了"):
        mp.main([str(lay), "--out", str(tmp_path / "p.json")])


def test_repo_full_2ch_plan_matches_layout(tmp_path, capsys):
    import importlib.util

    want = ROOT / "qwen3-30b" / "plan_full_2ch.json"
    lay = ROOT / "qwen3-30b" / "layout_full_2ch.json"
    if not (want.exists() and lay.exists()):
        pytest.skip("缺 qwen3-30b/plan_full_2ch.json")
    spec = importlib.util.spec_from_file_location("manual_plan", ROOT / "examples" / "manual_plan.py")
    mp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mp)
    out = tmp_path / "p.json"
    assert mp.main([str(lay), "--out", str(out)]) == 0
    assert out.read_bytes() == want.read_bytes()
    man = json.loads(want.read_text())
    assert {n["node"] for n in man["nodes"]} == {
        "N01", "N05", "N03", "N07", "N02", "N06", "N11", "N08", "N09", "N13", "N04", "N10"}


@pytest.mark.slow
def test_closed_loop_on_shared_pool(tmp_path) -> None:
    """手写的完整模型清单（两条 `back:any`）走连续性测试：三个 task 都派到共享池。"""
    from test_selector import CASES, _tiny_qwen3

    from p2pmoe.deploy.control import main
    from p2pmoe.deploy.manual import ManualSpec, build_manual_manifest
    from p2pmoe.planner.hf_config import model_spec_from_hf
    from p2pmoe.runtime.node import NodeServer

    ck = tmp_path / "ckpt"
    _tiny_qwen3(ck)
    cfg = json.loads((ck / "config.json").read_text())
    model, _ = model_spec_from_hf(cfg, name="tiny", ctx_max=64)
    srv = [NodeServer(f"v{i}", host="127.0.0.1", port=0) for i in range(6)]
    for s in srv:
        threading.Thread(target=s.serve_forever, daemon=True).start()
    time.sleep(0.3)
    lay = _layout(tmp_path / "l.json", [
        ("any", "v0", 2, [("v1", (3, 4)), ("v2", (5, 6))]),
        ("any", "v3", 2, [("v4", (3, 4)), ("v5", (5, 6))])])
    man, _ = build_manual_manifest(ManualSpec.from_file(lay, n_layers=6), model, model_name="tiny")
    plan = tmp_path / "plan.json"
    plan.write_text(man.to_json())

    rng = np.random.default_rng(0)
    sel = tmp_path / "sel_params.npz"
    np.savez(sel, mean=rng.random(16), scale=rng.random(16) + .5,
             coef=rng.normal(size=(3, 16)), intercept=np.zeros(3),
             classes=np.array(["gsm8k", "mbpp", "no_robots"]),
             n_layers=np.array(2), n_experts=np.array(8))
    cases = tmp_path / "cases.jsonl"
    rows = []
    for i in range(2):
        rows += [{"id": f"gsm8k:{i}", "task": "gsm8k",
                  "segments": [CASES[0][0], [f"Tom has {i + 3} apples. How many?", False]]},
                 {"id": f"mbpp:{i}", "task": "mbpp", "text": f"Write a function f{i}."},
                 {"id": f"no_robots:{i}", "task": "no_robots", "text": f"Write poem {i}."}]
    cases.write_text("\n".join(json.dumps(x) for x in rows))
    try:
        res = tmp_path / "res.json"
        assert main(["--agents", ",".join(f"{s.me}=127.0.0.1:{s.port}" for s in srv),
                     "--advertise", "127.0.0.1", "--model-dir", str(ck), "--device", "cpu",
                     "--chat", "--tasks", "gsm8k=1,mbpp=1,no_robots=1", "--requests", "0",
                     "--tokens", "3", "--skip-model-check", "--lr-model", str(sel),
                     "--prompts-file", str(cases), "--save-results", str(res), "--once",
                     "--closed-loop", "--usage-interval", "0.2", "--warmup", "1",
                     "--load-plan", str(plan)]) == 0
    finally:
        for s in srv:
            s._stop.set()
    out = json.loads(res.read_text())
    rq, run = out["requests"], out["run"]
    assert run["n_requests"] == 6 and run["n_timeout"] == 0 and run["inflight"] == 2
    assert {r["back"] for r in rq} <= {"Bany0", "Bany1"} and all(r["task"] == "any" for r in rq)
    assert {r["true_task"] for r in rq} == {"gsm8k", "mbpp", "no_robots"}
    assert all(r["n_generated"] >= 1 for r in rq)
