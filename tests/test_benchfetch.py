"""给不在清单里的节点拉几层（fetch --layers / deploy_15.sh benchfetch），再用 disk 模式量真权重。"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from p2pmoe.deploy import fetch as F

pytest.importorskip("torch")
pytest.importorskip("safetensors")

from test_fetch import upstream  # noqa: F401,E402

OPTS = {"ctx": [8], "prefill": [8], "decode_steps": 2, "warmup": 0, "repeats": 1}


def _layers(d, rng, out):
    return F.main(["--src", str(d), "--layers", rng, "--node", "T1", "--out", str(out)])


def test_layers_pulls_every_expert_of_those_layers(upstream, tmp_path):  # noqa: F811
    d, cfg = upstream
    E = cfg["num_experts"]
    assert _layers(d, "3-5", tmp_path / "a") == 0
    have = F.local_checkpoint_keys(tmp_path / "a")
    want = F.keys_for_plan(F.KeyPlan({l: list(range(E)) for l in (3, 4, 5)}), config=cfg)
    assert have == want
    assert "model.embed_tokens.weight" not in have and "lm_head.weight" not in have
    assert _layers(d, "1-2", tmp_path / "b") == 0
    assert "model.embed_tokens.weight" in F.local_checkpoint_keys(tmp_path / "b")
    assert _layers(d, "8-8", tmp_path / "c") == 0
    assert "lm_head.weight" in F.local_checkpoint_keys(tmp_path / "c")
    assert _layers(d, "0-3", tmp_path / "x") == 2                 # 越界
    assert _layers(d, "abc", tmp_path / "x") == 2


def test_disk_layout_reads_back_what_is_on_disk(upstream, tmp_path):  # noqa: F811
    from p2pmoe.runtime.loader import disk_layout

    d, cfg = upstream
    assert _layers(d, "3-5", tmp_path / "a") == 0
    le, emb, head = disk_layout(F.local_checkpoint_keys(tmp_path / "a"), cfg)
    assert sorted(le) == [3, 4, 5] and all(v == list(range(cfg["num_experts"])) for v in le.values())
    assert not emb and not head


def test_disk_mode_benches_the_layers_on_disk_then_frees_them(upstream, tmp_path):  # noqa: F811
    from p2pmoe.runtime.node import NodeServer

    d, cfg = upstream
    w = tmp_path / "w"
    assert _layers(d, "3-5", w) == 0
    s = NodeServer("T1", host="127.0.0.1", port=0)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    try:
        r = s.layer_bench({"mode": "disk", "model_dir": str(w), "device": "cpu", **OPTS})
        assert r["ok"], r
        res = r["result"]
        assert res["setup"]["source"] == "disk" and res["model"]["arch"] == "qwen3_moe"
        v = res["variants"]["all"]
        assert v["layers"] == [3, 5] and v["resident"] == cfg["num_experts"]
        assert v["decode"]["8"]["median"] > 0
        assert s.model is None, "量完要释放，不能留在 agent 上"
        r = s.layer_bench({"mode": "disk", "model_dir": str(tmp_path / "nothing"), "device": "cpu"})
        assert not r["ok"] and "config.json" in r["why"]
    finally:
        s._stop.set()


def test_launch_passes_layers_instead_of_plan(monkeypatch, tmp_path):
    from p2pmoe.deploy import launch as L

    hosts = tmp_path / "h.txt"
    hosts.write_text("N13  10.0.0.13:9101\nN14  10.0.0.14:9101\n", encoding="utf-8")
    out: list[str] = []
    monkeypatch.setattr("builtins.print", lambda *a, **k: out.append(" ".join(map(str, a))))
    assert L.main(["fetch", "--hosts", str(hosts), "--layers", "2-9", "--repo", "R",
                   "--out", "/w", "--workdir", "/c", "--dry-run"]) == 0
    lines = [x for x in out if "p2pmoe.deploy.fetch" in x]
    assert len(lines) == 2 and all("--layers 2-9" in x and "--plan" not in x for x in lines)
