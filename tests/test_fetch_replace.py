"""换方案时就地替换权重（fetch --replace / REPLACE=1 deploy_15.sh fetch）。

磁盘放不下新旧两份：每台先看已有权重是不是新清单要的那一份 —— 是就跳过，不是就
先删旧的再下新的。config / tokenizer 不删（控制机的 --model-dir 要读）。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from p2pmoe.deploy import fetch as F
from p2pmoe.deploy.manual import ManualSpec, build_manual_manifest
from p2pmoe.planner.hf_config import model_spec_from_hf

pytest.importorskip("safetensors")

from test_fetch import load_all, upstream  # noqa: F401,E402  （fixture 复用）


def _manifest(cfg, back_experts):
    spec, _ = model_spec_from_hf(cfg, name="t", ctx_max=128, dtype_bytes=4)
    layout = ManualSpec.from_dict({"channels": [{
        "front": [{"node": "n1", "layers": [1, 2]}],
        "back": [{"node": "n2", "layers": [3, 5]}, {"node": "n3", "layers": [6, 8]}],
    }], "experts": {str(l): back_experts for l in range(3, 9)}}, n_layers=8)
    return build_manual_manifest(layout, spec)[0]


def _plans(tmp_path, cfg):
    """旧方案 / 新方案：n1（前段）不变，n2、n3 的驻留专家换了。"""
    a, b = tmp_path / "old.json", tmp_path / "new.json"
    a.write_text(_manifest(cfg, [0, 1, 2]).to_json(), encoding="utf-8")
    b.write_text(_manifest(cfg, [3, 4, 5]).to_json(), encoding="utf-8")
    return a, b


def _fetch(src, plan, node, out, *extra):
    return F.main(["--src", str(src), "--plan", str(plan), "--node", node,
                   "--out", str(out), *extra])


def test_unchanged_node_is_skipped(upstream, tmp_path, caplog):  # noqa: F811
    d, cfg = upstream
    old, new = _plans(tmp_path, cfg)
    out = tmp_path / "w"
    assert _fetch(d, old, "n1", out) == 0
    st = (out / "model.safetensors").stat()
    caplog.set_level("INFO")
    assert _fetch(d, new, "n1", out, "--replace") == 0
    assert "跳过下载" in caplog.text
    st2 = (out / "model.safetensors").stat()
    assert (st.st_mtime_ns, st.st_ino) == (st2.st_mtime_ns, st2.st_ino), "一致的权重不该重写"


def test_changed_node_is_replaced(upstream, tmp_path):  # noqa: F811
    d, cfg = upstream
    old, new = _plans(tmp_path, cfg)
    out = tmp_path / "w"
    assert _fetch(d, old, "n2", out) == 0
    (out / "model.safetensors.partial").write_bytes(b"x" * 100)   # 上次没下完的残渣
    assert _fetch(d, new, "n2", out, "--replace") == 0
    man = F.DeploymentManifest.from_json(new.read_text(encoding="utf-8"))
    assert F.local_checkpoint_keys(out) == F.keys_for_node(man, "n2", config=cfg)
    assert set(load_all(out)) == F.keys_for_node(man, "n2", config=cfg)
    assert not (out / "model.safetensors.partial").exists()
    assert (out / "config.json").exists(), "config 必须保留"


def test_without_replace_old_and_new_would_coexist_on_disk(upstream, tmp_path, monkeypatch):  # noqa: F811
    """--replace 的意义：先删后下。下新权重时旧文件已不在盘上。"""
    d, cfg = upstream
    old, new = _plans(tmp_path, cfg)
    out = tmp_path / "w"
    assert _fetch(d, old, "n3", out) == 0
    seen = []
    real = F.fetch

    def spy(*a, **k):
        seen.append((out / "model.safetensors").exists())
        return real(*a, **k)

    monkeypatch.setattr(F, "fetch", spy)
    assert _fetch(d, new, "n3", out, "--replace") == 0
    assert seen == [False]


def test_truncated_file_counts_as_missing(upstream, tmp_path):  # noqa: F811
    d, cfg = upstream
    old, _ = _plans(tmp_path, cfg)
    out = tmp_path / "w"
    assert _fetch(d, old, "n2", out) == 0
    f = out / "model.safetensors"
    assert F.local_checkpoint_keys(out)
    f.write_bytes(f.read_bytes()[:-10])
    assert F.local_checkpoint_keys(out) is None
    assert _fetch(d, old, "n2", out, "--replace") == 0         # 截断的会被重下
    assert F.local_checkpoint_keys(out)


def test_dry_run_never_deletes(upstream, tmp_path):  # noqa: F811
    d, cfg = upstream
    old, new = _plans(tmp_path, cfg)
    out = tmp_path / "w"
    assert _fetch(d, old, "n2", out) == 0
    before = F.local_checkpoint_keys(out)
    assert _fetch(d, new, "n2", out, "--replace", "--dry-run") == 0
    assert F.local_checkpoint_keys(out) == before


def test_launch_passes_replace_to_every_node(monkeypatch, tmp_path):
    from p2pmoe.deploy import launch as L

    hosts = tmp_path / "h.txt"
    hosts.write_text("N01  10.0.0.1:9101\nN02  10.0.0.2:9101\n", encoding="utf-8")
    plan = tmp_path / "plan.json"
    plan.write_text('{"l0": 1, "model": {}, "segments": {}, "nodes": []}', encoding="utf-8")
    out: list[str] = []
    monkeypatch.setattr("builtins.print", lambda *a, **k: out.append(" ".join(map(str, a))))
    L.main(["fetch", "--hosts", str(hosts), "--plan", str(plan), "--repo", "R",
            "--out", "/w", "--workdir", "/c", "--replace", "--dry-run"])
    lines = [x for x in out if "p2pmoe.deploy.fetch" in x]
    assert len(lines) == 2 and all("--replace" in x for x in lines)
