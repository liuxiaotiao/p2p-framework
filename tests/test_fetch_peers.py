"""换方案时复用集群里已有的张量：本机旧权重 → 别的节点 → 上游（fetch --reuse / --peers）。

判据与 test_fetch 一样硬：拼出来的张量必须与直接从上游拉的逐位一致；另外要证明
**真的少从上游拿了** —— 上游用 serve_weights 起，数它发出去的字节。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from p2pmoe.deploy import fetch as F
from p2pmoe.deploy.serve_weights import WeightServer

pytest.importorskip("torch")
pytest.importorskip("safetensors")

from test_fetch import load_all, upstream  # noqa: F401,E402
from test_fetch_replace import _plans  # noqa: E402


def _same(a: Path, b: Path) -> None:
    import torch

    x, y = load_all(a), load_all(b)
    assert set(x) == set(y)
    for k in x:
        assert torch.equal(x[k], y[k]), k


def _run(*argv) -> int:
    return F.main([str(a) for a in argv])


@pytest.fixture
def up(upstream):  # noqa: F811
    d, cfg = upstream
    with WeightServer(d, host="127.0.0.1", port=0) as srv:
        yield f"http://127.0.0.1:{srv.port}", srv.stats, cfg, d


def _direct(d, plan, node, out):
    assert _run("--src", d, "--plan", plan, "--node", node, "--out", out) == 0
    return out


def test_local_reuse_only_pulls_what_changed(up, tmp_path):
    url, stats, cfg, d = up
    old, new = _plans(tmp_path, cfg)
    out = tmp_path / "w"
    assert _run("--base-url", url, "--plan", old, "--node", "n2", "--out", out) == 0
    full_old = stats["bytes"]
    before = stats["bytes"]
    assert _run("--base-url", url, "--plan", new, "--node", "n2", "--out", out, "--reuse") == 0
    pulled = stats["bytes"] - before
    _same(out, _direct(d, new, "n2", tmp_path / "ref"))
    assert pulled < full_old, "注意力 / 路由 / norm 没变，应当从本机旧文件拿"


def test_peer_supplies_tensors_and_upstream_only_the_rest(up, tmp_path):
    url, stats, cfg, d = up
    old, new = _plans(tmp_path, cfg)
    peer_dir = tmp_path / "peer"
    assert _run("--src", d, "--plan", new, "--node", "n2", "--out", peer_dir) == 0
    with WeightServer(peer_dir, host="127.0.0.1", port=0) as peer:
        before = stats["bytes"]
        out = tmp_path / "me"
        assert _run("--base-url", url, "--plan", new, "--node", "n2", "--out", out,
                    "--peers", f"N99=http://127.0.0.1:{peer.port}") == 0
        assert peer.stats["bytes"] > 0
    # 对等节点有这台要的全部张量：上游只发了 config / tokenizer 和文件头，不发张量数据
    assert stats["bytes"] - before < 64 << 10
    _same(out, _direct(d, new, "n2", tmp_path / "ref"))


def test_peer_of_another_model_is_not_used(up, tmp_path):
    url, stats, cfg, d = up
    _, new = _plans(tmp_path, cfg)
    peer_dir = tmp_path / "peer"
    assert _run("--src", d, "--plan", new, "--node", "n2", "--out", peer_dir) == 0
    c = json.loads((peer_dir / "config.json").read_text())
    c["hidden_size"] = c["hidden_size"] * 2
    (peer_dir / "config.json").write_text(json.dumps(c))
    with WeightServer(peer_dir, host="127.0.0.1", port=0) as peer:
        out = tmp_path / "me"
        assert _run("--base-url", url, "--plan", new, "--node", "n2", "--out", out,
                    "--peers", f"N99=http://127.0.0.1:{peer.port}") == 0
        assert peer.stats["bytes"] < 64 << 10        # 只被问了 config.json
    _same(out, _direct(d, new, "n2", tmp_path / "ref"))


def test_peer_dying_midway_falls_back_to_upstream(up, tmp_path):
    """A 有旧方案的张量（注意力、路由、旧专家），B 有新方案的全部；B 在开始传数据时掉线
    —— A 给的照用，B 该给的全部改从上游拿，结果仍与直接拉的逐位一致。"""
    url, stats, cfg, d = up
    old, new = _plans(tmp_path, cfg)
    a_dir, b_dir = tmp_path / "a", tmp_path / "b"
    assert _run("--src", d, "--plan", old, "--node", "n2", "--out", a_dir) == 0
    assert _run("--src", d, "--plan", new, "--node", "n2", "--out", b_dir) == 0
    man = F.DeploymentManifest.from_json(new.read_text(encoding="utf-8"))
    keys = F.keys_for_node(man, "n2", config=cfg)
    a = F.Donor("NA", F.Source(local=a_dir))
    b = F.Donor("NB", F.Source(local=b_dir))
    upsrc = F.Source(base_url=url)
    assigned = F.plan_sources([a, b], upsrc, keys)
    assert {d.label for d, _ in assigned} == {"NA", "NB"}

    def dead(name, start=None, end=None):
        raise OSError("connection reset by peer")

    b.src.read = dead                              # 文件头读完之后才掉线
    out = tmp_path / "me"
    got = F.fetch_reusing(assigned, upsrc, out)
    assert got.get("NA", 0) > 0 and got.get("上游", 0) > 0 and "NB" not in got
    _same(out, _direct(d, new, "n2", tmp_path / "ref"))


def test_stage_keeps_the_old_file_until_commit(up, tmp_path):
    url, stats, cfg, d = up
    old, new = _plans(tmp_path, cfg)
    out = tmp_path / "w"
    assert _run("--base-url", url, "--plan", old, "--node", "n2", "--out", out) == 0
    old_keys = F.local_checkpoint_keys(out)
    assert _run("--base-url", url, "--plan", new, "--node", "n2", "--out", out,
                "--reuse", "--stage") == 0
    assert F.local_checkpoint_keys(out) == old_keys, "别的节点可能还在读旧文件 —— 不能先盖掉"
    assert (out / F.STAGED).is_file()
    assert _run("--commit-staged", "--out", out) == 0
    assert not (out / F.STAGED).exists()
    _same(out, _direct(d, new, "n2", tmp_path / "ref"))
    assert _run("--commit-staged", "--out", out) == 0       # 再提交一次：什么都不做


def test_unchanged_node_is_skipped_with_reuse(up, tmp_path):
    url, stats, cfg, d = up
    old, _ = _plans(tmp_path, cfg)
    out = tmp_path / "w"
    assert _run("--base-url", url, "--plan", old, "--node", "n1", "--out", out) == 0
    st = (out / "model.safetensors").stat()
    before = stats["bytes"]
    assert _run("--base-url", url, "--plan", old, "--node", "n1", "--out", out,
                "--reuse", "--stage") == 0
    assert (out / "model.safetensors").stat().st_ino == st.st_ino
    assert not (out / F.STAGED).exists()
    assert stats["bytes"] - before < 64 << 10


def test_launch_starts_peers_stages_then_commits(monkeypatch, tmp_path):
    """控制机侧：先在全部节点起权重服务，各节点 --stage 拉，全部成功后统一 --commit-staged。"""
    from p2pmoe.deploy import launch as L

    hosts = tmp_path / "h.txt"
    hosts.write_text("N01  10.0.0.1:9101\nN02  10.0.0.2:9101\n", encoding="utf-8")
    allh = tmp_path / "all.txt"
    allh.write_text("N01  10.0.0.1:9101\nN02  10.0.0.2:9101\nN03  10.0.0.3:9101\n",
                    encoding="utf-8")
    plan = tmp_path / "plan.json"
    plan.write_text('{"l0": 1, "model": {}, "segments": {}, "nodes": []}', encoding="utf-8")
    cmds: list[tuple[str, str]] = []

    def fake_ssh(h, cmd, **kw):
        cmds.append((h.node_id, cmd))
        if "serve_weights" in cmd and "nohup" in cmd:
            return True, "serving"
        return True, "ok"

    monkeypatch.setattr(L, "_ssh", fake_ssh)
    monkeypatch.setattr(L, "_peer_alive", lambda url, timeout=5.0: True)
    assert L.main(["fetch", "--hosts", str(hosts), "--plan", str(plan), "--repo", "R",
                   "--out", "/w", "--workdir", "/c", "--replace", "--peers",
                   "--peer-hosts", str(allh), "--progress-every", "0"]) == 0
    started = [n for n, c in cmds if "nohup" in c]
    fetched = [(n, c) for n, c in cmds if "p2pmoe.deploy.fetch" in c and "--plan" in c]
    committed = [n for n, c in cmds if "--commit-staged" in c]
    stopped = [n for n, c in cmds if "pkill" in c]
    assert sorted(started) == ["N01", "N02", "N03"]             # 不在清单里的 N03 也当源
    assert len(fetched) == 2 and all("--stage" in c and "--peers" in c for _, c in fetched)
    assert all("N01=" not in c for n, c in fetched if n == "N01")  # 不把自己当对等节点
    assert sorted(committed) == ["N01", "N02"]
    assert sorted(stopped) == ["N01", "N02", "N03"]
    order = [i for i, (_, c) in enumerate(cmds)
             if "--commit-staged" in c or ("p2pmoe.deploy.fetch" in c and "--plan" in c)]
    kinds = ["commit" if "--commit-staged" in cmds[i][1] else "fetch" for i in order]
    assert kinds == ["fetch", "fetch", "commit", "commit"], "全部拉完才换"
