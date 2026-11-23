"""换模型时删各台另一个模型的权重目录（deploy_15.sh purge-weights / PURGE_OLD=… fetch）。

ssh 换成一个桩：把远端命令里的 /W 换成本机临时目录下各「节点」自己的目录，在本机执行。
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "deploy_15.sh"
pytestmark = pytest.mark.skipif(not SCRIPT.exists() or not Path("/proc/self/fd").exists(),
                                reason="要 deploy_15.sh 与 Linux 的 /proc")


@pytest.fixture
def pool(tmp_path):
    for n in ("n1", "n3"):
        d = tmp_path / n / "weights-q3-30b"
        d.mkdir(parents=True)
        (d / "config.json").write_text(json.dumps(
            {"model_type": "qwen3_moe", "num_hidden_layers": 48, "num_experts": 128}))
        (d / "model.safetensors").write_bytes(b"\0" * 4096)
    (tmp_path / "n2").mkdir()
    (tmp_path / "n1" / "weights-glm47").mkdir()
    stub = tmp_path / "ssh"
    stub.write_text("#!/bin/bash\nhost=${@: -2:1}; cmd=${@: -1}\n"
                    f'cmd=${{cmd//\\/W/{tmp_path}\\/$host}}\nexec bash -c "$cmd"\n')
    stub.chmod(0o755)
    hosts = tmp_path / "hosts"
    hosts.write_text("n1 n1:9101\nn2 n2:9101\nn3 n3:9101  # 注释\n")
    return tmp_path


def _run(pool: Path, *args: str, answer: str = "n", **env) -> str:
    r = subprocess.run(
        ["bash", str(SCRIPT), *args], input=answer + "\n", capture_output=True, text=True,
        timeout=60, cwd=ROOT,
        env={"PATH": "/usr/bin:/bin", "HOME": os.environ.get("HOME", "/tmp"),
             "SSH_BIN": str(pool / "ssh"), "HOSTS": str(pool / "hosts"), "WORKDIR": "/W",
             "WEIGHTS": "/W/weights-glm47", "PLAN": "/dev/null", **env})
    return re.sub(r"\x1b\[[0-9;]*m", "", r.stdout + r.stderr)


def test_lists_first_and_deletes_only_after_yes(pool):
    out = _run(pool, "purge-weights", "/W/weights-q3-30b", answer="n")
    assert "n1  有" in out and "qwen3_moe，48 层 × 128 专家" in out
    assert "n2  —  没有这个目录" in out
    assert (pool / "n1" / "weights-q3-30b").exists()          # 回 n：什么都没删
    out = _run(pool, "purge-weights", "/W/weights-q3-30b", answer="y")
    assert out.count("✓ 已删") == 2
    assert not (pool / "n1" / "weights-q3-30b").exists()
    assert not (pool / "n3" / "weights-q3-30b").exists()
    assert (pool / "n1" / "weights-glm47").exists()            # 别的目录不碰


def test_refuses_current_weights_and_odd_names(pool):
    assert "就是当前的 WEIGHTS" in _run(pool, "purge-weights", "/W/weights-glm47")
    assert "不以 weights 开头" in _run(pool, "purge-weights", "/W/models")
    assert "要绝对路径" in _run(pool, "purge-weights", "weights-q3-30b")
    assert (pool / "n1" / "weights-glm47").exists()


def test_busy_directory_is_not_deleted(pool):
    f = pool / "n3" / "weights-q3-30b" / "config.json"
    with open(f):                                              # 模拟 agent 还开着权重
        out = _run(pool, "purge-weights", "/W/weights-q3-30b", CONFIRM="1")
    assert "有进程开着里面的文件" in out and "先 bash ./deploy_15.sh stop" in out
    assert "已删" not in out                                   # 有一台不行就一台都不删
    assert (pool / "n1" / "weights-q3-30b").exists()


def test_unreachable_node_shows_the_ssh_error(pool):
    stub = pool / "ssh"
    stub.write_text(stub.read_text().replace(
        "exec bash", 'case "$host" in n2) echo "ssh: connect to host n2 port 22: No route to host" >&2; exit 255 ;; esac\nexec bash'))
    out = _run(pool, "purge-weights", "/W/weights-q3-30b", CONFIRM="1")
    assert "n2  ✗ ssh 连不上 n2：ssh: connect to host n2 port 22: No route to host" in out
    assert "已删" not in out


def test_a_flaky_ssh_is_retried(pool):
    """kex 阶段被信号打断这种一次性的 255：重试一次就过，不该拦住整批。"""
    stub = pool / "ssh"
    stub.write_text(stub.read_text().replace(
        "exec bash", f'f={pool}/flaky.$host; if [ "$host" = n3 ] && [ ! -e "$f" ]; then touch "$f"; '
        'echo "kex_exchange_identification: Interrupted system call" >&2; exit 255; fi\nexec bash'))
    out = _run(pool, "purge-weights", "/W/weights-q3-30b", CONFIRM="1")
    assert out.count("✓ 已删") == 2 and "✗" not in out
