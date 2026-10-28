"""deploy_15.sh 在新终端里忘了 source env.sh 时自动补上。

症状（真机上踩过两次）：fetch 预检用旧默认值 —— /home/ubuntu/P2P_MoE、系统 python3 3.8 —— 报
`Addr = tuple[str, int]` / `'type' object is not subscriptable`。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "deploy_15.sh"

pytestmark = pytest.mark.skipif(not (SCRIPT.exists() and (ROOT / "qwen3-30b" / "env.sh").exists()),
                                reason="没有 deploy_15.sh 或 qwen3-30b/env.sh")


def _prefix_vars(env: dict) -> dict:
    """只跑脚本开头（自动 source + 默认值），打印关心的几个变量。"""
    src = SCRIPT.read_text(encoding="utf-8")
    head = src.split("# 权重放哪儿", 1)[0]
    probe = ROOT / ".test-autoenv.sh"
    probe.write_text(head + '\necho "WORKDIR=$WORKDIR"; echo "NODE_PY=${NODE_PY:-}"; '
                     'echo "PLAN=$PLAN"; echo "HOSTS=$HOSTS"; echo "REPO=$REPO"\n', encoding="utf-8")
    try:
        r = subprocess.run(["bash", str(probe)], capture_output=True, text=True, timeout=30,
                           cwd=ROOT, env={"PATH": "/usr/bin:/bin", **env})
    finally:
        probe.unlink(missing_ok=True)
    assert r.returncode == 0, r.stderr
    return dict(line.split("=", 1) for line in r.stdout.splitlines() if "=" in line)


def test_missing_workdir_sources_env_sh():
    v = _prefix_vars({})
    assert v["WORKDIR"] == "/home/ubuntu/p2p-framework"
    assert v["NODE_PY"].endswith("envs/moe/bin/python")
    assert v["REPO"] == "Qwen/Qwen3-30B-A3B"
    assert v["PLAN"] == "qwen3-30b/plan_es90_fill.json"      # env.sh 的默认


def test_exported_plan_and_hosts_survive():
    v = _prefix_vars({"PLAN": "qwen3-30b/plan_vllm_rr_fill.json", "HOSTS": "task/hosts_x.txt"})
    assert v["PLAN"] == "qwen3-30b/plan_vllm_rr_fill.json" and v["HOSTS"] == "task/hosts_x.txt"
    assert v["WORKDIR"] == "/home/ubuntu/p2p-framework"


def test_explicit_workdir_is_respected():
    v = _prefix_vars({"WORKDIR": "/somewhere", "NODE_PY": "py"})
    assert v["WORKDIR"] == "/somewhere" and v["NODE_PY"] == "py"
