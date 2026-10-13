"""`launch start` 的远端命令必须立刻返回。

真机上 start 一直报「ssh 超过 60s 没返回」0/15，而 agent 其实已经起来了：
`mkdir … && cd … && setsid nohup agent … & echo $!` 里 `&` 作用于整条 `&&` 链，
bash 开的后台子 shell 仍连着 ssh 的输出管道，要等 agent 退出才结束。

这里用 `bash -c` + 捕获输出代替 ssh —— 捕获管道等 EOF 的行为与 ssh 等通道关闭相同。
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from p2pmoe.deploy.launch import Host, _start_cmd


def test_start_command_returns_while_the_agent_keeps_running(tmp_path) -> None:
    fake = tmp_path / "py"
    fake.write_text("#!/bin/bash\necho started \"$@\"\nsleep 30\n")
    fake.chmod(0o755)
    cmd = _start_cmd(Host("N01", "x"), str(tmp_path), str(fake), str(tmp_path / "log"))
    t = time.time()
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=5)
    assert time.time() - t < 3, "启动命令没有立即返回 —— ssh 会挂到超时"
    pid = int(r.stdout.strip())
    try:
        time.sleep(0.3)
        os.kill(pid, 0)                                   # 还活着
        log = (tmp_path / "log" / "agent-N01.log").read_text()
        assert "started -m p2pmoe.deploy.agent --id N01" in log
    finally:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except ProcessLookupError:
            pass


def test_the_background_job_is_only_the_agent() -> None:
    cmd = _start_cmd(Host("N01", "x"), "/w", "/py", "/l")
    assert "{ setsid nohup" in cmd and "< /dev/null & }" in cmd
    assert shlex.split(cmd)  # 引号配平
