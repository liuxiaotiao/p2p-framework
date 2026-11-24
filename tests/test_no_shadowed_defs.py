"""同一个模块里不能有两个同名的顶层函数 / 类 —— 后一个会悄悄盖掉前一个。

2026-11-24：launch.py 里给节点互传加的 `_stop_cmd(out, port)` 盖掉了停 agent 的
`_stop_cmd(h, by_port=…)`，`deploy_15.sh stop / start` 全部报 TypeError。"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_no_module_redefines_a_top_level_name():
    bad = []
    for f in sorted((ROOT / "p2pmoe").rglob("*.py")) + sorted((ROOT / "examples").glob("*.py")):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        names = Counter(n.name for n in tree.body
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))
        bad += [f"{f.relative_to(ROOT)}: {k}" for k, c in names.items() if c > 1]
    assert not bad, "重名的顶层定义：" + "；".join(bad)


def test_agent_stop_and_peer_stop_are_different_commands():
    import sys

    sys.path.insert(0, str(ROOT))
    from p2pmoe.deploy import launch as L

    a = L._stop_cmd(L.Host("N01", "h"), by_port=True)
    assert "p2pmoe.deploy.agent --id N01" in a
    b = L._serve_stop_cmd("/w", 9400)
    assert "p2pmoe.deploy.serve_weights --dir /w --bind 0.0.0.0:9400" in b
    assert "pkill -f" not in b                    # 会先杀掉执行它的远端 shell
