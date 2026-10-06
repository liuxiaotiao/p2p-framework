"""换模型时 sync 不能把另一个模型的权重删掉。

`deploy_15.sh sync` 用 `rsync --delete`。以前只排除**当前** `$WEIGHTS`：把 WEIGHTS
改成 weights-q3-30b 之后，各节点上旧的 weights/（Qwen3-Next 那 141GB）就不再受
保护 —— 控制机的 weights/ 里只有 config 与 tokenizer，于是节点上的分片被当成
「源端没有」删掉。在一台真机上用 rsync 复现过：旧规则下节点只剩一个 config.json。
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

SH = (Path(__file__).resolve().parents[1] / "deploy_15.sh").read_text(encoding="utf-8")


def _block(name: str) -> str:
    i = SH.index(f"{name}() {{")
    return SH[i:SH.index("\n}\n", i)]


def test_sync_excludes_every_weights_dir() -> None:
    b = _block("cmd_sync")
    assert "rsync -az --delete" in b
    assert "--exclude '/weights*/'" in b, "sync 只排除当前 WEIGHTS —— 换模型会删掉另一份"
    assert "--exclude '/Delta_30B/'" in b


def test_bootstrap_tarball_excludes_every_weights_dir() -> None:
    assert re.search(r"tar czf .*?--exclude '\./weights\*'", SH, re.S)


@pytest.mark.skipif(shutil.which("rsync") is None, reason="没有 rsync")
def test_rsync_really_keeps_both(tmp_path) -> None:
    src, dst = tmp_path / "src", tmp_path / "dst"
    for d in ("weights", "weights-q3-30b", "p2pmoe"):
        (src / d).mkdir(parents=True)
    (src / "weights" / "config.json").write_text("meta only")
    (src / "p2pmoe" / "a.py").write_text("code")
    for d in ("weights", "weights-q3-30b"):
        (dst / d).mkdir(parents=True)
        (dst / d / "model-00001.safetensors").write_text("shard")
    subprocess.run(["rsync", "-az", "--delete", "--exclude", "weights-q3-30b",
                    "--exclude", "/weights*/", "./", f"{dst}/"], cwd=src, check=True)
    assert (dst / "weights" / "model-00001.safetensors").exists()
    assert (dst / "weights-q3-30b" / "model-00001.safetensors").exists()
    assert (dst / "p2pmoe" / "a.py").exists()
