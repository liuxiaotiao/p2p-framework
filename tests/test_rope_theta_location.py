"""rope_theta 的位置：顶层，或 transformers 5.x 存的 rope_parameters 里。

Qwen3-MoE（Qwen3-30B-A3B）那条路以前只读顶层。transformers 5.x 写出的 config
把它挪进了 `rope_parameters`，于是会悄悄落到默认的 1e6 —— 不报错，只是输出变差。
用 1e6 当测试值测不出来（那正是默认值），所以这里用 1e7。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

pytest.importorskip("torch")

from p2pmoe.runtime.torch_model import TorchModelConfig
from p2pmoe.sim.fake_checkpoint import TINY_QWEN3_MOE


def _cfg(**kw):
    c = {k: v for k, v in TINY_QWEN3_MOE.items() if k != "rope_theta"}
    c.update(kw)
    return c


def test_top_level() -> None:
    assert TorchModelConfig.from_hf(_cfg(rope_theta=1e7)).rope_theta == 1e7


def test_transformers_5_rope_parameters() -> None:
    c = _cfg(rope_parameters={"rope_type": "default", "rope_theta": 1e7})
    assert TorchModelConfig.from_hf(c).rope_theta == 1e7


def test_nested_wins_when_both_present() -> None:
    c = _cfg(rope_theta=1e6, rope_parameters={"rope_theta": 1e7})
    assert TorchModelConfig.from_hf(c).rope_theta == 1e7


def test_default_when_absent() -> None:
    assert TorchModelConfig.from_hf(_cfg()).rope_theta == 1e6
