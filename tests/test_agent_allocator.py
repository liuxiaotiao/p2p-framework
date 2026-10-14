"""agent 在 import torch 之前把显存分配器设成按需扩段（见 agent._allocator_env）。"""

from __future__ import annotations

import subprocess
import sys

import pytest

from p2pmoe.deploy import agent


def test_sets_expandable_segments_when_unset(monkeypatch):
    monkeypatch.setattr("importlib.metadata.version", lambda name: "2.4.1+cu121")
    env: dict[str, str] = {}
    assert agent._allocator_env(env) == "expandable_segments:True"
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


def test_user_setting_wins(monkeypatch):
    monkeypatch.setattr("importlib.metadata.version", lambda name: "2.4.1")
    env = {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"}
    assert agent._allocator_env(env) is None
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:128"


@pytest.mark.parametrize("v", ["2.0.1", "1.13.1+cu117"])
def test_old_torch_is_left_alone(monkeypatch, v):
    monkeypatch.setattr("importlib.metadata.version", lambda name: v)
    env: dict[str, str] = {}
    assert agent._allocator_env(env) is None and env == {}


def test_set_before_torch_is_imported():
    """真起一个解释器：import agent 之后 torch 还没被导入，而变量已经设好。"""
    code = ("import os, sys; os.environ.pop('PYTORCH_CUDA_ALLOC_CONF', None); "
            "import p2pmoe.deploy.agent as a; "
            "print('torch' in sys.modules, os.environ.get('PYTORCH_CUDA_ALLOC_CONF'))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={k: v for k, v in __import__('os').environ.items()
                              if k != "PYTORCH_CUDA_ALLOC_CONF"}, check=True).stdout.split()
    assert out[0] == "False"
    try:
        import importlib.metadata as md
        md.version("torch")
        assert out[1] == "expandable_segments:True"
    except md.PackageNotFoundError:
        pass
