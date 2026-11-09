"""Petals 分配 → 布局 → 清单（examples/petals_to_layout.py + manual_plan.py）。REPRODUCE 5.14。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "examples" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_slices_are_0_based_half_open_and_front_is_first_l0_layers():
    pl = _load("petals_to_layout")
    lay = pl.convert({"chains": {"A": {"servers": [
        {"node": "a", "layers": "8:20"}, {"node": "b", "layers": "0:8"}, {"node": "c", "layers": "20:48"}]}}})
    ch = lay["channels"][0]
    assert lay["l0"] == 8 and ch["task"] == "any"
    assert ch["front"] == [{"node": "b", "layers": [1, 8]}]
    assert ch["back"] == [{"node": "a", "layers": [9, 20]}, {"node": "c", "layers": [21, 48]}]


def test_l0_inside_a_server_is_refused():
    pl = _load("petals_to_layout")
    with pytest.raises(SystemExit, match="中间"):
        pl.convert({"chains": {"A": {"servers": [
            {"node": "a", "layers": "0:12"}, {"node": "b", "layers": "12:48"}]}}}, l0=8)


def test_repo_petals_cap8_plan_is_reproducible(tmp_path, capsys):
    src = ROOT / "qwen3-30b" / "petals" / "plan_cap8_petals.json"
    want_lay = ROOT / "qwen3-30b" / "layout_petals_cap8.json"
    want = ROOT / "qwen3-30b" / "plan_petals_cap8.json"
    if not (src.exists() and want.exists()):
        pytest.skip("缺 qwen3-30b/petals/")
    lay = tmp_path / "l.json"
    assert _load("petals_to_layout").main([str(src), "--out", str(lay)]) == 0
    assert lay.read_bytes() == want_lay.read_bytes()
    out = tmp_path / "p.json"
    assert _load("manual_plan").main([str(lay), "--out", str(out)]) == 0
    assert out.read_bytes() == want.read_bytes()
    man = json.loads(want.read_text())
    assert {n["node"] for n in man["nodes"]} == {
        "N06", "N01", "N07", "N08", "N09", "N10", "N02", "N03", "N04", "N05", "N11", "N12"}
    assert all(len(l["experts"]) == 128 for n in man["nodes"] for l in n["layers"])
