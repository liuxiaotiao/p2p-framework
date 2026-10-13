"""samples.txt → cases.jsonl：prompt 与选择器训练时逐字相同。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "build_cases", Path(__file__).resolve().parent.parent / "examples" / "build_cases.py")
bc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bc)

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def _cache(tmp: Path) -> Path:
    c = tmp / "cache"
    for task, rows in {
        "gsm8k": [{"question": f"Q{i}?", "answer": "#### 1"} for i in range(5)],
        "mbpp": [{"task_id": i, "text": f"Write f{i}.", "code": "",
                  "test_list": [f"assert f{i}(1) == 1", f"assert f{i}(2) == 2"]}
                 for i in range(5)],
        "no_robots": [
            {"prompt": "p0", "category": "Chat", "messages": [
                {"role": "user", "content": "hi"}, {"role": "assistant", "content": "a"},
                {"role": "user", "content": "more"}, {"role": "assistant", "content": "b"}]},
            {"prompt": "p1", "category": "Gen", "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "Write a poem."},
                {"role": "assistant", "content": "x"}]},
            {"prompt": "p2", "category": "Gen", "messages": [
                {"role": "user", "content": "Tell me a joke."},
                {"role": "assistant", "content": "y"}]},
        ],
    }.items():
        (c / task).mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(rows), c / task / "test.parquet")
    return c


def test_builds_training_identical_prompts(tmp_path, monkeypatch):
    monkeypatch.setattr(bc, "_datasets", lambda: None)        # 走 parquet 那条路
    s = tmp_path / "samples.txt"
    s.write_text("gsm8k/test/3\nmbpp/test/1\nnorobots/test/2\nnorobots/test/1\n")
    out = tmp_path / "cases.jsonl"
    assert bc.main(["--samples", str(s), "--out", str(out), "--cache",
                    str(_cache(tmp_path))]) == 0
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert [r["id"] for r in rows] == ["gsm8k:3", "mbpp:1", "no_robots:2", "no_robots:1"]
    assert rows[0]["segments"] == [[bc.GSM8K_TPL, True], ["Q3?", False]]
    assert rows[1]["segments"] == [[bc.MBPP_TPL, True], ["Write f1.", False],
                                   [bc.MBPP_MID, True],
                                   ["assert f1(1) == 1\nassert f1(2) == 2", False]]
    assert rows[2]["segments"] == [["Tell me a joke.", False]]
    assert rows[3]["segments"] == [["Be brief.\n\n", True], ["Write a poem.", False]]
    # 控制面能直接读
    from p2pmoe.deploy.control import _load_cases_jsonl
    cs = _load_cases_jsonl(out)
    assert [u for u, _ in cs] == ["gsm8k", "mbpp", "no_robots", "no_robots"]
    assert cs[0][1]["id"] == "gsm8k:3"


def test_rejects_multi_turn_and_out_of_range(tmp_path, monkeypatch):
    monkeypatch.setattr(bc, "_datasets", lambda: None)
    c = _cache(tmp_path)
    s = tmp_path / "samples.txt"
    s.write_text("norobots/test/0\n")
    assert bc.main(["--samples", str(s), "--out", str(tmp_path / "o.jsonl"),
                    "--cache", str(c)]) == 1
    s.write_text("gsm8k/test/99\n")
    assert bc.main(["--samples", str(s), "--out", str(tmp_path / "o.jsonl"),
                    "--cache", str(c)]) == 1


def test_bodies_cross_check(tmp_path, monkeypatch):
    monkeypatch.setattr(bc, "_datasets", lambda: None)
    c = _cache(tmp_path)
    bd = tmp_path / "bodies"
    bd.mkdir()
    (bd / "gsm8k_bodies.jsonl").write_text(
        "\n".join(json.dumps({"sample_idx": i, "body": f"Q{i}?"}) for i in range(5)))
    (bd / "mbpp_bodies.jsonl").write_text("\n".join(json.dumps(
        {"sample_idx": i, "body": f"Write f{i}.assert f{i}(1) == 1\nassert f{i}(2) == 2"})
        for i in range(5)))
    s = tmp_path / "samples.txt"
    s.write_text("gsm8k/test/2\nmbpp/test/4\n")
    args = ["--samples", str(s), "--out", str(tmp_path / "o.jsonl"), "--cache", str(c),
            "--bodies-dir", str(bd)]
    assert bc.main(args) == 0
    (bd / "gsm8k_bodies.jsonl").write_text(json.dumps({"sample_idx": 2, "body": "other"}))
    assert bc.main(args) == 1
