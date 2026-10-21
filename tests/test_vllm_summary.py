"""vllm/summary_by_gpu.json → layer_bench 表（p2pmoe/deploy/vllm_summary.py）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from p2pmoe.deploy.layer_bench import per_node_memory, per_node_ms
from p2pmoe.deploy.vllm_summary import _gen_mean, _interp, to_layer_table

ROOT = Path(__file__).resolve().parents[1]

SUMMARY = {
    "X": {
        "gpu": "GPU-X", "nodes": ["N1", "N2"], "ms_per_layer": 1.0,
        "ms_per_layer_nodes": {"N1": 0.9, "N2": 1.1},
        "decode": {"128": {"512": {"1": 1.0, "2": 9.0}, "1024": {"1": 2.0}}},
        "prefill": {"128": {"256": 10.0, "512": 20.0}},
        "decode_overhead_ms": {"128": {"512": {"1": 3.0}, "1024": {"1": 3.0}}},
        "memory": {"128": {"weight_mib_per_layer": 1000.0, "device_non_kv_mib_per_layer": 1010.0,
                           "non_layer_weight_mib": 1000.0, "device_non_kv_mib_fixed": 2500.0,
                           "device_total_mib": 20000.0, "kv_kib_per_token_per_layer": 2.0}},
    }
}


def test_interp_and_generation_mean():
    pts = {512: 1.0, 1024: 2.0}
    assert _interp(pts, 768) == pytest.approx(1.5)
    assert _interp(pts, 100) == 1.0 and _interp(pts, 4096) == 2.0
    # 上下文 512..1023 的平均 = 1 + (0..511 的均值)/512
    assert _gen_mean(pts, 512, 512) == pytest.approx(1.0 + 255.5 / 512)


def test_table_values_and_node_scaling():
    t = to_layer_table(SUMMARY, prompt=512, gen=512, batch=1)
    d = 1.0 + 255.5 / 512
    n1 = t["per_node"]["N1"]
    assert n1["decode@p512g512"] == pytest.approx(round(d * 0.9, 4))
    assert n1["prefill@512"] == pytest.approx(20.0 * 0.9)
    assert n1["req@p512g512"] == pytest.approx(round((20.0 + 512 * d) / 512 * 0.9, 4))
    assert t["per_node"]["N2"]["prefill@512"] == pytest.approx(22.0)
    assert n1["fixed_ms_per_token"] == pytest.approx(3.0)
    mib = 2 ** 20 / 1e9
    m = n1["memory"]
    assert m["total_gb"] == pytest.approx(20000 * mib, abs=1e-4)
    assert m["context_gb"] == pytest.approx(1500 * mib, abs=1e-4)      # 固定 − 词嵌入与输出头
    assert m["overhead_ratio"] == pytest.approx(1.01)
    assert m["embed_gb"] == m["lm_head_gb"] == pytest.approx(500 * mib, abs=1e-4)
    # 规划器的读取接口直接能用
    assert per_node_ms(t, "req@p512g512")["N2"] == t["per_node"]["N2"]["req@p512g512"]
    assert per_node_memory(t, 512)["N1"]["total_gb"] == m["total_gb"]


def test_missing_batch_is_an_error():
    with pytest.raises(ValueError, match="batch=4"):
        to_layer_table(SUMMARY, batch=4)


def test_repo_table_matches_the_summary():
    src, out = ROOT / "vllm" / "summary_by_gpu.json", ROOT / "qwen3-30b" / "layer_bench_vllm.json"
    if not (src.exists() and out.exists()):
        pytest.skip("缺 vllm/ 或 qwen3-30b/layer_bench_vllm.json")
    want = to_layer_table(json.loads(src.read_text(encoding="utf-8")), prompt=512, gen=512, batch=1)
    assert json.loads(out.read_text(encoding="utf-8")) == want
