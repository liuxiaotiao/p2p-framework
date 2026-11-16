"""selector_body_L2_cmw：counts 份额 + 平均门控权重（verify/train_selector_L2.py --feature counts+meanw）。

特征口径必须与训练时逐位一致：每层拼 [count/Σcount (E), wsum/count (E)]，按层展平。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from p2pmoe.runtime.lr_classifier import (FeatureSpec, _SELECTOR_FEATURES, _features,
                                          load_lr_params)

ROOT = Path(__file__).resolve().parents[1]
SEL = ROOT / "selector" / "selector_body_L2_cmw_params.npz"


def _verify_feats(c, w, n):
    """verify/task_selector.py 的 features(..., feature="prefill_counts_body+meanw") 原样照抄。"""
    c = np.asarray(c, np.float64)[:, :n, :]
    s = c.sum(2, keepdims=True); s[s == 0] = 1.0
    w = np.asarray(w, np.float64)[:, :n, :]
    mw = np.divide(w, c, out=np.zeros_like(w), where=c > 0)
    return np.concatenate([c / s, mw], 2).reshape(len(c), -1)


def test_features_match_training_formula():
    rng = np.random.default_rng(0)
    C = rng.integers(0, 6, size=(20, 2, 16)).astype(float)
    C[0, 1] = 0                                          # 一层全零：份额与平均权重都取 0
    W = C * rng.random((20, 2, 16))
    spec = FeatureSpec(layers=[0, 1], layer_base=0, **_SELECTOR_FEATURES["prefill_counts_body+meanw"])
    ours = np.stack([_features(spec, C[i], W[i], np.ones(2)) for i in range(len(C))])
    assert np.array_equal(ours, _verify_feats(C, W, 2))


def test_repo_selector_loads_as_512_dim_two_layer_body_counts_plus_meanw():
    if not SEL.exists():
        pytest.skip("缺 selector/selector_body_L2_cmw_params.npz")
    p = load_lr_params(SEL)
    assert p["coef"].shape == (5, 512)
    assert p["meta"]["feature"] == "prefill_counts_body+meanw" and p["meta"]["n_layers"] == 2
    f = p["features"]
    assert f["stats"] == ["count", "mean_weight"] and f["layers"] == [0, 1] and f["tokens"] == "body"
