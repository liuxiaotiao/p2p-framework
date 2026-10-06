"""前段出口的 LR 识别器。

分四层测：

1. **逐层统计**本身 —— 每层都留下来、数对、捎带格式往返无损，且没开的时候
   报文与以前逐字节相同（老节点、decode、后段不付这份字节）。
2. **装载** —— sklearn 的 LR / Pipeline / 字典打包、npz、json 都要化成同一组
   numpy 参数，且**概率与 sklearn 的 predict_proba 逐位一致**。这是唯一真正
   要紧的断言：训练时说 0.83，节点上就得是 0.83。
3. **特征拼法与核对** —— 维度不对、层越界、类别对不上，都要在控制机装载时
   就报，并且报得出「这个维度对应几层」。
4. **端到端** —— 真进程：动态模式下 LR 决定绑哪个池；静态模式下只旁路上报；
   逐层统计落盘后离线重算，与在线结果一致。
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from p2pmoe.runtime.identify import HistogramClassifier
from p2pmoe.runtime.lr_classifier import (
    FeatureSpec,
    LRClassifier,
    load_lr_params,
    make_classifier,
    parse_class_map,
)
from p2pmoe.runtime.model import (
    LayerStat,
    MoEStats,
    SegmentModel,
    ToyMoEConfig,
    embed_tokens,
)

sklearn = pytest.importorskip("sklearn")
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler  # noqa: E402

CFG = ToyMoEConfig(n_layers=6, d_model=64, n_experts=16, d_ff=64, vocab=128)
E = CFG.n_experts


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _rand_layers(rng, l0: int, e: int = E, tokens: int = 20, k: int = 2,
                 bias: int | None = None) -> dict[int, LayerStat]:
    """一条请求前 l0 层的统计。bias 给了就让每层偏向某几个专家（可分）。"""
    out = {}
    for l in range(1, l0 + 1):
        p = np.ones(e)
        if bias is not None:
            p[(np.arange(3) + bias * 3 + l) % e] += 12.0
        p /= p.sum()
        picks = rng.choice(e, size=(tokens, k), p=p)
        cnt = np.bincount(picks.reshape(-1), minlength=e).astype(np.int64)
        w = rng.uniform(0.2, 0.8, size=(tokens, k))
        mass = np.zeros(e)
        np.add.at(mass, picks.reshape(-1), w.reshape(-1))
        out[l] = LayerStat(cnt, mass, tokens)
    return out


def _dataset(rng, l0: int, n_per: int, classes=(0, 1, 2)):
    """按默认特征口径（count，层优先）拼出 X、y —— 也就是「外面训练」那一步。"""
    X, y, L = [], [], []
    for c in classes:
        for _ in range(n_per):
            lay = _rand_layers(rng, l0, bias=c)
            X.append(np.concatenate([lay[l].count for l in range(1, l0 + 1)]))
            y.append(c)
            L.append(lay)
    return np.array(X, dtype=float), np.array(y), L


def _st(layers) -> MoEStats:
    return MoEStats(hist=np.zeros(E), layers=layers)


# --------------------------------------------------------------------------- #
# 1. 逐层统计
# --------------------------------------------------------------------------- #
def test_every_front_layer_is_kept_separately() -> None:
    """求和的 hist 会把「第 1 层偏爱 7」与「第 3 层偏爱 7」混成一个数 —— LR 要的是区分。"""
    m = SegmentModel(CFG, {l: list(range(E)) for l in (1, 2, 3)})
    ids = list(range(10))
    _, st = m.forward("r", embed_tokens(CFG, ids))
    assert sorted(st.layers) == [1, 2, 3]
    for l, s in st.layers.items():
        assert s.n_tokens == len(ids)
        assert s.count.sum() == len(ids) * CFG.top_k, "每个 token 每层恰好选 k 个"
    # 逐层之和就是原来那个求和直方图 —— 老识别器看到的东西一点没变
    assert np.allclose(sum(s.mass for s in st.layers.values()), st.hist)


def test_layer_stats_survive_the_wire() -> None:
    rng = np.random.default_rng(0)
    st = _st(_rand_layers(rng, 4))
    back = MoEStats.from_wire(json.loads(json.dumps(st.to_wire())))
    assert sorted(back.layers) == [1, 2, 3, 4]
    for l in back.layers:
        assert np.array_equal(back.layers[l].count, st.layers[l].count)
        assert np.allclose(back.layers[l].mass, st.layers[l].mass, atol=1e-6)
        assert back.layers[l].n_tokens == st.layers[l].n_tokens


def test_without_layers_the_wire_is_unchanged() -> None:
    """decode、后段、老节点：报文里不能多出东西。"""
    st = _st(_rand_layers(np.random.default_rng(1), 3))
    assert "lay" not in st.to_wire(layers=False)
    assert "lay" not in MoEStats.zeros(E).to_wire()
    assert set(MoEStats.zeros(E).to_wire()) == {"hist", "ntl", "miss", "mass"}


def test_merge_across_hops_keeps_all_layers() -> None:
    """前段跨两台机器：第一台带 1–2 层、第二台算 3–4 层，tail 要看到 1–4。"""
    rng = np.random.default_rng(2)
    lay = _rand_layers(rng, 4)
    a = _st({1: lay[1], 2: lay[2]})
    b = _st({3: lay[3], 4: lay[4]})
    m = b.merge(MoEStats.from_wire(a.to_wire()))
    assert sorted(m.layers) == [1, 2, 3, 4]


def test_the_node_only_piggybacks_layers_on_front_prefill() -> None:
    src = (Path(__file__).resolve().parents[1] / "p2pmoe/runtime/node.py").read_text(
        encoding="utf-8")
    assert 'lay = self.cfg.role == "front" and phase == "prefill"' in src
    assert "st.to_wire(layers=lay)" in src


# --------------------------------------------------------------------------- #
# 2. 装载：概率必须与 sklearn 逐位一致
# --------------------------------------------------------------------------- #
L0 = 3
PRI = {"mbpp": 5.0, "gsm8k": 3.0, "math": 1.0}
CMAP = "0=mbpp,1=gsm8k,2=math"


@pytest.fixture(scope="module")
def data():
    return _dataset(np.random.default_rng(7), L0, 30)


def _save(tmp_path, obj, name="m.pkl") -> Path:
    p = tmp_path / name
    with open(p, "wb") as f:
        pickle.dump(obj, f)
    return p


def _agree(clf: LRClassifier, est, layers_list, X) -> None:
    for lay, x in zip(layers_list, X):
        ours = clf.proba_raw(clf.features(lay))
        ref = est.predict_proba(x[None, :])[0]
        assert np.allclose(ours, ref, atol=1e-9), (ours, ref)


@pytest.mark.parametrize("pre", [None, StandardScaler, MinMaxScaler, MaxAbsScaler])
def test_matches_sklearn_predict_proba(tmp_path, data, pre) -> None:
    X, y, L = data
    lr = LogisticRegression(max_iter=2000, C=0.5)
    est = lr if pre is None else Pipeline([("s", pre()), ("lr", lr)])
    est.fit(X, y)
    clf = LRClassifier.load(_save(tmp_path, est), l0=L0, n_experts=E, priors=PRI,
                            class_map=CMAP)
    _agree(clf, est, L[::7], X[::7])


def test_two_affine_steps_compose(tmp_path, data) -> None:
    X, y, L = data
    est = Pipeline([("a", MinMaxScaler()), ("b", StandardScaler()),
                    ("lr", LogisticRegression(max_iter=2000))]).fit(X, y)
    clf = LRClassifier.load(_save(tmp_path, est), l0=L0, n_experts=E, priors=PRI,
                            class_map=CMAP)
    _agree(clf, est, L[::9], X[::9])


def test_binary_model(tmp_path, data) -> None:
    X, y, L = data
    keep = y < 2
    est = LogisticRegression(max_iter=2000).fit(X[keep], y[keep])
    clf = LRClassifier.load(_save(tmp_path, est), l0=L0, n_experts=E,
                            priors={"mbpp": 1, "gsm8k": 1}, class_map="mbpp,gsm8k")
    assert clf.proba == "binary"
    _agree(clf, est, [l for l, k in zip(L, keep) if k][::5], X[keep][::5])


def test_dict_bundle_and_joblib(tmp_path, data) -> None:
    import joblib

    X, y, L = data
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(max_iter=2000).fit(sc.transform(X), y)
    p = tmp_path / "bundle.joblib"
    joblib.dump({"model": lr, "scaler": sc}, p)
    clf = LRClassifier.load(p, l0=L0, n_experts=E, priors=PRI, class_map=CMAP)
    ref = Pipeline([("s", sc), ("lr", lr)])
    _agree(clf, ref, L[::11], X[::11])


def test_npz_and_json_give_the_same_answer(tmp_path, data) -> None:
    X, y, L = data
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(max_iter=2000).fit(sc.transform(X), y)
    np.savez(tmp_path / "m.npz", coef=lr.coef_, intercept=lr.intercept_,
             classes=np.array(["mbpp", "gsm8k", "math"]), mean=sc.mean_, scale=sc.scale_)
    (tmp_path / "m.json").write_text(json.dumps({
        "classes": ["mbpp", "gsm8k", "math"], "coef": lr.coef_.tolist(),
        "intercept": lr.intercept_.tolist(),
        "scaler": {"mean": sc.mean_.tolist(), "scale": sc.scale_.tolist()}}))
    a = LRClassifier.load(tmp_path / "m.npz", l0=L0, n_experts=E, priors=PRI)
    b = LRClassifier.load(tmp_path / "m.json", l0=L0, n_experts=E, priors=PRI)
    ref = Pipeline([("s", sc), ("lr", lr)])
    _agree(a, ref, L[::13], X[::13])
    _agree(b, ref, L[::13], X[::13])


def test_wire_roundtrip_predicts_identically(tmp_path, data) -> None:
    """节点上还原出来的识别器，必须与控制机上装的那个打一样的分。"""
    X, y, L = data
    est = Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(max_iter=2000))]).fit(X, y)
    clf = LRClassifier.load(_save(tmp_path, est), l0=L0, n_experts=E, priors=PRI,
                            class_map=CMAP)
    node = make_classifier(json.loads(json.dumps(clf.to_wire())))
    assert isinstance(node, LRClassifier)
    for lay in L[::5]:
        a, b = clf.predict_stats(_st(lay)), node.predict_stats(_st(lay))
        assert a.task == b.task and abs(a.confidence - b.confidence) < 1e-12


def test_old_histogram_wire_still_loads() -> None:
    """没有 kind 的老格式 = 直方图识别器，节点照旧认。"""
    h = HistogramClassifier({"A": np.ones(4), "B": np.arange(4.0)}, {"A": 1, "B": 1})
    got = make_classifier(h.to_wire())
    assert isinstance(got, HistogramClassifier)
    assert got.predict_stats(MoEStats(hist=np.ones(4))).task == "A"


# --------------------------------------------------------------------------- #
# 3. 特征拼法与核对
# --------------------------------------------------------------------------- #
def test_feature_layout_and_stats_by_hand() -> None:
    rng = np.random.default_rng(3)
    lay = _rand_layers(rng, 2, e=4, tokens=5)
    c = np.stack([lay[1].count, lay[2].count]).astype(float)
    w = np.stack([lay[1].mass, lay[2].mass])
    mw = np.divide(w, c, out=np.zeros_like(w), where=c > 0)

    def feat(**kw):
        coef = np.zeros((2, 100))
        spec = FeatureSpec(**kw)
        d = spec.expected_dim(2, 4)
        clf = LRClassifier(classes=["a", "b"], coef=np.zeros((2, d)), intercept=np.zeros(2),
                           proba="softmax", spec=spec, l0=2, n_experts=4,
                           priors={"a": 1, "b": 1})
        del coef
        return clf.features(lay)

    assert np.allclose(feat(), c.reshape(-1))
    assert np.allclose(feat(stats=["freq"]), (c / 5).reshape(-1))
    assert np.allclose(feat(stats=["count"], normalize="per_layer"),
                       (c / c.sum(1, keepdims=True)).reshape(-1))
    assert np.allclose(feat(stats=["count", "mean_weight"]),
                       np.concatenate([c, mw], axis=1).reshape(-1))     # 层优先
    assert np.allclose(feat(stats=["count", "mean_weight"], layout="stat_major"),
                       np.concatenate([c.reshape(-1), mw.reshape(-1)]))
    assert np.allclose(feat(log1p=True), np.log1p(c).reshape(-1))
    v = feat(vector_norm="l2")
    assert abs(np.linalg.norm(v) - 1) < 1e-12
    # 只取第 1 层（0-based），即规划器的第 2 层
    assert np.allclose(feat(layers=[1]), c[1])


def test_dimension_mismatch_says_how_many_layers(tmp_path, data) -> None:
    """用 3 层训练的模型装到 L₀=5 上 —— 报错要直接说出「像是 3 层训练的」。"""
    X, y, _ = data
    est = LogisticRegression(max_iter=500).fit(X, y)
    with pytest.raises(ValueError, match=r"3 层") as ei:
        LRClassifier.load(_save(tmp_path, est), l0=5, n_experts=E, priors=PRI,
                          class_map=CMAP)
    assert "--lr-features" in str(ei.value)
    # 而显式指定只用前 3 层（0-based 0,1,2）就装得上
    LRClassifier.load(_save(tmp_path, est), l0=5, n_experts=E, priors=PRI,
                      class_map=CMAP, features='{"layers": [0, 1, 2]}')


def test_layers_beyond_the_front_are_rejected(tmp_path, data) -> None:
    X, y, _ = data
    est = LogisticRegression(max_iter=500).fit(X, y)
    with pytest.raises(ValueError, match="前段只有"):
        LRClassifier.load(_save(tmp_path, est), l0=2, n_experts=E, priors=PRI,
                          class_map=CMAP, features='{"layers": [0, 1, 2]}')


def test_integer_classes_need_a_map(tmp_path, data) -> None:
    X, y, _ = data
    est = LogisticRegression(max_iter=500).fit(X, y)
    with pytest.raises(ValueError, match="--lr-classes"):
        LRClassifier.load(_save(tmp_path, est), l0=L0, n_experts=E, priors=PRI)
    assert parse_class_map("a,b,c", [0, 1, 2]) == {"0": "a", "1": "b", "2": "c"}
    assert parse_class_map("2=z", [0, 1, 2]) == {"0": "0", "1": "1", "2": "z"}
    with pytest.raises(ValueError, match="不是模型的类别"):
        parse_class_map("7=z", [0, 1, 2])


def test_undeployed_class_goes_to_the_low_confidence_path(tmp_path, data) -> None:
    """模型认识 math，但没给 math 建后段：判给它的请求没有专属池。

    框架里没有全量兜底池（III.7.5），于是走低置信那条路 —— 绑最大先验池、
    全程监控。**置信度不在已部署的池之间重归一**：模型把 99% 给了 math 时，
    mbpp 那 0.01% 不能被放大成「很确定是 mbpp」。"""
    X, y, L = data
    est = LogisticRegression(max_iter=2000).fit(X, y)
    clf = LRClassifier.load(_save(tmp_path, est), l0=L0, n_experts=E,
                            priors={"mbpp": 5, "gsm8k": 3}, class_map=CMAP)
    v = clf.predict_stats(_st(L[-1]))                      # 一条 math
    assert v.zone == "prior" and v.task == "mbpp"          # 最大先验池
    assert set(v.scores) == {"mbpp", "gsm8k"}
    assert sum(v.scores.values()) < 0.5, "不该在已部署的池之间重归一"
    assert set(v.raw) == {"mbpp", "gsm8k", "math"}         # 下标类别报 task 名
    assert max(v.raw, key=v.raw.get) == "math"
    assert v.note and "math" in v.note and not v.error


def test_classes_can_be_merged_into_one_pool(tmp_path, data) -> None:
    """`--lr-classes 2=gsm8k`：math 并进 gsm8k 池，两者概率相加。"""
    X, y, L = data
    est = LogisticRegression(max_iter=2000).fit(X, y)
    clf = LRClassifier.load(_save(tmp_path, est), l0=L0, n_experts=E,
                            priors={"mbpp": 5, "gsm8k": 3},
                            class_map="0=mbpp,1=gsm8k,2=gsm8k")
    v = clf.predict_stats(_st(L[-1]))                      # 一条 math
    assert v.task == "gsm8k" and v.zone == "commit"
    p = clf.proba_raw(clf.features(L[-1]))
    assert v.confidence == pytest.approx(p[1] + p[2])
    assert set(v.raw) == {"mbpp", "gsm8k#1", "gsm8k#2"}


def test_missing_layer_falls_back_to_prior_with_a_reason(tmp_path, data) -> None:
    """捎带链路丢了一层：不能悄悄补零给出一个像样的分数，也不能让请求卡死。"""
    X, y, L = data
    est = LogisticRegression(max_iter=500).fit(X, y)
    clf = LRClassifier.load(_save(tmp_path, est), l0=L0, n_experts=E, priors=PRI,
                            class_map=CMAP)
    lay = dict(L[0])
    del lay[2]
    v = clf.predict_stats(_st(lay))
    assert v.zone == "prior" and v.task == "mbpp"          # 最大先验池
    assert v.error and "缺第 [2] 层" in v.error


def test_custom_feature_function(tmp_path, data) -> None:
    X, y, L = data
    Xl = np.log1p(X)
    est = LogisticRegression(max_iter=2000).fit(Xl, y)
    fn = tmp_path / "feat.py"
    fn.write_text("def f(count, weight, tokens, layers):\n"
                  "    return np.log1p(count).reshape(-1)\n")
    clf = LRClassifier.load(_save(tmp_path, est), l0=L0, n_experts=E, priors=PRI,
                            class_map=CMAP, feature_fn=f"{fn}:f")
    node = make_classifier(json.loads(json.dumps(clf.to_wire())))   # 源码随清单走
    _agree(node, est, L[::8], Xl[::8])
    fn.write_text("def f(count, weight, tokens, layers):\n    return count[:1].reshape(-1)\n")
    with pytest.raises(ValueError, match="特征函数输出"):
        LRClassifier.load(_save(tmp_path, est), l0=L0, n_experts=E, priors=PRI,
                          class_map=CMAP, feature_fn=f"{fn}:f")


def test_inspect_cli(tmp_path, data, capsys) -> None:
    from p2pmoe.deploy.lr_tool import main as _main

    X, y, _ = data
    p = _save(tmp_path, LogisticRegression(max_iter=500).fit(X, y))
    assert _main(["inspect", "--model", str(p), "--l0", "3", "--n-experts", str(E),
                  "--lr-classes", CMAP, "--tasks", "mbpp=5,gsm8k=3"]) == 0
    out = capsys.readouterr().out
    assert "48 维" in out and "math" in out


# --------------------------------------------------------------------------- #
# 4. 端到端：真进程
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def toy_plan():
    import examples.e2e as e2e
    from p2pmoe.planner.experts import build_placement, union_placement
    from p2pmoe.planner.network import MeasurementCache
    from p2pmoe.planner.pipeline import plan
    from p2pmoe.planner.types import PlannerConfig, TaskProfile
    from p2pmoe.runtime.corpus import make_corpus, profile_from_corpus, sample_prompt
    from p2pmoe.runtime.wire import LinkTable
    from p2pmoe.sim.network import SimNetwork

    cfg = ToyMoEConfig()
    corpus = make_corpus(cfg, ["X", "Y", "Z"], seed=3, shared_clusters=0)
    profiles = profile_from_corpus(cfg, corpus)
    plcs = {u: build_placement(p, 0.95) for u, p in profiles.items()}
    uni = union_placement(list(plcs.values()))
    spec = e2e.toy_model_spec(cfg)
    nodes = e2e.build_pool()
    tasks = [TaskProfile(name=u, lam=l, experts_per_layer=plcs[u].as_experts_per_layer(),
                         placement=plcs[u]) for u, l in e2e.TASKS]
    sim = SimNetwork([n.id for n in nodes], seed=3, good_access=(12.0, 16.0),
                     bad_access=(28.0, 33.0), bad_frac=0.2, backbone=(2.0, 5.0),
                     jitter=(4.0, 9.0))
    pc = PlannerConfig(eta=0.15, beta=1.3, j_cap_ms=30.0, theta=0.8,
                       kappa_over=0.3, n_standby=0, seed=3)
    net = MeasurementCache(sim, k=pc.k_probe, j_cap_ms=pc.j_cap_ms, k_gate=pc.k_gate)
    res = plan(nodes, spec, tasks, uni, net, pc,
               {l: 0.70 + 0.05 * l for l in range(2, cfg.n_layers)}, p_min=0.80)
    assert res.manifest is not None and res.manifest.ok
    links = LinkTable(
        p50={(a, b): sim.true_p50(a, b) for a in sim.node_ids for b in sim.node_ids if a != b},
        jitter={(a, b): sim.true_jitter(a, b) for a in sim.node_ids for b in sim.node_ids
                if a != b},
        scale=0.05,
    )

    # 「外面训练」：在线配置（前段只装并集）下跑语料，按默认口径拼特征，训 LR
    front = SegmentModel(cfg, {l: uni.at(l) for l in range(1, res.l0 + 1)})
    X, y = [], []
    for u in ("X", "Y", "Z"):
        for j in range(20):
            _, st = front.forward(f"{u}{j}", embed_tokens(
                cfg, sample_prompt(corpus, u, 12, seed=3000 + j)))
            X.append(np.concatenate([st.layers[l].count for l in range(1, res.l0 + 1)]))
            y.append(u)
            front.drop_kv(f"{u}{j}")
    est = Pipeline([("s", StandardScaler()),
                    ("lr", LogisticRegression(max_iter=3000, C=0.3))]).fit(np.array(X, float), y)
    return dict(cfg=cfg, corpus=corpus, res=res, links=links, est=est,
                tasks=dict(e2e.TASKS), sample=sample_prompt)


@pytest.mark.slow
def test_e2e_dynamic_lr_drives_binding_and_dump_rechecks(toy_plan, tmp_path) -> None:
    from p2pmoe.deploy.control import _dump_front_stats
    from p2pmoe.runtime.coordinator import LocalCluster
    from p2pmoe.deploy.lr_tool import main as _main

    t = toy_plan
    mp = _save(tmp_path, t["est"])
    clf = LRClassifier.load(mp, l0=t["res"].l0, n_experts=t["cfg"].n_experts,
                            priors=t["tasks"])
    base = {u: 1.0 for u in t["tasks"]}         # 不想让通道二在这里插手
    recs = []
    with LocalCluster(t["res"].manifest, t["cfg"], t["links"], clf, report_layers=True,
                      baselines=base, priors=t["tasks"], alarm_factor=50.0) as cl:
        cl.coord.max_tokens = 4
        for i, u in enumerate(["X", "Y", "Z", "X", "Y", "Z"]):
            r = cl.coord.submit(f"r{i}", t["sample"](t["corpus"], u, 12, seed=7000 + i),
                                true_task=u)
            assert r.done.wait(timeout=120), f"r{i} 没跑完"
            recs.append(r)
        assert not cl.coord.errors, cl.coord.errors[:2]

    assert all(r.clf_kind == "lr" for r in recs), "识别结果不是 LR 给的"
    assert all(r.raw_scores for r in recs)
    assert sum(r.task == r.true_task for r in recs) >= 5, \
        [(r.true_task, r.task, round(r.conf, 2)) for r in recs]
    # 逐层统计带回来了，且恰好是前 L₀ 层
    assert all(sorted(int(k) for k in r.front_layers) == list(range(1, t["res"].l0 + 1))
               for r in recs)

    # 落盘 → 离线重算：与在线结果一致（证明节点上拼的特征与这里一样）
    dump = tmp_path / "front.jsonl"
    _dump_front_stats(dump, recs, t["res"].l0, t["cfg"].n_experts)
    rows = [json.loads(x) for x in dump.read_text().splitlines()]
    assert len(rows) == len(recs)
    for row, r in zip(rows, recs):
        st = MoEStats(hist=np.zeros(t["cfg"].n_experts), layers={
            int(l): LayerStat.from_wire(s) for l, s in row["layers"].items()})
        assert clf.predict_stats(st).task == r.task
    assert _main(["check", "--model", str(mp), "--dump", str(dump)]) == 0


@pytest.mark.slow
def test_e2e_static_mode_reports_shadow_only(toy_plan, tmp_path) -> None:
    """静态模式下 LR 只旁路：配对照旧定死，识别结果单独上报。"""
    from p2pmoe.runtime.coordinator import LocalCluster

    t = toy_plan
    man = t["res"].manifest
    fronts = sorted(s for s, d in man.segments.items() if d["role"] == "front")
    backs = sorted((s, d["task"]) for s, d in man.segments.items()
                   if d["role"].startswith("back"))
    wiring = {f: b for f, b in zip(fronts, backs)}
    clf = LRClassifier.load(_save(tmp_path, t["est"]), l0=t["res"].l0,
                            n_experts=t["cfg"].n_experts, priors=t["tasks"])
    with LocalCluster(man, t["cfg"], t["links"], clf, static_wiring=wiring,
                      priors=t["tasks"]) as cl:
        cl.coord.max_tokens = 3
        u = wiring[fronts[0]][1]
        r = cl.coord.submit("s0", t["sample"](t["corpus"], u, 12, seed=8100),
                            true_task=u, task=u)
        assert r.done.wait(timeout=120)
        assert not cl.coord.errors, cl.coord.errors[:2]
    assert r.task == u, "静态模式下配对不能被识别结果改动"
    assert r.shadow is not None and r.shadow["task"] in t["tasks"]
    assert r.clf_kind is None, "静态模式不走 classified 那条路"


# --------------------------------------------------------------------------- #
# 多跳前段：逐层统计必须跨机器捎带到 tail
# --------------------------------------------------------------------------- #
def test_layers_hop_across_a_multi_node_front() -> None:
    """真机的前段是多跳的（L₀=11 时 F0 = N01→N07→N10）。toy 规划出来的前段是
    单节点，统计不用跨机器 —— 那条端到端测试因此看不见捎带链路断掉。这里
    手搭一条两跳前段：n1 算 1–2 层，n2 算 3–4 层并在出口做 LR。"""
    from p2pmoe.runtime.node import NodeConfig, NodeServer
    from p2pmoe.runtime.wire import LinkTable

    mc = ToyMoEConfig()
    allx = list(range(mc.n_experts))
    rng = np.random.default_rng(5)
    clf = LRClassifier(classes=["X", "Y"], coef=rng.normal(size=(2, 4 * mc.n_experts)),
                       intercept=np.zeros(2), proba="softmax", spec=FeatureSpec(),
                       l0=4, n_experts=mc.n_experts, priors={"X": 1, "Y": 1})

    def mk(nid, layers, nxt, tail):
        s = NodeServer(nid, host="127.0.0.1", port=0)
        s.apply_config(NodeConfig(
            node_id=nid, role="front", segment="F0",
            layer_experts={l: allx for l in layers}, next_hop=nxt, seg_head="n1",
            is_head=nid == "n1", is_tail=tail, peers={}, links=LinkTable().to_dict(),
            coordinator=("127.0.0.1", 1), model={},
            classifier=clf.to_wire() if tail else None))
        sent, rep = [], []
        s.pool.send = lambda dest, h, arr=None, **k: sent.append((dest, h, arr))
        s._report = rep.append
        return s, sent, rep

    n1, sent1, _ = mk("n1", (1, 2), "n2", False)
    n2, _, rep2 = mk("n2", (3, 4), None, True)

    n1._on_prefill({"req": "r", "ids": list(range(12))})
    (dest, h, arr), = sent1
    assert dest == "n2" and h["type"] == "hop"
    assert sorted(h["stats"]["lay"]) == ["1", "2"], "前段第一跳没把逐层统计带上"
    n2._on_hop(json.loads(json.dumps(h)), arr)

    (msg,) = [m for m in rep2 if m["type"] == "classified"]
    assert msg["clf"] == "lr"
    assert "clf_error" not in msg, msg.get("clf_error")
    assert msg["zone"] != "prior" or msg["conf"] > 0

    # decode 步：不付这份字节
    sent1.clear()
    n1._on_loop({"req": "r", "token": 3})
    assert "lay" not in sent1[0][1]["stats"]
