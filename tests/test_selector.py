"""task 选择器（selector/selector_body_L8）接进框架。

选择器只有在特征与训练时**逐字一致**时才有意义（SELECTOR_USAGE.md 第 3、6 节）。
这里按那几条约束逐条钉住：

1. **模型文件** —— joblib（sklearn Pipeline + 元数据）与 _params.npz 两种格式都直接
   能装，特征口径、类别名、层数从文件里读，不用再手配；概率与 sklearn 一致。
2. **只数 body token** —— prompt 按 PromptEncoder 的方式构造（一条 user 消息、
   sentinel 切模板、trim 检测、offset 重叠规则），body 区间随 prefill 下发，
   前段每一跳只对这些位置计数。收到整条 prompt 的计数时拒绝打分。
3. **前段要装全** —— 选择器按完整模型的路由训练，前 8 层缺专家会让特征偏离。
4. **没部署的类** —— 判给 no_robots 这类没有后段的类时，走低置信那条路。
5. **端到端** —— 控制面 → 协调器 → torch 节点 → LR，真进程跑一遍。
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from p2pmoe.runtime.lr_classifier import LRClassifier, load_lr_params
from p2pmoe.runtime.model import LayerStat, MoEStats, SegmentModel, ToyMoEConfig, embed_tokens

SEL_DIR = ROOT / "selector"
JOBLIB = SEL_DIR / "selector_body_L8.joblib"
NPZ = SEL_DIR / "selector_body_L8_params.npz"
FIX = ROOT / "tests" / "data" / "selector_fixture.npz"
TASKS5 = ["gsm8k", "math", "mbpp", "humaneval", "no_robots"]


def _st(counts: np.ndarray, masked: bool = True) -> MoEStats:
    """[8, 128] 计数 → 前段 tail 收到的逐层统计（规划器层号 1..8）。"""
    return MoEStats(hist=np.zeros(counts.shape[1]), layers={
        l + 1: LayerStat(counts[l].astype(np.int64), np.zeros(counts.shape[1]),
                         int(counts[l].sum() // 8), masked)
        for l in range(counts.shape[0])})


# --------------------------------------------------------------------------- #
# 1. 模型文件
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module", params=["joblib", "npz"])
def sel(request):
    path = JOBLIB if request.param == "joblib" else NPZ
    if not path.exists():
        pytest.skip(f"没有 {path}")
    if request.param == "joblib":
        pytest.importorskip("sklearn")
    return LRClassifier.load(path, l0=8, n_experts=128,
                             priors={u: 1.0 for u in TASKS5})


def test_the_file_describes_itself(sel) -> None:
    """特征口径、层数、类别名都来自文件 —— 不用 --lr-features / --lr-classes。"""
    assert sel.labels == TASKS5
    assert sel.spec.layers == list(range(8)) and sel.spec.layer_base == 0
    assert sel.spec.stats == ["count"] and sel.spec.normalize == "per_layer"
    assert sel.needs_body
    assert sel.coef.shape == (5, 1024)
    assert sel.tau_hi == 0.9, "选择器默认用 SELECTOR_USAGE 推荐的 0.9"


def test_matches_sklearn_on_real_training_samples(sel) -> None:
    """60 条真实训练样本（5 类各 12 条，counts_body 前 8 层），参考概率是
    selector_body_L8.joblib 的 sklearn pipeline 算的。"""
    z = np.load(FIX)
    for c, ref in zip(z["counts_body"], z["ref_proba"]):
        v = sel.predict_stats(_st(c))
        got = np.array([v.raw[u] for u in TASKS5])
        assert np.abs(got - ref).max() < 1e-6
        assert TASKS5[int(np.argmax(got))] == TASKS5[int(np.argmax(ref))]
    acc = np.mean([sel.predict_stats(_st(c)).task == y
                   for c, y in zip(z["counts_body"], z["labels"])])
    assert acc > 0.95


def test_load_lr_params_reads_the_class_names() -> None:
    if not NPZ.exists():
        pytest.skip("没有 _params.npz")
    raw = load_lr_params(NPZ)
    assert raw["classes"] == TASKS5
    assert raw["meta"]["n_layers"] == 8 and raw["meta"]["n_experts"] == 128


def test_wrong_model_is_rejected() -> None:
    """Qwen3-Next 是 512 专家 —— 拿 128 专家的选择器去装，要当场拒绝。"""
    if not NPZ.exists():
        pytest.skip("没有 _params.npz")
    with pytest.raises(ValueError, match="128 个专家"):
        LRClassifier.load(NPZ, l0=8, n_experts=512, priors={"mbpp": 1})


def test_front_shorter_than_8_layers_is_rejected() -> None:
    if not NPZ.exists():
        pytest.skip("没有 _params.npz")
    with pytest.raises(ValueError, match="--min-l0 8"):
        LRClassifier.load(NPZ, l0=6, n_experts=128, priors={"mbpp": 1})


# --------------------------------------------------------------------------- #
# 2/4. 打分口径
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def deployed():
    if not NPZ.exists():
        pytest.skip("没有 _params.npz")
    return LRClassifier.load(NPZ, l0=8, n_experts=128, priors={"mbpp": 5, "gsm8k": 3})


def test_whole_prompt_counts_are_refused(deployed) -> None:
    """整条 prompt（含对话模板）的计数喂给按 body 训练的模型：分数看起来照样像
    回事，但特征整体偏了。必须拒绝，而不是给一个假分数。"""
    z = np.load(FIX)
    v = deployed.predict_stats(_st(z["counts_body"][0], masked=False))
    assert v.zone == "prior" and "body" in v.error


def test_undeployed_class_goes_to_prior(deployed) -> None:
    z = np.load(FIX)
    nr = z["counts_body"][list(z["labels"]).index("no_robots")]
    v = deployed.predict_stats(_st(nr))
    assert max(v.raw, key=v.raw.get) == "no_robots"
    assert v.zone == "prior" and v.task == "mbpp"           # 最大先验池
    assert "no_robots" in v.note


def test_merged_pools(deployed) -> None:
    """humaneval 并进 mbpp、math 并进 gsm8k：池的置信 = 类概率之和。"""
    sel = LRClassifier.load(NPZ, l0=8, n_experts=128, priors={"mbpp": 5, "gsm8k": 3},
                            class_map="humaneval=mbpp,math=gsm8k")
    z = np.load(FIX)
    he = z["counts_body"][list(z["labels"]).index("humaneval")]
    v = sel.predict_stats(_st(he))
    assert v.task == "mbpp"
    assert v.confidence == pytest.approx(v.raw["mbpp"] + v.raw["humaneval"])


# --------------------------------------------------------------------------- #
# 2. body 区间：prompt 构造与 PromptEncoder 逐 token 一致
# --------------------------------------------------------------------------- #
QWEN3_LIKE = (
    "{%- for message in messages %}{{- '<|im_start|>' + message.role + '\\n' + "
    "message.content{TRIM} + '<|im_end|>' + '\\n' }}{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}"
    "{%- if enable_thinking is defined and enable_thinking is false %}"
    "{{- '<think>\\n\\n</think>\\n\\n' }}{%- endif %}{%- endif %}")
SPECIALS = ["<|im_start|>", "<|im_end|>", "<think>", "</think>", "<|endoftext|>"]

CASES = [
    # gsm8k：任务指令模板 + 题目
    [("Solve the following math problem. Reason step by step, and put your final "
      "answer after '#### '.\n\nProblem: ", True), ("Janet's ducks lay 16 eggs.", False)],
    # mbpp：两段 body 中间夹模板
    [("You are an expert Python programmer. Write a Python function for the following "
      "task. Only output the code.\n\nTask: ", True),
     ("Write a python function to reverse a list.", False),
     ("\nYour code should pass these tests:\n", True),
     ("assert f([1,2]) == [2,1]\nassert f([]) == []", False)],
    # humaneval：以换行结尾（会 trim 的模板下要同步剥）
    [("You are an expert Python programmer. Complete the following function. Only "
      "output the complete function.\n\n", True),
     ('def f(x: str) -> List[str]:\n    """ doc\n    """\n', False)],
    [("   leading and trailing  \n\n", False)],
    [("Be brief.\n\n", True), ("Write a poem 北京", False)],
]


def _tokenizer():
    tk = pytest.importorskip("tokenizers")
    from tokenizers import decoders, models, pre_tokenizers, processors, trainers

    tok = tk.Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    corpus = ["".join(t for c in CASES for t, _ in c)] * 40
    tok.train_from_iterator(corpus, trainers.BpeTrainer(
        vocab_size=500, special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    return tok


def _textio(tok, trim: bool):
    from p2pmoe.runtime.text import TextIO, Tokenizer

    tpl = QWEN3_LIKE.replace("{TRIM}", " | trim" if trim else "")
    return TextIO(tok=Tokenizer(tok, config={"chat_template": tpl}), chat=True,
                  template_kw={"enable_thinking": False}), tpl


def _mask(n, ranges):
    m = [False] * n
    for a, b in ranges:
        m[a:b] = [True] * (b - a)
    return m


@pytest.mark.parametrize("trim", [False, True])
def test_body_tokens_by_construction(trim) -> None:
    """不依赖外部代码的性质：body 里没有模板的字、没有特殊 token；
    非思考模式的空 <think> 在；trim 模板下两端空白被剥掉。"""
    tok = _tokenizer()
    tio, _ = _textio(tok, trim)
    for segs in CASES:
        bp = tio.encode_body(segs)
        assert bp.text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        body_ids = [i for i, m in zip(bp.ids, _mask(len(bp.ids), bp.body)) if m]
        body_txt = tok.decode(body_ids, skip_special_tokens=False)
        assert not any(s in body_txt for s in SPECIALS)
        for t, is_tpl in segs:
            if not is_tpl and t.strip():
                assert t.strip()[:10] in body_txt
            if is_tpl and len(t.strip()) > 12:
                assert t.strip()[5:15] not in body_txt, "模板的字算进了 body"
        assert bp.trimmed == (trim and segs[0][0][0].isspace()
                              or trim and segs[-1][0][-1].isspace())


def _probe_module():
    """训练端的 moe_prefill_probe.py —— 不是 pip 包，是旁边 verify/ 目录里的脚本，
    所以按路径导入（它也不该进 requirements）。"""
    import importlib

    vd = _verify_dir()
    if vd is None:
        pytest.skip("找不到 moe_prefill_probe.py（设 P2P_VERIFY_DIR）")
    sys.path.insert(0, str(vd))
    return importlib.import_module("moe_prefill_probe")


def _verify_dir() -> Path | None:
    for c in (os.environ.get("P2P_VERIFY_DIR"), ROOT.parent / "verify"):
        if c and (Path(c) / "moe_prefill_probe.py").exists():
            return Path(c)
    return None


@pytest.mark.parametrize("trim", [False, True])
def test_body_tokens_match_prompt_encoder(trim) -> None:
    """与选择器训练端的 moe_prefill_probe.PromptEncoder 逐 token 比：ids、全文、body 掩码。

    要那份代码在旁边（../verify/ 或 P2P_VERIFY_DIR）才跑 —— 它不在本仓库里。
    """
    mpp = _probe_module()
    pytest.importorskip("transformers")
    PromptEncoder, _Rec = mpp.PromptEncoder, mpp._Rec
    from transformers import PreTrainedTokenizerFast

    tok = _tokenizer()
    tio, tpl = _textio(tok, trim)
    hf = PreTrainedTokenizerFast(tokenizer_object=tok, chat_template=tpl,
                                 pad_token="<|endoftext|>", padding_side="left")
    enc = PromptEncoder(hf, ("full", "body"), (), enable_thinking=False)
    for segs in CASES:
        assert enc.verify("".join(s for s, _ in segs)), enc.last_diff
        text, spans = enc.build(segs)
        b, m = enc.encode_batch([_Rec(0, text, spans, 0)])
        bp = tio.encode_body(segs)
        assert bp.text == text
        assert bp.ids == b["input_ids"][0].tolist()
        assert _mask(len(bp.ids), bp.body) == m["body"][0].tolist()


def test_body_ranges_rule() -> None:
    """重叠即算、空区间（特殊 token）不算、相邻的合并。"""
    from p2pmoe.runtime.text import body_ranges

    offs = [(0, 0), (0, 5), (5, 9), (9, 12), (12, 12), (12, 20)]
    assert body_ranges(offs, [(7, 15)]) == [(2, 4), (5, 6)]
    assert body_ranges(offs, []) == []


# --------------------------------------------------------------------------- #
# 2. body 区间沿前段多跳传下去
# --------------------------------------------------------------------------- #
def test_body_mask_rides_along_a_multi_node_front() -> None:
    """head(f) 拿到 cmask，随 hop 交给下一台；每台都只对 body 位置计数。
    decode 不带。"""
    from p2pmoe.runtime.node import NodeConfig, NodeServer
    from p2pmoe.runtime.wire import LinkTable

    mc = ToyMoEConfig()
    allx = list(range(mc.n_experts))

    def mk(nid, layers, nxt, tail):
        s = NodeServer(nid, host="127.0.0.1", port=0)
        s.apply_config(NodeConfig(
            node_id=nid, role="front", segment="F0",
            layer_experts={l: allx for l in layers}, next_hop=nxt, seg_head="n1",
            is_head=nid == "n1", is_tail=tail, peers={}, links=LinkTable().to_dict(),
            coordinator=("127.0.0.1", 1), model={}, report_layers=tail))
        sent, rep = [], []
        s.pool.send = lambda dest, h, arr=None, **k: sent.append((dest, h, arr))
        s._report = rep.append
        return s, sent, rep

    n1, sent1, _ = mk("n1", (1, 2), "n2", False)
    n2, _, rep2 = mk("n2", (3, 4), None, True)
    ids = list(range(12))
    body = [[3, 6], [8, 11]]
    n1._on_prefill({"req": "r", "ids": ids, "cmask": body})
    (_, h, arr), = sent1
    assert h["cmask"] == body
    n2._on_hop(json.loads(json.dumps(h)), arr)
    (msg,) = [m for m in rep2 if m["type"] == "classified"]
    lay = msg["layers"]
    assert sorted(lay) == ["1", "2", "3", "4"]
    for l, s in lay.items():
        assert s["t"] == 6 and s.get("m") == 1, f"第 {l} 层没按 body 计数"
        assert sum(s["c"]) == 6 * mc.top_k

    # 同一输入不带掩码：计数是全部 12 个 token 的，且计算本身不受掩码影响
    m = SegmentModel(mc, {l: allx for l in (1, 2, 3, 4)})
    x = embed_tokens(mc, ids)
    y0, st0 = m.forward("a", x)
    mask = np.zeros(12, bool)
    mask[3:6] = mask[8:11] = True
    y1, st1 = m.forward("b", x, count_mask=mask)
    assert np.allclose(y0, y1) and np.allclose(st0.hist, st1.hist)
    assert st0.layers[1].n_tokens == 12 and st1.layers[1].n_tokens == 6

    sent1.clear()
    n1._on_loop({"req": "r", "token": 3})
    assert "cmask" not in sent1[0][1]


def test_coordinator_sends_body_only_in_body_mode() -> None:
    from p2pmoe.runtime.coordinator import Coordinator, RequestRecord

    rec = RequestRecord(req="r", true_task=None, ids=[1, 2, 3])
    assert "cmask" not in Coordinator._prefill_msg(None, rec)
    rec.body = [(1, 3)]
    assert Coordinator._prefill_msg(None, rec)["cmask"] == [[1, 3]]


# --------------------------------------------------------------------------- #
# 测试集 JSONL
# --------------------------------------------------------------------------- #
def test_jsonl_cases(tmp_path) -> None:
    from p2pmoe.deploy.control import load_cases

    f = tmp_path / "c.jsonl"
    f.write_text("\n".join([
        json.dumps({"task": "gsm8k", "text": "48 个朋友"}),
        "# 注释",
        json.dumps({"task": "gsm8k", "segments": [["Problem: ", True], ["x", False]]}),
        json.dumps({"task": "no_robots", "text": "poem", "system": "Be brief."}),
        json.dumps({"text": "无标注"}),
    ]), encoding="utf-8")
    got = load_cases(f, ["gsm8k", "mbpp"])
    assert got[0] == ("gsm8k", "48 个朋友")
    assert got[1] == ("gsm8k", {"segments": [("Problem: ", True), ("x", False)],
                                "system": None})
    assert got[2] == ("no_robots", {"segments": [("poem", False)], "system": "Be brief."})
    assert got[3] == (None, "无标注")


# --------------------------------------------------------------------------- #
# 3. 规划：probe 统计 → CSV / 画像
# --------------------------------------------------------------------------- #
def test_probe_to_data_roundtrip(tmp_path) -> None:
    import examples.probe_to_data as p2d
    from p2pmoe.deploy.control import _load_profile_file
    from p2pmoe.planner.from_data import load_activation_csv

    rng = np.random.default_rng(1)
    L, E, k = 4, 16, 2
    probe = tmp_path / "probe"
    probe.mkdir()
    for u in ("aa", "bb"):
        n, T = 5, 7
        idx = rng.integers(0, E, size=(n, L, T, k))
        cnt = np.zeros((n, L, E), np.int32)
        for a in range(n):
            for l in range(L):
                np.add.at(cnt[a, l], idx[a, l].reshape(-1), 1)
        np.savez(probe / f"{u}_prefill_persample.npz", counts_full=cnt,
                 wsum_full=cnt * 0.4, tokens_full=np.full(n, T),
                 counts_body=cnt, wsum_body=cnt * 0.4, tokens_body=np.full(n, T))
        (probe / f"{u}_prefill_meta.json").write_text(json.dumps({"top_k": k, "model": "/m/X"}))
    out = tmp_path / "out"
    assert p2d.main(["--probe-dir", str(probe), "--out", str(out)]) == 0
    prof, _ = load_activation_csv(out / "aa_expert_activation.csv", task="aa", phase="prefill")
    assert (prof.n_layers, prof.n_experts) == (L, E)
    with pytest.raises(ValueError, match="全为 0"):
        load_activation_csv(out / "aa_expert_activation.csv", task="aa", phase="decode")
    plcs, profs = _load_profile_file(out / "profile_prefill.json", ["aa", "bb"], L, E,
                                     coverage=0.7, top_k=k)
    assert set(plcs) == {"aa", "bb"} and profs is not None


def test_control_warns_when_the_front_is_not_full() -> None:
    src = (ROOT / "p2pmoe/deploy/control.py").read_text(encoding="utf-8")
    assert "前段没装全识别器要看的层" in src and "--front-full" in src


# --------------------------------------------------------------------------- #
# 5. 端到端：控制面 → 协调器 → torch 节点 → LR
# --------------------------------------------------------------------------- #
def _tiny_qwen3(d: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from p2pmoe.sim.fake_checkpoint import TINY_QWEN3_MOE, write_fake_checkpoint

    write_fake_checkpoint(d, dict(TINY_QWEN3_MOE, num_hidden_layers=6), seed=3, n_shards=2)
    tok = _tokenizer()
    tok.save(str(d / "tokenizer.json"))
    (d / "tokenizer_config.json").write_text(json.dumps(
        {"chat_template": QWEN3_LIKE.replace("{TRIM}", ""), "eos_token": "<|im_end|>"}))
    (d / "generation_config.json").write_text(json.dumps(
        {"eos_token_id": tok.token_to_id("<|im_end|>")}))


@pytest.mark.slow
@pytest.mark.parametrize("static", [False, True])
def test_e2e_selector_through_the_control_plane(tmp_path, static) -> None:
    from p2pmoe.deploy.control import main
    from p2pmoe.runtime.node import NodeServer

    ck = tmp_path / "ckpt"
    _tiny_qwen3(ck)
    # 选择器格式的 npz（与 _params.npz 同键）：前 2 层 × 8 专家，3 类
    rng = np.random.default_rng(0)
    sel = tmp_path / "sel_params.npz"
    np.savez(sel, mean=rng.random(16), scale=rng.random(16) + .5,
             coef=rng.normal(size=(3, 16)), intercept=np.zeros(3),
             classes=np.array(["gsm8k", "mbpp", "no_robots"]),
             n_layers=np.array(2), n_experts=np.array(8))
    cases = tmp_path / "cases.jsonl"
    cases.write_text("\n".join(json.dumps(x) for x in [
        {"task": "gsm8k", "segments": [CASES[0][0], CASES[0][1]]},
        {"task": "mbpp", "text": "Write a python function to reverse a list."},
        {"task": "no_robots", "text": "Write a poem.", "system": "Be brief."},
    ]))
    srv = [NodeServer(f"v{i}", host="127.0.0.1", port=0) for i in range(13)]
    for s in srv:
        threading.Thread(target=s.serve_forever, daemon=True).start()
    time.sleep(0.3)
    try:
        dump, res = tmp_path / "front.jsonl", tmp_path / "res.json"
        args = ["--agents", ",".join(f"{s.me}=127.0.0.1:{s.port}" for s in srv),
                "--advertise", "127.0.0.1", "--model-dir", str(ck), "--device", "cpu",
                "--chat", "--tasks", "gsm8k=1,mbpp=1", "--requests", "3", "--tokens", "2",
                "--k-probe", "2", "--skip-model-check", "--lr-model", str(sel),
                "--prompts-file", str(cases), "--dump-front-stats", str(dump),
                "--save-results", str(res), "--once"] + (["--static"] if static else [])
        assert main(args) == 0
    finally:
        for s in srv:
            s._stop.set()
    out = json.loads(res.read_text())
    assert "只数 body token" in out["classifier"]
    rows = [json.loads(x) for x in dump.read_text().splitlines()]
    assert len(rows) == 3
    for r, q in zip(rows, out["requests"]):
        assert r["n_body"] < r["n_prompt"], "body 应该比整条 prompt 短（不含对话模板）"
        assert all(s.get("m") == 1 and s["t"] == r["n_body"] for s in r["layers"].values())
        if static:
            assert "shadow" in q and "error" not in q["shadow"]
        else:
            assert q["zone"] in ("commit", "observe", "prior")
    # gsm8k 那条：指令模板不进 body
    assert rows[0]["n_body"] < 12
    # 控制面把思考模式关了：prompt 与按 enable_thinking=False 构造的一模一样
    from p2pmoe.runtime.text import TextIO
    tio = TextIO.from_model_dir(ck, chat=True)
    tio.template_kw["enable_thinking"] = False
    want = tio.encode_body([("Write a python function to reverse a list.", False)])
    assert rows[1]["n_prompt"] == len(want.ids)
    assert [list(r) for r in want.body] == rows[1]["body"]


# --------------------------------------------------------------------------- #
# 三级核对工具
# --------------------------------------------------------------------------- #
def test_training_templates_are_verbatim() -> None:
    """lr_tool 里抄的训练模板必须与 moe_prefill_probe.prompt_segments 逐字一致。"""
    prompt_segments = _probe_module().prompt_segments

    from p2pmoe.deploy.lr_tool import train_segments

    exs = {"gsm8k": {"question": "Q?"}, "math": {"problem": "P?"},
           "humaneval": {"prompt": "def f():\n"},
           "mbpp": {"text": "Write f.", "test_list": ["assert f(1)", "assert f(2)"]}}
    for u, ex in exs.items():
        ref = prompt_segments(u, ex)
        body = "".join(t for t, tpl in ref if not tpl)          # bodies.jsonl 存的就是这个
        got, exact = train_segments(u, body)
        assert exact and got == ref, u


def _probe(tmp_path, counts_by_idx):
    d = tmp_path / "probe"
    d.mkdir()
    idx = sorted(counts_by_idx)
    np.savez(d / "gsm8k_prefill_persample.npz", sample_idx=np.array(idx),
             counts_body=np.stack([counts_by_idx[i] for i in idx]))
    return d


def test_check_compares_counts_against_training(tmp_path, capsys) -> None:
    from p2pmoe.deploy.lr_tool import _compare_counts

    if not NPZ.exists():
        pytest.skip("没有 _params.npz")
    sel = LRClassifier.load(NPZ, l0=8, n_experts=128, priors={"gsm8k": 1})
    z = np.load(FIX)
    c0 = z["counts_body"][0].astype(np.int64)
    full = np.zeros((48, 128), np.int64)
    full[:8] = c0
    d = _probe(tmp_path, {7: full, 9: full})
    off = c0.copy()
    off[3, int(np.argmax(off[3]))] -= 1
    off[3, int(np.argmin(off[3]))] += 1                       # 一次路由翻转
    rows = [{"case_id": "gsm8k:7", "layers": {str(l + 1): LayerStat(c0[l], np.zeros(128), 1, True).to_wire() for l in range(8)}},
            {"case_id": "gsm8k:9", "layers": {str(l + 1): LayerStat(off[l], np.zeros(128), 1, True).to_wire() for l in range(8)}}]
    _compare_counts(rows, d, sel)
    out = capsys.readouterr().out
    assert "1/2 条逐个专家完全相同" in out


def test_cases_roundtrip_keeps_the_sample_id(tmp_path) -> None:
    from p2pmoe.deploy.control import load_cases
    from p2pmoe.deploy.lr_tool import main as lr_main

    d = tmp_path / "probe"
    d.mkdir()
    (d / "gsm8k_bodies.jsonl").write_text(json.dumps({"sample_idx": 5, "n_tok": 9, "body": "Q?"}))
    out = tmp_path / "c.jsonl"
    assert lr_main(["cases", "--probe-dir", str(d), "--n", "1", "--out", str(out)]) == 0
    (lab, payload), = load_cases(out, ["gsm8k"])
    assert lab == "gsm8k" and payload["id"] == "gsm8k:5"
    assert payload["segments"][0][1] is True and payload["segments"][1] == ("Q?", False)
