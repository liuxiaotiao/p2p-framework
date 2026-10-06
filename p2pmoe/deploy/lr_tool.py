"""前段 LR 识别器的命令行：部署前核对模型，部署后拿落盘的逐层统计复核。

    python3 -m p2pmoe.deploy.lr_tool inspect --model task/lr.pkl --l0 11 \\
        --n-experts 512 --lr-classes 0=mbpp,1=gsm8k,2=math --tasks mbpp=5,gsm8k=3
    python3 -m p2pmoe.deploy.lr_tool check --model task/lr.pkl --dump results/front.jsonl

task 选择器（selector_body_L8）的三级核对 —— 从不要集群到要集群：

    # 1. 只要 tokenizer：按训练模板重建每条训练 prompt，token 数与 probe 记录的比
    python3 -m p2pmoe.deploy.lr_tool tokens --model-dir $WEIGHTS --probe-dir $PROBE
    # 2. 从训练样本里抽一批做测试集（带样本编号）
    python3 -m p2pmoe.deploy.lr_tool cases --probe-dir $PROBE --n 4 --out task3/sel_cases.jsonl
    # 3. 在集群上跑完（DUMP_FRONT=...），逐样本比前段收到的逐层计数与训练时的 counts_body
    python3 -m p2pmoe.deploy.lr_tool check --model selector/selector_body_L8_params.npz \
        --dump results/front.jsonl --probe-dir $PROBE

只在控制机上跑，不需要 torch。识别器本身在 `p2pmoe/runtime/lr_classifier.py`。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from ..runtime.lr_classifier import (
    LRClassifier,
    _cmap_str,
    load_lr_params,
    parse_class_map,
)
from ..runtime.model import LayerStat, MoEStats

__all__ = ["main"]


# --------------------------------------------------------------------------- #
# 命令行：部署前核对模型，部署后拿落盘的逐层统计复核
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    from collections import Counter

    ap = argparse.ArgumentParser(
        prog="python3 -m p2pmoe.deploy.lr_tool",
        description="前段 LR 识别器：部署前核对 / 部署后用落盘的逐层统计复核")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--model", required=True, help="LR 模型（pkl/joblib/npz/json）")
        p.add_argument("--lr-classes", default=None)
        p.add_argument("--lr-features", default=None, help="特征描述（JSON 字符串或文件）")
        p.add_argument("--lr-feature-fn", default=None, help="文件.py:函数名")
        p.add_argument("--tasks", default=None, help="部署的 task，如 mbpp=5,gsm8k=3")

    p1 = sub.add_parser("inspect", help="装一遍，打印它会怎么拼特征，维度不对就报错")
    common(p1)
    p1.add_argument("--l0", type=int, required=True)
    p1.add_argument("--n-experts", type=int, default=512)

    p2 = sub.add_parser("check", help="拿 --dump-front-stats 落的 JSONL 重算一遍识别")
    common(p2)
    p2.add_argument("--dump", required=True, type=Path)
    p2.add_argument("--probe-dir", type=Path, default=None,
                    help="moe_prefill_probe 的输出目录：逐样本比逐层计数（要 cases 生成的测试集）")

    p3 = sub.add_parser("tokens", help="只用 tokenizer：重建训练 prompt，与 probe 的 token 数比")
    p3.add_argument("--model-dir", required=True, help="有 tokenizer.json 的目录")
    p3.add_argument("--probe-dir", required=True, type=Path)
    p3.add_argument("--tasks", nargs="+", default=None)
    p3.add_argument("--thinking", choices=("off", "on", "default"), default="off")

    p4 = sub.add_parser("cases", help="从训练样本生成测试集 JSONL（带样本编号，可逐样本复核）")
    p4.add_argument("--probe-dir", required=True, type=Path)
    p4.add_argument("--tasks", nargs="+", default=None)
    p4.add_argument("--n", type=int, default=4, help="每个 task 几条")
    p4.add_argument("--raw", action="store_true",
                    help="去掉任务指令模板、只给原题（模拟真实用户输入；SELECTOR_USAGE 8.2）")
    p4.add_argument("--out", required=True, type=Path)

    a = ap.parse_args(argv)

    if a.cmd == "tokens":
        return _cmd_tokens(a)
    if a.cmd == "cases":
        return _cmd_cases(a)

    def priors_of(cls_names):
        if a.tasks:
            return {k: float(v) for k, v in
                    (x.split("=") for x in a.tasks.split(",") if x.strip())}
        return {c: 1.0 for c in cls_names}

    if a.cmd == "inspect":
        raw = load_lr_params(a.model)
        cmap = parse_class_map(a.lr_classes or _cmap_str(raw.get("classes_map")),
                               raw["classes"])
        clf = LRClassifier.load(a.model, l0=a.l0, n_experts=a.n_experts,
                                priors=priors_of(list(cmap.values())),
                                class_map=a.lr_classes, features=a.lr_features,
                                feature_fn=a.lr_feature_fn)
        print("✓", clf.describe())
        if clf.unrecognised:
            print(f"  ⚠ 部署了 {clf.unrecognised}，但模型不认识它们 —— 这些池只能靠先验/换绑进请求")
        extra = sorted(set(clf.classes) - set(clf.priors))
        if extra:
            print(f"  · 模型还认识 {extra}，没有部署 —— 判成它们的请求在已部署的 task 间重归一")
        return 0

    rows = [json.loads(x) for x in a.dump.read_text(encoding="utf-8").splitlines()
            if x.strip()]
    if not rows:
        print(f"{a.dump} 是空的")
        return 1
    l0 = int(rows[0]["l0"])
    E = int(rows[0]["n_experts"])
    # 部署了哪些池：命令行给了就用它，否则用落盘时记下的，再否则按标注里出现过的
    dep = rows[0].get("deployed")
    tasks = a.tasks or ",".join(f"{t}=1" for t in (dep or sorted(
        {r["true_task"] for r in rows if r.get("true_task")})))
    a.tasks = tasks
    raw = load_lr_params(a.model)
    cmap = parse_class_map(a.lr_classes or _cmap_str(raw.get("classes_map")),
                           raw["classes"])
    clf = LRClassifier.load(a.model, l0=l0, n_experts=E,
                            priors=priors_of(list(cmap.values())),
                            class_map=a.lr_classes, features=a.lr_features,
                            feature_fn=a.lr_feature_fn)
    print("  ", clf.describe())
    conf: Counter = Counter()
    agree = n_ok = n_lab = 0
    for r in rows:
        st = MoEStats(hist=np.zeros(E), layers={
            int(l): LayerStat.from_wire(s) for l, s in r["layers"].items()})
        v = clf.predict_stats(st)
        if r.get("lr_task") is not None:
            agree += v.task == r["lr_task"]
        if r.get("true_task"):
            n_lab += 1
            n_ok += v.task == r["true_task"]
            conf[(r["true_task"], v.task)] += 1
        print(f"  {r['req']:<8} 真实 {str(r.get('true_task')):<7} 重算 {v.task:<7}"
              f" c={v.confidence:.2f} {v.zone:<7} 在线 {r.get('lr_task')}"
              + (f"  ⚠ {v.error}" if v.error else ""))
    if n_lab:
        print(f"\n  准确率 {n_ok}/{n_lab} = {n_ok / n_lab:.0%}")
        ts = sorted({t for t, _ in conf} | {p for _, p in conf})
        print("  混淆（行=真实，列=判成）：")
        print(f"  {'':<12}" + "".join(f"{t[:10]:>11}" for t in ts))
        for t in ts:
            print(f"  {t[:10]:<12}" + "".join(f"{conf[(t, p)]:>11}" for p in ts))
    have = sum(r.get("lr_task") is not None for r in rows)
    if have:
        print(f"  与在线结果一致 {agree}/{have}"
              + ("" if agree == have else " —— 不一致说明节点上的模型/特征描述与这里不同"))
    if a.probe_dir:
        _compare_counts(rows, a.probe_dir, clf)
    return 0


# --------------------------------------------------------------------------- #
# task 选择器的训练口径（moe_prefill_probe.prompt_segments，逐字照抄）
# --------------------------------------------------------------------------- #
TRAIN_TEMPLATES = {
    "gsm8k": ("Solve the following math problem. Reason step by step, and put your "
              "final answer after '#### '.\n\nProblem: "),
    "math": ("Solve the following math problem. Reason step by step, and put your "
             "final answer in \\boxed{}.\n\nProblem: "),
    "humaneval": ("You are an expert Python programmer. Complete the following function. "
                  "Only output the complete function.\n\n"),
    "mbpp": ("You are an expert Python programmer. Write a Python function for the "
             "following task. Only output the code.\n\nTask: "),
}
MBPP_MID = "\nYour code should pass these tests:\n"


def train_segments(task: str, body: str, *, raw: bool = False) -> tuple[list, bool]:
    """bodies.jsonl 里的一条 → 训练时的 segments。返回 (segments, 是否精确重建)。

    * gsm8k / math / humaneval：模板 + 题目，精确；
    * mbpp：bodies 里题目与测试用例是直接拼在一起的（中间那段模板没存），按第一个
      `assert` 切开 —— 绝大多数精确，题目正文里恰好出现 assert 时会切错；
    * no_robots：没有任务模板，但个别样本带 system prompt（bodies 里没存），
      那几条整条 prompt 的 token 数会对不上，body 数不受影响。
    """
    if raw:
        if task == "mbpp":
            i = body.find("assert")
            body = body[:i] + "\n" + body[i:] if i > 0 else body
        return [(body, False)], True
    if task in ("gsm8k", "math", "humaneval"):
        return [(TRAIN_TEMPLATES[task], True), (body, False)], True
    if task == "mbpp":
        i = body.find("assert")
        if i <= 0:
            return [(TRAIN_TEMPLATES["mbpp"], True), (body, False)], False
        return [(TRAIN_TEMPLATES["mbpp"], True), (body[:i], False), (MBPP_MID, True),
                (body[i:], False)], True
    return [(body, False)], task == "no_robots"


def _probe_tasks(d: Path, tasks) -> list[str]:
    return tasks or sorted(p.name.removesuffix("_bodies.jsonl")
                           for p in d.glob("*_bodies.jsonl"))


def _bodies(d: Path, task: str) -> list[dict]:
    return [json.loads(x) for x in (d / f"{task}_bodies.jsonl").read_text(
        encoding="utf-8").splitlines() if x.strip()]


def _cmd_tokens(a) -> int:
    from ..runtime.text import TextIO

    tio = TextIO.from_model_dir(a.model_dir, chat=True)
    if a.thinking != "default":
        tio.template_kw["enable_thinking"] = a.thinking == "on"
    print(f"  tokenizer {a.model_dir}，enable_thinking={tio.template_kw.get('enable_thinking')}")
    bad_total = 0
    for u in _probe_tasks(a.probe_dir, a.tasks):
        with np.load(a.probe_dir / f"{u}_prefill_persample.npz") as z:
            ref = {int(i): (int(n), int(b)) for i, n, b in
                   zip(z["sample_idx"], z["n_prompt_tokens"], z["tokens_body"])}
        ok_full = ok_body = n = approx = 0
        first_bad = None
        for r in _bodies(a.probe_dir, u):
            segs, exact = train_segments(u, r["body"])
            bp = tio.encode_body(segs)
            want_n, want_b = ref[int(r["sample_idx"])]
            n += 1
            approx += not exact
            ok_full += len(bp.ids) == want_n
            ok_body += bp.n_body == want_b
            if (bp.n_body != want_b) and first_bad is None:
                first_bad = (r["sample_idx"], len(bp.ids), want_n, bp.n_body, want_b)
        bad_total += n - ok_body
        note = "（no_robots 带 system 的那几条整条长度对不上是预期的）" if u == "no_robots" else ""
        print(f"  {u:<10} {n:>5} 条  整条 prompt 一致 {ok_full}/{n}  body 一致 {ok_body}/{n}{note}")
        if first_bad:
            i, g, w, gb, wb = first_bad
            print(f"      首个不一致：样本 {i} 整条 {g} vs {w}，body {gb} vs {wb}")
    print("  ✓ body token 数与训练时完全一致" if bad_total == 0 else
          f"  ✗ {bad_total} 条 body 数不一致 —— 模板 / tokenizer / enable_thinking 与训练不同")
    return 0 if bad_total == 0 else 1


def _cmd_cases(a) -> int:
    rng = np.random.default_rng(0)
    out = []
    for u in _probe_tasks(a.probe_dir, a.tasks):
        rows = _bodies(a.probe_dir, u)
        pick = sorted(rng.choice(len(rows), size=min(a.n, len(rows)), replace=False))
        for j in pick:
            r = rows[int(j)]
            segs, _ = train_segments(u, r["body"], raw=a.raw)
            out.append({"id": f"{u}:{r['sample_idx']}", "task": u,
                        "segments": [[t, tpl] for t, tpl in segs]})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in out) + "\n",
                     encoding="utf-8")
    print(f"  {len(out)} 条 → {a.out}"
          + ("（去掉了任务指令模板）" if a.raw else "（与训练时同样带任务指令模板）"))
    print("  注意：这些是**训练样本**，识别准确率必然偏高。它的用处是逐样本核对计数 ——"
          "\n  跑完后 lr_tool check --probe-dir 会把前段收到的逐层计数与训练时的逐个比较")
    return 0


def _compare_counts(rows: list[dict], probe: Path, clf: LRClassifier) -> None:
    """前段 tail 收到的逐层计数 vs 训练时 probe 存的 counts_body，逐样本。

    bf16 下不同 batch 组成会有极少量路由翻转（SELECTOR_USAGE 第 6 节），所以除了
    「完全相同」也报 L1 差占比。"""
    cache: dict[str, dict[int, np.ndarray]] = {}
    exact = n = 0
    diffs = []
    for r in rows:
        cid = r.get("case_id")
        if not cid or ":" not in cid:
            continue
        u, idx = cid.split(":", 1)
        if u not in cache:
            f = probe / f"{u}_prefill_persample.npz"
            if not f.exists():
                continue
            with np.load(f) as z:
                cache[u] = {int(i): c for i, c in zip(z["sample_idx"], z["counts_body"])}
        ref = cache[u].get(int(idx))
        if ref is None:
            continue
        want = ref[[l - 1 for l in clf.layers]].astype(np.int64)
        got = np.stack([LayerStat.from_wire(r["layers"][str(l)]).count for l in clf.layers])
        n += 1
        exact += bool((got == want).all())
        diffs.append(np.abs(got - want).sum() / max(want.sum(), 1))
    if not n:
        print("\n  --probe-dir：dump 里没有带样本编号的请求 —— 测试集要用 lr_tool cases 生成")
        return
    print(f"\n  逐层计数 vs 训练时：{exact}/{n} 条逐个专家完全相同，"
          f"L1 差中位 {np.median(diffs):.2%}，最大 {max(diffs):.2%}")
    if max(diffs) > 0.05:
        print("  ⚠ 差得不少 —— 检查前段是否装全了前几层（--front-full）、模板与思考模式")


if __name__ == "__main__":
    raise SystemExit(main())
