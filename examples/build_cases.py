"""TestSamples/samples.txt → 控制面的测试集 JSONL（--prompts-file）。

    python3 examples/build_cases.py \
        --samples TestSamples/samples.txt --out TestSamples/cases.jsonl

samples.txt 每行 `数据集/split/下标`，下标是 **HF 原始 split 里的行号**：

    gsm8k/test/541       openai/gsm8k (main)                 question
    mbpp/test/305        google-research-datasets/mbpp (full) text + test_list
    norobots/test/88     HuggingFaceH4/no_robots              messages（单轮）

prompt 与选择器训练时（verify/moe_prefill_probe.py 的 prompt_segments）**逐字相同**：
任务指令模板 = 模板段（不进选择器特征），题目 = body；No Robots 没有任务模板，
system prompt（如果有）作为模板段放在最前。

数据从哪来（按顺序试）：
  1. `--data-dir` 下 save_to_disk 的目录（{dir}/gsm8k、{dir}/mbpp、{dir}/no_robots）
  2. `datasets.load_dataset`（装了 datasets 时；遵守 HF_ENDPOINT / HF 缓存）
  3. 直接下 parquet（遵守 HF_ENDPOINT；urllib 不通就用 curl），缓存到 --cache
`--bodies-dir` 给了探针落盘的 *_bodies.jsonl 目录时，逐条核对 gsm8k/mbpp 的题目。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

SPECS = {
    # 名字: (repo, config, parquet 路径模板)
    "gsm8k": ("openai/gsm8k", "main", "main/{split}-00000-of-00001.parquet"),
    "mbpp": ("google-research-datasets/mbpp", "full", "full/{split}-00000-of-00001.parquet"),
    "no_robots": ("HuggingFaceH4/no_robots", "default", "data/{split}-00000-of-00001.parquet"),
}
ALIASES = {"norobots": "no_robots", "no-robots": "no_robots", "gsm8k": "gsm8k", "mbpp": "mbpp",
           "no_robots": "no_robots"}

GSM8K_TPL = ("Solve the following math problem. Reason step by step, and put your "
             "final answer after '#### '.\n\nProblem: ")
MBPP_TPL = ("You are an expert Python programmer. Write a Python function for the "
            "following task. Only output the code.\n\nTask: ")
MBPP_MID = "\nYour code should pass these tests:\n"


def segments(task: str, ex: dict) -> list[tuple[str, bool]]:
    """与 moe_prefill_probe.prompt_segments 逐字相同。"""
    if task == "gsm8k":
        return [(GSM8K_TPL, True), (ex["question"], False)]
    if task == "mbpp":
        return [(MBPP_TPL, True), (ex["text"], False), (MBPP_MID, True),
                ("\n".join(ex["test_list"]), False)]
    if task == "no_robots":
        msgs = list(ex["messages"])
        roles = [m["role"] for m in msgs]
        core = roles[1:] if roles and roles[0] == "system" else roles
        if core != ["user", "assistant"]:
            raise ValueError(f"不是单轮对话（{roles}）—— 训练时这种样本被丢掉了")
        prompt_msgs = msgs[:-1]
        segs = [(m["content"] + "\n\n", True) for m in prompt_msgs[:-1]]
        segs.append((prompt_msgs[-1]["content"], False))
        return segs
    raise ValueError(task)


# --------------------------------------------------------------------------- #
# 取数据
# --------------------------------------------------------------------------- #
def _endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".partial")
    hdr = {"User-Agent": "p2pmoe-build-cases"}
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        hdr["Authorization"] = f"Bearer {tok}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=hdr),
                                    timeout=60) as r, tmp.open("wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
    except Exception as e:
        cmd = ["curl", "-fsSL", "--retry", "3", "-o", str(tmp), url]
        if tok:
            cmd[1:1] = ["-H", f"Authorization: Bearer {tok}"]
        if subprocess.run(cmd).returncode != 0:
            raise RuntimeError(f"下载失败 {url}（urllib: {e}）") from e
    os.replace(tmp, dst)


def _read_parquet(p: Path) -> list[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise SystemExit("读 parquet 要 pyarrow：pip install pyarrow"
                         "（requirements-control.txt 里有）") from e
    return pq.read_table(p).to_pylist()


def _datasets():
    """可选：装了 HF datasets 就用它（缓存、镜像都由它管），没装就走 parquet。"""
    import importlib
    try:
        return importlib.import_module("datasets")
    except ImportError:
        return None


def load_split(task: str, split: str, *, data_dir: Path | None, cache: Path) -> list:
    repo, config, path_tpl = SPECS[task]
    hf = _datasets()
    if data_dir and (data_dir / task).is_dir():
        if hf is None:
            raise SystemExit(f"{data_dir / task} 是 save_to_disk 的目录，读它要装 datasets")
        ds = hf.load_from_disk(str(data_dir / task))
        if hasattr(ds, "keys") and not hasattr(ds, "num_rows"):
            ds = ds[split]
        print(f"  {task}: 本地 {data_dir / task}（{len(ds)} 行）")
        return ds
    try:
        if hf is not None:
            ds = hf.load_dataset(repo, config if config != "default" else None, split=split)
            print(f"  {task}: datasets.load_dataset({repo}, {config}, {split})（{len(ds)} 行）")
            return ds
    except Exception as e:
        print(f"  {task}: datasets 加载失败（{type(e).__name__}: {str(e)[:120]}），改下 parquet")
    dst = cache / task / f"{split}.parquet"
    if not dst.exists():
        ep = _endpoint()
        urls = [f"{ep}/datasets/{repo}/resolve/main/{path_tpl.format(split=split)}"]
        try:   # 转换分支（任何数据集都有），作后备
            with urllib.request.urlopen(f"{ep}/api/datasets/{repo}/parquet/{config}/{split}",
                                        timeout=30) as r:
                urls += list(json.loads(r.read()))
        except Exception:
            pass
        err = None
        for u in urls:
            try:
                print(f"  {task}: 下载 {u}")
                _download(u, dst)
                err = None
                break
            except Exception as e:
                err = e
        if err:
            raise SystemExit(f"{task}: 所有地址都下载失败：{err}\n"
                             f"  没外网的话：在能上网的机器上把 parquet 放到 {dst}，或给 --data-dir")
    rows = _read_parquet(dst)
    print(f"  {task}: {dst}（{len(rows)} 行）")
    return rows


# --------------------------------------------------------------------------- #
def parse_samples(path: Path) -> list[tuple[str, str, int, str]]:
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split("/")
        if len(parts) != 3:
            raise SystemExit(f"看不懂这一行（要 数据集/split/下标）：{ln}")
        name, split, idx = parts
        task = ALIASES.get(name.lower())
        if task is None:
            raise SystemExit(f"不认识的数据集 {name}（认识 {sorted(set(ALIASES))}）")
        out.append((task, split, int(idx), ln))
    return out


def check_bodies(cases: list[dict], bodies_dir: Path) -> int:
    """gsm8k/mbpp：探针落盘的 body（= 全部 body 段拼接）应与这里拼出来的一致。"""
    bad = n = 0
    for task in ("gsm8k", "mbpp"):
        f = bodies_dir / f"{task}_bodies.jsonl"
        if not f.exists():
            continue
        ref = {}
        for ln in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(ln)
            ref[int(r.get("orig_idx", r["sample_idx"]))] = r["body"]
        for c in cases:
            if c["task"] != task or c["idx"] not in ref:
                continue
            n += 1
            mine = "".join(t for t, tpl in c["segments"] if not tpl)
            if mine != ref[c["idx"]]:
                bad += 1
                print(f"  ✗ {c['src']}: body 与 {f.name} 不一致")
    print(f"  核对 {n} 条 body：{n - bad} 一致" + (f"，{bad} 不一致" if bad else " ✓"))
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--samples", type=Path, default=Path("TestSamples/samples.txt"))
    ap.add_argument("--out", type=Path, default=Path("TestSamples/cases.jsonl"))
    ap.add_argument("--data-dir", type=Path, default=None, help="save_to_disk 的数据目录")
    ap.add_argument("--cache", type=Path, default=Path("TestSamples/.hf_cache"))
    ap.add_argument("--bodies-dir", type=Path, default=None,
                    help="探针的 *_bodies.jsonl 所在目录（全量 split 那份），用来核对")
    a = ap.parse_args(argv)

    want = parse_samples(a.samples)
    print(f"{a.samples}：{len(want)} 条 —— "
          + "，".join(f"{t} {sum(1 for w in want if w[0] == t)}"
                     for t in dict.fromkeys(w[0] for w in want)))
    splits = {}
    for task, split, _, _ in want:
        if (task, split) not in splits:
            splits[(task, split)] = load_split(task, split, data_dir=a.data_dir,
                                               cache=a.cache)
    cases, errs = [], []
    for task, split, idx, src in want:
        ds = splits[(task, split)]
        if not 0 <= idx < len(ds):
            errs.append(f"{src}: 下标越界（{task} {split} 只有 {len(ds)} 行）")
            continue
        try:
            segs = segments(task, dict(ds[idx]))
        except ValueError as e:
            errs.append(f"{src}: {e}")
            continue
        cases.append({"id": f"{task}:{idx}", "task": task, "src": src, "idx": idx,
                      "segments": [[t, tpl] for t, tpl in segs]})
    if errs:
        print("\n".join("  ✗ " + e for e in errs))
        return 1
    if a.bodies_dir and check_bodies(cases, a.bodies_dir):
        return 1
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps({k: c[k] for k in ("id", "task", "src", "segments")},
                               ensure_ascii=False) + "\n")
    print(f"→ {a.out}（{len(cases)} 条）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
