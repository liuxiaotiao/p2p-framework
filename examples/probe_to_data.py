"""把 moe_prefill_probe.py 的逐样本路由统计，转成规划器与控制面要的两种输入。

    python3 examples/probe_to_data.py \
        --probe-dir Delta_30B/probe_30b_ref20/prefill_Qwen3-30B-A3B \
        --tasks gsm8k mbpp math humaneval no_robots --out task3 \
        --topo task/topo_fabric_mbps.json

产出（都在 --out 下）：

* `{task}_expert_activation.csv` —— `examples/task_deploy.py --data` 读的格式
  （layer, expert, prefill_count, decode_count, prefill_mean_weight,
  decode_mean_weight，层与专家 0-based）。**只有 prefill 相**：probe 只跑 prefill，
  decode 两列写 0。所以规划时要加 `--phase prefill`。
* `profile_prefill.json` —— `control --profile` 读的质量格式
  （`{"tasks": {u: {"n_tokens", "layers": {1-based 层: 归一化质量}}}}`），
  动态模式的后段驻留集从这里取。
* 给了 `--topo` 就把拓扑也拷过去，`--data` 一个目录就齐了。

口径
----
默认用 `full`（整条 prompt：对话模板 + 任务指令模板 + 题目）—— 服务时前段跑的
就是整条 prompt，后段要驻留的是这些 token 会路由到的专家。`--variant body` 只数
题目本身，与 task 选择器的特征同口径，但**不是**后段该装什么的依据。

prefill 与 decode 的分布不同（实测 prefill 更集中），后段在 decode 阶段会比按
prefill 选的驻留集多 miss 一些。有 decode 统计时应以 decode 为准（task/ 下
Qwen3-Next 的 CSV 就是这样）。
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_probe(d: Path, task: str, variant: str) -> tuple[np.ndarray, np.ndarray, int, dict]:
    """→ (counts[L,E], wsum[L,E], 该口径的 token 总数, meta)。对样本求和。"""
    f = d / f"{task}_prefill_persample.npz"
    if not f.exists():
        raise SystemExit(f"找不到 {f}")
    with np.load(f) as z:
        ck, wk, tk = f"counts_{variant}", f"wsum_{variant}", f"tokens_{variant}"
        if ck not in z.files:
            raise SystemExit(f"{f} 里没有 {ck}（采集时要加 --variants full body）")
        c = z[ck].astype(np.int64).sum(0)
        w = z[wk].astype(np.float64).sum(0)
        n = int(z[tk].sum())
        n_samples = int(z[ck].shape[0])
    mf = d / f"{task}_prefill_meta.json"
    meta = json.loads(mf.read_text(encoding="utf-8")) if mf.exists() else {}
    meta["n_samples"] = n_samples
    top_k = int(meta.get("top_k", 0) or 0)
    if top_k:
        want = n * top_k * c.shape[0]
        if int(c.sum()) != want:
            raise SystemExit(f"{task}: 计数和 {int(c.sum())} ≠ token 数 × top_k × 层数 {want}"
                             f" —— 采集不完整")
    return c, w, n, meta


def write_csv(path: Path, c: np.ndarray, w: np.ndarray) -> None:
    L, E = c.shape
    mean = np.divide(w, c, out=np.zeros_like(w), where=c > 0)
    with path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["layer", "expert", "prefill_count", "decode_count",
                     "prefill_mean_weight", "decode_mean_weight"])
        for l in range(L):
            for e in range(E):
                wr.writerow([l, e, int(c[l, e]), 0, f"{mean[l, e]:.6f}", "0.000000"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--probe-dir", required=True, type=Path,
                    help="moe_prefill_probe.py 的 --output-dir")
    ap.add_argument("--tasks", nargs="+", default=None,
                    help="默认：目录里所有 *_prefill_persample.npz")
    ap.add_argument("--variant", choices=("full", "body"), default="full")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--topo", type=Path, default=None, help="拓扑 JSON，拷进 --out")
    ap.add_argument("--model-name", default=None)
    a = ap.parse_args(argv)

    tasks = a.tasks or sorted(p.name.removesuffix("_prefill_persample.npz")
                              for p in a.probe_dir.glob("*_prefill_persample.npz"))
    if not tasks:
        raise SystemExit(f"{a.probe_dir} 里没有 *_prefill_persample.npz")
    a.out.mkdir(parents=True, exist_ok=True)
    prof = {"tasks": {}}
    shape = None
    for u in tasks:
        c, w, n, meta = load_probe(a.probe_dir, u, a.variant)
        if shape and c.shape != shape:
            raise SystemExit(f"{u} 的形状 {c.shape} 与前面的 {shape} 不同 —— 不是同一个模型")
        shape = c.shape
        write_csv(a.out / f"{u}_expert_activation.csv", c, w)
        z = w.sum(1, keepdims=True)
        mass = np.divide(w, z, out=np.zeros_like(w), where=z > 0)
        prof["tasks"][u] = {
            "n_tokens": n,
            "layers": {str(l + 1): [round(float(x), 8) for x in mass[l]]
                       for l in range(c.shape[0])},
        }
        print(f"  {u:<10} {meta.get('n_samples', '?'):>5} 条  {n:>7} 个 {a.variant} token"
              f"  {c.shape[0]} 层 × {c.shape[1]} 专家  → {u}_expert_activation.csv")
    L, E = shape
    model = a.model_name or Path(str(meta.get("model", "model"))).name
    prof.update(model=model, n_layers=L, n_experts=E, phase="prefill",
                variant=a.variant, source=str(a.probe_dir))
    (a.out / "profile_prefill.json").write_text(
        json.dumps(prof, ensure_ascii=False), encoding="utf-8")
    print(f"  画像 → {a.out / 'profile_prefill.json'}（{model}，{L} 层 × {E} 专家，prefill 相）")
    if a.topo:
        shutil.copy(a.topo, a.out / "topo_fabric_mbps.json")
        print(f"  拓扑 → {a.out / 'topo_fabric_mbps.json'}")
    print("\n  下一步（只有 prefill 统计，所以要 --phase prefill）：")
    print(f"    python3 examples/task_deploy.py --data {a.out} --config qwen3-30b-a3b "
          f"--phase prefill --tasks {','.join(f'{u}=1' for u in tasks[:3])} --min-l0 8 \\")
    print(f"        --save-plan {a.out}/plan.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
