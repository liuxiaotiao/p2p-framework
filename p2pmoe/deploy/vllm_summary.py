"""把 vLLM 的逐层实测（`vllm/summary_by_gpu.json`）转成规划器认的 layer_bench 表。

    python -m p2pmoe.deploy.vllm_summary vllm/summary_by_gpu.json \\
        --prompt 512 --gen 512 --batch 1 --out qwen3-30b/layer_bench_vllm.json
    python3 examples/task_deploy.py ... --layer-times qwen3-30b/layer_bench_vllm.json \\
        --layer-time-key req@p512g512 --measured-mem --mem-margin-gb 0 --kv-reserve-gb 2

输入（按 GPU 型号汇总）
----------------------
* ``decode[专家数][上下文][batch]``：decode 每 token 每层的耗时（ms）；
* ``prefill[专家数][prompt 长度]``：一次 prefill 整层的耗时（ms）；
* ``decode_overhead_ms[专家数][上下文][batch]``：每步 decode 里不随层数增长的那部分（ms/token）；
* ``ms_per_layer_nodes``：逐节点的 decode 均值（batch 1、全部专家、各上下文平均）——
  用它与型号均值之比，把型号曲线换算到每台；
* ``memory[专家数]``：整卡容量、每层权重与分配后实占、非 KV 的固定开销（含词嵌入 + 输出头）、
  KV 每 token 每层字节数（MiB / KiB，1 MiB = 2^20 字节）。

输出的每层耗时（每台一个值，ms/层）
----------------------------------
给定 prompt P、生成 G、batch B，取 ``专家数``（默认 128 —— 规划偏保守；batch 1 时 64 与 128
的 decode 几乎一样）：

* ``decode@pPgG``：上下文从 P 涨到 P+G−1 的这 G 步 decode 的平均（按量过的上下文点线性插值）；
* ``prefill@P``：prefill P 个 token 的整层耗时；
* ``req@pPgG`` = (prefill@P + G × decode@pPgG) / G：一条请求摊到每个生成 token 的每层耗时 ——
  把 prefill 也算进去，规划默认用它。

显存（GB = 1e9 字节）
--------------------
* ``total_gb``：整卡容量；
* ``context_gb``：vLLM 的非权重固定开销 = 非 KV 固定部分 − 词嵌入与输出头（CUDA context、
  CUDA graph、profile 时的激活峰值都在里面），所以 ``prefill_extra_gb`` 记 0；
* ``overhead_ratio``：每层实占 ÷ 每层权重（分配取整/碎片）；
* 词嵌入、输出头各取非层权重的一半，由规划器计到前段第一台 / 后段最后一台。

vLLM 按 ``gpu_memory_utilization`` 圈显存：按本表规划的节点要把它设到
「权重 + 固定开销 + 留给 KV 的量」÷ 整卡容量以上，否则 KV 拿不到规划时留的那份。
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path
from typing import Mapping

__all__ = ["to_layer_table", "main"]

MIB = 2 ** 20 / 1e9          # 1 MiB → GB(1e9)


def _interp(points: Mapping[int, float], x: float) -> float:
    """分段线性插值；超出量过的范围就取端点值。"""
    xs = sorted(points)
    if x <= xs[0]:
        return points[xs[0]]
    if x >= xs[-1]:
        return points[xs[-1]]
    for a, b in zip(xs, xs[1:]):
        if a <= x <= b:
            return points[a] + (points[b] - points[a]) * (x - a) / (b - a)
    raise AssertionError("unreachable")


def _curve(tbl: Mapping, batch: int | None = None) -> dict[int, float]:
    out = {}
    for ctx, v in tbl.items():
        x = v if batch is None else v.get(str(batch))
        if x is not None:
            out[int(ctx)] = float(x)
    return out


def _gen_mean(curve: Mapping[int, float], prompt: int, gen: int) -> float:
    """上下文 prompt, prompt+1, …, prompt+gen−1 这 gen 步的平均。"""
    return st.fmean(_interp(curve, prompt + i) for i in range(gen))


def to_layer_table(summary: Mapping, *, prompt: int = 512, gen: int = 512, batch: int = 1,
                   experts: int = 128) -> dict:
    if prompt <= 0 or gen <= 0 or batch <= 0:
        raise ValueError("prompt / gen / batch 都要是正数")
    ek = str(experts)
    kd, kp, kr = f"decode@p{prompt}g{gen}", f"prefill@{prompt}", f"req@p{prompt}g{gen}"
    keys = [kd, kp, kr]
    per_node: dict[str, dict] = {}
    per_gpu: dict[str, dict] = {}
    for g, s in summary.items():
        if ek not in s.get("decode", {}):
            raise ValueError(f"{g} 没有 {experts} 个专家的 decode 实测；有的是 {sorted(s.get('decode', {}))}")
        dec = _curve(s["decode"][ek], batch)
        if not dec:
            raise ValueError(f"{g} 没有 batch={batch} 的 decode 实测")
        pre = _curve(s["prefill"][ek])
        ovh = _curve(s.get("decode_overhead_ms", {}).get(ek, {}), batch)
        d = _gen_mean(dec, prompt, gen)
        p = _interp(pre, prompt)
        base = {kd: d, kp: p, kr: (p + gen * d) / gen}
        fixed_ms = _gen_mean(ovh, prompt, gen) if ovh else None

        m = s["memory"][ek]
        nonlayer = float(m["non_layer_weight_mib"])
        mem = {
            "total_gb": round(float(m["device_total_mib"]) * MIB, 4),
            "context_gb": round((float(m["device_non_kv_mib_fixed"]) - nonlayer) * MIB, 4),
            "prefill_extra_gb": 0.0,
            "prefill_extra_at": None,
            "overhead_ratio": round(float(m["device_non_kv_mib_per_layer"])
                                    / float(m["weight_mib_per_layer"]), 4),
            "kv_bytes_per_token_layer": float(m["kv_kib_per_token_per_layer"]) * 1024,
            "weights_per_layer_gb": round(float(m["weight_mib_per_layer"]) * MIB, 4),
            "embed_gb": round(nonlayer / 2 * MIB, 4),
            "lm_head_gb": round(nonlayer / 2 * MIB, 4),
        }
        ref = float(s["ms_per_layer"])
        nodes = s.get("ms_per_layer_nodes") or {n: ref for n in s.get("nodes", [])}
        for nid, v in nodes.items():
            f = float(v) / ref if ref else 1.0
            per_node[nid] = {"gpu": s.get("gpu", g), "source": "vllm", "node_factor": round(f, 4),
                             **{k: round(x * f, 4) for k, x in base.items()},
                             "fixed_ms_per_token": None if fixed_ms is None else round(fixed_ms, 4),
                             "memory": dict(mem)}
        ids = sorted(nodes)
        per_gpu[s.get("gpu", g)] = {
            "nodes": ids,
            **{k: {"mean": round(st.fmean(per_node[i][k] for i in ids), 4),
                   "std": round(st.stdev([per_node[i][k] for i in ids]), 4) if len(ids) > 1 else 0.0,
                   "n": len(ids)} for k in keys},
            "fixed_ms_per_token": None if fixed_ms is None else round(fixed_ms, 4),
            "memory": {k: {"mean": v, "std": 0.0, "n": len(ids)}
                       for k, v in mem.items() if isinstance(v, (int, float))},
        }
    return {"source": "vllm", "default_key": kr, "keys": keys,
            "setup": {"prompt": prompt, "gen": gen, "batch": batch, "experts": experts},
            "per_node": dict(sorted(per_node.items())), "per_gpu": per_gpu}


def _print(t: dict) -> None:
    ks = t["keys"]
    print(f"vLLM 实测 → 每层耗时（ms），{t['setup']}")
    print(f"  {'节点':<6}{'GPU':<18}" + "".join(f"{k:>16}" for k in ks) + f"{'固定开销/token':>16}")
    for n, v in t["per_node"].items():
        fx = v.get("fixed_ms_per_token")
        print(f"  {n:<6}{str(v['gpu'])[:17]:<18}" + "".join(f"{v[k]:>16.4f}" for k in ks)
              + (f"{fx:>16.3f}" if fx is not None else f"{'—':>16}"))
    print("显存（GB）")
    for g, v in t["per_gpu"].items():
        m = {k: x["mean"] for k, x in v["memory"].items()}
        print(f"  {g[:17]:<18} 整卡 {m['total_gb']:.3f}  非权重固定 {m['context_gb']:.3f}  "
              f"开销比 {m['overhead_ratio']:.4f}  每层权重 {m['weights_per_layer_gb']:.3f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("summary", type=Path)
    ap.add_argument("--prompt", type=int, default=512)
    ap.add_argument("--gen", type=int, default=512)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--experts", type=int, default=128)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    t = to_layer_table(json.loads(a.summary.read_text(encoding="utf-8")),
                       prompt=a.prompt, gen=a.gen, batch=a.batch, experts=a.experts)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(t, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    _print(t)
    print(f"→ {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
