#!/usr/bin/env python3
"""手写的分配 → 部署清单（不经过规划器）。

    python3 examples/manual_plan.py qwen3-30b/layout_full_2ch.json \\
        --config qwen3-30b-a3b --out qwen3-30b/plan_full_2ch.json \\
        --layer-times qwen3-30b/layer_bench_wash1.json --kv-reserve-gb 2 \\
        --topo task3/topo_fabric_mbps.json

布局文件就是 `p2pmoe/deploy/manual.py` 认的格式（层号 1-based、闭区间）::

    {"l0": 8,
     "channels": [
       {"task": "any",
        "front": [{"node": "N01", "layers": [1, 8]}],
        "back":  [{"node": "N05", "layers": [9, 16]}, …, {"node": "N06", "layers": [41, 48]}]},
       …]}

* 专家默认**全装**（完整模型）；要裁剪就在布局里写 `experts`（见 manual.py）；
* `task` 写某个 task 名就是那个 task 的专属后段；写 `any` 就进**共享池** ——
  识别成任何一个没有专属后段的 task，都派到这里（coordinator.SHARED_POOL）；
* 产出的清单与规划器的格式一样：`PLAN=<它> bash ./deploy_15.sh fetch/verify/continuous`。

给了 `--layer-times` 就按实测显存核对每台装不装得下（每台可用 = 整卡 − context − 激活峰值
− 余量 − KV 预留，与 task_deploy --measured-mem 同一套）；再给 `--topo` 就按实测每层耗时与
拓扑的单向 p50 估每个「前段 × 后段」组合的每 token 时延 —— 只是估算，手动布局不做任何均匀性保证。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    from p2pmoe.deploy.manual import ManualSpec, build_manual_manifest
    from p2pmoe.planner.hf_config import PRESETS, model_spec_from_hf

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("layout", type=Path)
    ap.add_argument("--config", default="qwen3-30b-a3b",
                    help="预置名或 config.json 路径")
    ap.add_argument("--ctx", type=int, default=2560, help="KV 按多长的上下文记进清单（只影响 kv_gb 字段）")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layer-times", type=Path, default=None,
                    help="layerbench 结果（逐节点每层耗时与显存实测）")
    ap.add_argument("--layer-time-key", default="decode@1024")
    ap.add_argument("--mem-prefill-tokens", type=int, default=512)
    ap.add_argument("--mem-margin-gb", type=float, default=0.0)
    ap.add_argument("--kv-reserve-gb", type=float, default=2.0)
    ap.add_argument("--topo", type=Path, default=None, help="拓扑 JSON（估时延用）")
    a = ap.parse_args(argv)

    hf = dict(PRESETS[a.config]) if a.config in PRESETS else json.loads(
        (Path(a.config) / "config.json" if Path(a.config).is_dir() else Path(a.config))
        .read_text(encoding="utf-8"))
    model, _ = model_spec_from_hf(hf, name=a.config, ctx_max=a.ctx)
    spec = ManualSpec.from_file(a.layout, n_layers=model.n_layers)
    try:
        man, _ = build_manual_manifest(spec, model, model_name=a.config)
    except ValueError as e:
        print(e)
        return 2
    a.out.write_text(man.to_json(), encoding="utf-8")

    emb = int(hf["vocab_size"]) * int(hf["hidden_size"]) * 2 / 1e9
    segs: dict[str, list] = {}
    for p in man.nodes:
        segs.setdefault(p.segment, []).append(p)
    for v in segs.values():
        v.sort(key=lambda p: p.position)

    print(f"L₀ = {man.l0}，{len(spec.channels)} 条通道，{len(man.nodes)} 台")
    for sid, ps in segs.items():
        role = ps[0].role
        print(f"  {sid:<8}{role:<16}" + " → ".join(
            f"{p.node}[{p.layer_range[0]}–{p.layer_range[1]}]" for p in ps))

    ok = True
    ms = {}
    if a.layer_times:
        from p2pmoe.deploy.layer_bench import per_node_memory, per_node_ms

        table = json.loads(a.layer_times.read_text(encoding="utf-8"))
        mem = per_node_memory(table, a.mem_prefill_tokens)
        ms = per_node_ms(table, a.layer_time_key)
        print(f"\n显存（GB；可用 = 整卡 − context − 激活峰值 − 余量 {a.mem_margin_gb:g} "
              f"− KV 预留 {a.kv_reserve_gb:g}）")
        print(f"  {'节点':<6}{'层':<9}{'专家/层':>8}{'权重':>8}{'首尾表':>8}{'需要':>8}{'可用':>8}")
        for p in man.nodes:
            m = mem.get(p.node) or {}
            if not m.get("total_gb"):
                print(f"  {p.node:<6}（没有实测显存，跳过）")
                continue
            ratio = max(1.0, float(m.get("overhead_ratio") or 1.0))
            extra = (emb if p.role.startswith("front") and p.is_head else 0.0) + \
                    (emb if p.role.startswith("back") and p.is_tail else 0.0)
            need = p.weight_gb * ratio + extra
            have = (float(m["total_gb"]) - float(m.get("context_gb") or 0)
                    - float(m.get("prefill_extra_gb") or 0) - a.mem_margin_gb - a.kv_reserve_gb)
            bad = need > have
            ok &= not bad
            ne = round(sum(len(l.experts) for l in p.layers) / max(len(p.layers), 1))
            print(f"  {p.node:<6}{p.layer_range[0]}–{p.layer_range[1]:<6}{ne:>8}"
                  f"{p.weight_gb:>8.2f}{extra:>8.2f}{need:>8.2f}{have:>8.2f}"
                  + ("  ✗ 装不下" if bad else ""))

    if a.topo and ms:
        from p2pmoe.planner.from_data import load_topology

        topo = load_topology(a.topo, d_model=int(hf["hidden_size"]))
        hop = lambda x, y: 0.0 if x == y else topo.p50_ms.get((x, y), float("nan"))

        def delay(ps):
            c = sum((p.layer_range[1] - p.layer_range[0] + 1) * ms.get(p.node, float("nan"))
                    for p in ps)
            h = sum(hop(x.node, y.node) for x, y in zip(ps, ps[1:]))
            return c, h

        fronts = [s for s, ps in segs.items() if ps[0].role.startswith("front")]
        backs = [s for s, ps in segs.items() if ps[0].role.startswith("back")]
        print(f"\n每 token 时延估算（ms；计算 = 层数 × {a.layer_time_key}，跳 = 拓扑单向 p50）")
        for s in fronts + backs:
            c, h = delay(segs[s])
            print(f"  {s:<8}计算 {c:6.1f}  段内跳 {h:5.2f}")
        print(f"  {'前段':<8}{'后段':<10}{'正向':>7}{'回环':>7}{'合计':>8}")
        for f in fronts:
            for b in backs:
                cf, hf_ = delay(segs[f])
                cb, hb = delay(segs[b])
                w = hop(segs[f][-1].node, segs[b][0].node)
                d = hop(segs[b][-1].node, segs[f][0].node)
                print(f"  {f:<8}{b:<10}{w:>7.2f}{d:>7.2f}{cf + hf_ + w + cb + hb + d:>8.1f}")

    print(f"\n清单已存 {a.out}" + ("" if ok else "（⚠ 有节点装不下，见上表）"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
