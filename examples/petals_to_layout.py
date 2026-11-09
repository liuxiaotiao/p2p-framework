#!/usr/bin/env python3
"""Petals 的分配（chains → servers → "a:b" 层切片）→ manual_plan.py 认的布局 JSON。

    python3 examples/petals_to_layout.py qwen3-30b/petals/plan_cap8_petals.json \\
        --out qwen3-30b/layout_petals_cap8.json
    python3 examples/manual_plan.py qwen3-30b/layout_petals_cap8.json --out qwen3-30b/plan_petals_cap8.json \\
        --layer-times qwen3-30b/layer_bench_wash1_1105.json --kv-reserve-gb 2 --topo task4/topo_fabric_mbps.json

* Petals 的 ``layers`` 是 0-based 半开切片：``"0:8"`` = 第 1–8 层；这里转成 1-based 闭区间 ``[1, 8]``；
* 每条链切成前段 + 后段：前段 = 从第 1 层起、到第 L₀ 层为止的那几台（L₀ 默认取第一台的末层，
  也可用 ``--l0`` 指定，但 L₀ 必须正好落在两台的交界上 —— 一台机器只属于一条段）；
* 专家全装（Petals 跑的是完整模型），后段标 ``task: any`` 进共享池：识别成哪个 task 都派到这里；
* Petals 的 ``client`` 不进清单 —— 这里的控制器照旧在 wash1 上。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def convert(plan: dict, l0: int | None = None) -> dict:
    chans, seen = [], {}
    for name, ch in plan["chains"].items():
        segs = []
        for s in ch["servers"]:
            a, b = (int(x) for x in str(s["layers"]).split(":"))
            segs.append({"node": s["node"], "layers": [a + 1, b]})
        segs.sort(key=lambda x: x["layers"][0])
        cut = segs[0]["layers"][1] if l0 is None else l0
        ends = [x["layers"][1] for x in segs]
        if cut not in ends:
            raise SystemExit(f"链 {name}：L₀={cut} 落在某台机器的层区间中间（各台末层 {ends}）")
        for x in segs:
            if x["node"] in seen:
                raise SystemExit(f"{x['node']} 同时出现在链 {seen[x['node']]} 和 {name}")
            seen[x["node"]] = name
        chans.append({"task": "any", "chain": name,
                      "front": [x for x in segs if x["layers"][1] <= cut],
                      "back": [x for x in segs if x["layers"][0] > cut]})
    l0s = {c["front"][-1]["layers"][1] for c in chans}
    if len(l0s) != 1:
        raise SystemExit(f"各条链的 L₀ 不一致：{sorted(l0s)} —— 用 --l0 统一")
    return {"_note": "由 examples/petals_to_layout.py 从 Petals 分配转来：完整模型（专家全装），"
                     "原表层号是 0-based 切片（0:8 = 第 1–8 层）；task=any → 共享后段池。",
            "l0": l0s.pop(), "channels": chans}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("petals", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--l0", type=int, default=None)
    a = ap.parse_args(argv)
    lay = convert(json.loads(a.petals.read_text(encoding="utf-8")), a.l0)
    a.out.write_text(json.dumps(lay, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for c in lay["channels"]:
        print(f"  链 {c['chain']}：前段 " + " → ".join(f"{x['node']}{x['layers']}" for x in c["front"])
              + "；后段 " + " → ".join(f"{x['node']}{x['layers']}" for x in c["back"]))
    print(f"L₀ = {lay['l0']}，布局已存 {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
