#!/usr/bin/env python3
"""换了节点编号 / 换了机器之后，把旧的逐层实测表（layer_bench_*.json）改挂到新编号上。

    python3 examples/remap_layer_bench.py qwen3-30b/layer_bench_wash1.json \\
        --hosts task/hosts.txt --topo task4/topo_fabric_mbps.json \\
        --out qwen3-30b/layer_bench_wash1_1105.json

逐层耗时与显存是**按机器**测的（rutg1、wash2 …），而表里的键是节点编号（N01 …）。编号重排后
直接用旧表，N09 会拿到另一台机器、甚至另一种卡的数据。这里按主机名对回去：

* 旧表里每个编号是哪台机器：`--ref` 表的 raw[编号].host（layerbench 原始结果里记着）；
  vLLM 表没有 raw，用 `--ref qwen3-30b/layer_bench_wash1.json` 借它的编号 → 主机名；
* 新编号是哪台机器：`--hosts` 每行注释里第一个括号中的主机名，如 `# A30 24GB (rutg1；…)`；
* 新拓扑里每个编号的卡型（`--topo` 的 role）用来核对：主机名对上但卡型不同的、以及旧表里
  没有的新机器，**借同型号卡里耗时居中的那台**的实测（标 source = copied:<主机名>）。
  借来的只是估计 —— 新机器尽快跑一次 `deploy_15.sh layerbench`。
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from pathlib import Path

ROLE_GPU = {"A30": "NVIDIA A30", "RTX6000": "Quadro RTX 6000", "T4": "Tesla T4"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("table", type=Path)
    ap.add_argument("--hosts", type=Path, required=True)
    ap.add_argument("--topo", type=Path, required=True)
    ap.add_argument("--ref", type=Path, default=None,
                    help="取旧编号 → 主机名的表（默认就是 table 自己；vLLM 表要给 layer_bench_wash1.json）")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    old = json.loads(a.table.read_text(encoding="utf-8"))
    ref = json.loads((a.ref or a.table).read_text(encoding="utf-8"))
    old_host = {n: r.get("host") for n, r in (ref.get("raw") or {}).items()}
    if not old_host:
        raise SystemExit(f"{a.ref or a.table} 里没有 raw[编号].host —— 用 --ref 给一份带 raw 的 layerbench 表")
    host_old = {h: n for n, h in old_host.items() if h}

    new_host: dict[str, str] = {}
    for line in a.hosts.read_text(encoding="utf-8").splitlines():
        body, _, note = line.partition("#")
        f = body.split()
        if len(f) >= 2:
            m = re.search(r"\(([A-Za-z0-9_.-]+)", note)
            new_host[f[0]] = m.group(1) if m else ""
    role = {n["id"]: n.get("role") for n in json.loads(a.topo.read_text(encoding="utf-8"))["nodes"]}

    pn, raw = old["per_node"], old.get("raw") or {}
    key = old.get("default_key") or next(k for k in old["keys"] if "@" in k)

    def donor(gpu: str) -> str:
        cand = sorted((float(v[key]), n) for n, v in pn.items() if v.get("gpu") == gpu and v.get(key))
        if not cand:
            raise SystemExit(f"旧表里没有 {gpu} 的实测，借不了")
        return cand[len(cand) // 2][1]

    out_pn, out_raw, rows = {}, {}, []
    for n in sorted(role):
        gpu = ROLE_GPU.get(role[n], role[n])
        h = new_host.get(n, "")
        src = host_old.get(h)
        how = f"{src}（{h}）"
        if src is None or pn.get(src, {}).get("gpu") != gpu:
            why = "旧表里没有这台" if src is None else f"旧 {src} 是 {pn[src].get('gpu')}"
            src = donor(gpu)
            how = f"借 {src}（{old_host.get(src)}）—— {why}"
            e = dict(pn[src], source=f"copied:{old_host.get(src)}")
        else:
            e = dict(pn[src])
        out_pn[n] = e
        if src in raw:
            r = dict(raw[src], node=n)
            if e.get("source", "").startswith("copied:"):
                r["host"] = h or r.get("host")
                r["copied_from"] = old_host.get(src)
            out_raw[n] = r
        rows.append((n, h, role[n], how, e.get(key)))

    gpus = {}
    for g in sorted({v["gpu"] for v in out_pn.values()}):
        ids = sorted(n for n, v in out_pn.items() if v["gpu"] == g)
        gv = dict(old["per_gpu"].get(g, {}))
        gv["nodes"] = ids
        gpus[g] = gv
    new = dict(old, per_node=out_pn, per_gpu=gpus)
    if raw:
        new["raw"] = out_raw
    new["remapped"] = {"from": str(a.table), "hosts": str(a.hosts), "topo": str(a.topo),
                       "map": {n: how for n, _, _, how, _ in rows}}
    a.out.write_text(json.dumps(new, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"{'新编号':<7}{'主机':<8}{'卡型':<9}{key:>12}  数据来源")
    for n, h, r, how, v in rows:
        print(f"{n:<7}{h:<8}{r:<9}{v if v is not None else '—':>12}  {how}")
    print(f"\n已存 {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
