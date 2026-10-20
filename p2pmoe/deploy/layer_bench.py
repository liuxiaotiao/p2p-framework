"""逐层计时：这台 GPU 跑一层 Qwen3-MoE 要多久。

    # 每台节点上（agent 要先停掉：它占着显存，也会和计时抢 GPU）
    python -m p2pmoe.deploy.layer_bench run --model-dir $WEIGHTS --node N03 \\
        --out /tmp/p2pmoe/layer_bench-N03.json
    # 控制机上把各台的结果合起来
    python -m p2pmoe.deploy.layer_bench merge results/layer_bench/*.json \\
        --out results/layer_bench.json
    # 规划时用实测值代替拓扑里的抽象算力
    python3 examples/task_deploy.py ... --layer-times results/layer_bench.json

或者一条命令走完全部 13/15 台：`bash ./deploy_15.sh layerbench`。

为什么要量
----------
规划器的计算项是 `层数 × ms_per_layer`。拓扑文件里只有一个抽象的 compute_capacity，
`load_topology` 按反比缩放成「最快那档 0.35ms/层」—— 相对快慢大致对，绝对值却差得
很远：实测 TPOT 130–145ms / 48 层，摊到每层是 2–3ms，网络一跳才 0.03–13.5ms。
计算项被低估 7 倍左右，规划器就会把「多一跳」看得比「换一张慢卡」贵得多。

量什么
------
用**与节点服务时完全相同的代码路径**（`TorchSegmentModel`，同一个 dtype、同一个
miss 策略、同一条 decode 快路径），只是权重换成同形状的随机数：

* **decode**：先 prefill 一段长为 ctx 的上下文，再逐 token 走 decode，每步计时。
  这是生成阶段每个 token 在本节点每层付的计算时间 —— 规划器默认用它；
* **prefill**：一次喂 T 个 token，看一层要多久（TTFT 里本节点的那一份）。

每 token 激活的是 top-k 个专家，与驻留多少无关（I.1.1），所以计算量不依赖驻留集；
`--resident` 可以量不同驻留数，确认这一点。随机权重下路由近似均匀，与真实路由的偏斜
不同，但每 token 算的专家个数一样 —— 计算量一样。

结果是本机实测，含当时的频率、温度与驱动状态。同型号的卡之间也可能差 10% 以上，
所以按节点记，不按型号记；缺某台时规划器才退到「同型号的均值」。
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics as st
import sys
from pathlib import Path
from typing import Mapping, Sequence

log = logging.getLogger("p2pmoe.layer_bench")

__all__ = ["merge", "per_node_ms", "per_node_memory", "run_remote", "main"]

DEFAULT_KEY = "decode@1024"
# 与执行层（p2pmoe/runtime/layer_bench.py）一致；那边才 import torch
DEFAULT_CTX = (*range(128, 1025, 128), 2048)      # 128, 256, …, 1024，外加 2048
DEFAULT_PREFILL = (*range(128, 1025, 128), 2048)


# --------------------------------------------------------------------------- #
def _value(r: dict, key: str, variant: str = "all") -> float | None:
    """key 形如 decode@1024 / prefill@512（prefill 取每层总时长）。"""
    kind, _, at = key.partition("@")
    v = (r.get("variants") or {}).get(variant) or next(iter((r.get("variants") or {}).values()), None)
    if not v or kind not in ("decode", "prefill"):
        return None
    s = v.get(kind, {}).get(at)
    return None if s is None or "median" not in s else float(s["median"])


MEM_FIELDS = ("total_gb", "context_gb", "overhead_ratio", "kv_bytes_per_token_layer",
              "prefill_extra_gb", "decode_extra_gb", "weights_per_layer_gb",
              "embed_gb", "lm_head_gb")


def _memory(r: dict) -> dict:
    """单台结果 → 显存的几个关键量（GB，1e9）。取「全部专家」那一组；没有就取第一组。

    prefill_extra_gb / decode_extra_gb 取量过的各长度里最大的那个 —— 规划要的是上限。
    """
    v = (r.get("variants") or {}).get("all") or next(iter((r.get("variants") or {}).values()), {})
    m = dict(v.get("memory") or {})
    dm = r.get("device_memory") or {}
    out = {"total_gb": dm.get("total_gb"), "context_gb": dm.get("context_gb"),
           "overhead_ratio": m.get("overhead_ratio"),
           "kv_bytes_per_token_layer": m.get("kv_bytes_per_token_layer"),
           "prefill_extra_gb": max(m.get("prefill_extra_gb", {}).values(), default=None),
           "prefill_extra_at": max((int(k) for k in m.get("prefill_extra_gb", {})), default=None),
           "decode_extra_gb": max(m.get("decode_extra_gb", {}).values(), default=None),
           "weights_per_layer_gb": m.get("weights_per_layer_gb"),
           "embed_gb": m.get("embed_gb"), "lm_head_gb": m.get("lm_head_gb")}
    return out


def merge(results: Sequence[dict], *, key: str = DEFAULT_KEY) -> dict:
    """各节点的结果 → 一张表：逐节点、逐 GPU 型号（mean ± std）。"""
    nodes: dict = {}
    for r in results:
        nid = r.get("node") or r.get("host")
        nodes[nid] = r
    by_gpu: dict[str, list[str]] = {}
    for nid, r in nodes.items():
        by_gpu.setdefault(r["gpu"].get("name", "?"), []).append(nid)
    keys = sorted({f"{k}@{a}" for r in nodes.values() for v in r["variants"].values()
                   for k in ("decode", "prefill") for a in v.get(k, {})},
                  key=lambda s: (s.split("@")[0], int(s.split("@")[1])))
    gpus = {}
    for g, ids in sorted(by_gpu.items()):
        gpus[g] = {"nodes": sorted(ids)}
        for k in keys:
            xs = [x for x in (_value(nodes[i], k) for i in ids) if x is not None]
            if xs:
                gpus[g][k] = {"mean": round(st.mean(xs), 4),
                              "std": round(st.stdev(xs), 4) if len(xs) > 1 else 0.0,
                              "n": len(xs)}
    per_node = {nid: {"gpu": r["gpu"].get("name"),
                      "source": (r.get("setup") or {}).get("source", "synthetic"),
                      "segment": r.get("segment"),
                      **{k: _value(r, k) for k in keys},
                      "memory": _memory(r)} for nid, r in sorted(nodes.items())}
    for g, ids in by_gpu.items():
        mem = {}
        for f in MEM_FIELDS:
            xs = [per_node[i]["memory"].get(f) for i in ids]
            xs = [float(x) for x in xs if x is not None]
            if xs:
                mem[f] = {"mean": round(st.mean(xs), 4),
                          "std": round(st.stdev(xs), 4) if len(xs) > 1 else 0.0, "n": len(xs)}
        gpus[g]["memory"] = mem
    return {"default_key": key, "keys": keys, "per_node": per_node, "per_gpu": gpus,
            "raw": nodes}


def per_node_memory(table: Mapping) -> dict[str, dict]:
    """merge 的产物 → {节点: 显存量表}（见 _memory）。"""
    return {n: dict(v.get("memory") or {}) for n, v in table.get("per_node", {}).items()}


def per_node_ms(table: Mapping, key: str = DEFAULT_KEY) -> dict[str, float]:
    """merge 的产物（或单台的 run 结果列表）→ {节点: 每层毫秒}。缺值的节点不出现。"""
    pn = table.get("per_node", {})
    return {n: float(v[key]) for n, v in pn.items() if v.get(key) is not None}


def _print_table(m: dict) -> None:
    keys = m["keys"]
    f3 = lambda x: f"{x:>8.3f}" if x is not None else f"{'—':>8}"
    for kind, title in (("decode", "decode：每 token 每层耗时（ms，中位数），列为上下文长度"),
                        ("prefill", "prefill：一次喂 T 个 token 时整层耗时（ms，中位数），列为 T")):
        ks = [k for k in keys if k.startswith(kind + "@")]
        if not ks:
            continue
        print(f"\n{title}")
        print(f"  {'节点':<6}{'GPU':<22}{'权重':<7}" + "".join(f"{k.split('@')[1]:>8}" for k in ks))
        for n, v in m["per_node"].items():
            src = {"loaded": "已加载", "synthetic": "随机"}.get(v.get("source"), "?")
            print(f"  {n:<6}{str(v['gpu'])[:21]:<22}{src:<7}" + "".join(f3(v.get(k)) for k in ks))
        for g, gv in m["per_gpu"].items():
            print(f"  {'均值':<6}{g[:21]:<22}{'':<7}"
                  + "".join(f3(gv[k]["mean"]) if k in gv else f"{'—':>8}" for k in ks))
            print(f"  {'std':<6}{'':<22}{'':<7}"
                  + "".join(f3(gv[k]["std"]) if k in gv else f"{'—':>8}" for k in ks))
    print("\n显存（GB = 1e9 字节）：整卡容量 / CUDA context / 分配器开销比 / 每 token 每层 KV / "
          "prefill 激活峰值 / 每层权重")
    print(f"  {'节点':<6}{'容量':>8}{'context':>9}{'开销比':>8}{'KV B/tok·层':>13}"
          f"{'激活峰值':>10}{'每层权重':>10}")
    f = lambda x, d=3: "—" if x is None else f"{x:.{d}f}"
    for n, v in m["per_node"].items():
        mm = v.get("memory") or {}
        act = mm.get("prefill_extra_gb")
        print(f"  {n:<6}{f(mm.get('total_gb'), 2):>8}{f(mm.get('context_gb')):>9}"
              f"{f(mm.get('overhead_ratio')):>8}{f(mm.get('kv_bytes_per_token_layer'), 0):>13}"
              f"{f(act):>10}{f(mm.get('weights_per_layer_gb')):>10}"
              + (f"  （T={mm.get('prefill_extra_at')}）" if act is not None else ""))
    print("\nGPU 型号 → 节点：" + "；".join(f"{g} {v['nodes']}" for g, v in m["per_gpu"].items()))


# --------------------------------------------------------------------------- #
def run_remote(addrs: Mapping[str, tuple[str, int]], *, hf_config: Mapping | None,
               mode: str = "auto", opts: Mapping | None = None, timeout: float = 900.0,
               relay=None, parallel: int = 16, on_done=None) -> tuple[dict, dict]:
    """从控制机让每个 agent 在**自己的 GPU**上计时，收回结果。

    不经 ssh、不停 agent：走控制面的 TCP（与 capabilities / configure 同一条路）。
    agent 装着权重时（auto / loaded）量的就是它在清单里分到的那几层、那些专家；
    没装时（auto / synthetic）按 hf_config 建同形状的随机权重量。
    返回 (结果 {节点: run 结果}, 失败 {节点: 原因})。
    """
    from concurrent.futures import ThreadPoolExecutor

    from ..runtime.wire import rpc

    msg = {"type": "layer_bench", "mode": mode, **(opts or {})}
    if hf_config is not None:
        msg["hf_config"] = dict(hf_config)

    def one(node):
        try:
            r = rpc(addrs[node], msg, timeout=timeout, relay=relay, to=node)
        except (OSError, TimeoutError) as e:
            return node, None, f"{type(e).__name__}: {e}"
        if r.get("type") == "error":
            return node, None, (r.get("trace") or "节点报错").strip().splitlines()[-1]
        if not r.get("ok"):
            return node, None, r.get("why", "未知原因")
        return node, r["result"], None

    got: dict = {}
    bad: dict = {}
    with ThreadPoolExecutor(max_workers=max(1, min(parallel, len(addrs)))) as ex:
        for node, res, why in ex.map(one, sorted(addrs)):
            if res is not None:
                got[node] = res
            else:
                bad[node] = why
            if on_done is not None:
                on_done(node, res, why)
    return got, bad


# --------------------------------------------------------------------------- #
def _hf_config(model_dir: str | None, preset: str | None) -> dict:
    if model_dir:
        p = Path(model_dir) / "config.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        if not preset:
            raise SystemExit(f"{p} 不存在 —— --model-dir 指到节点的权重目录（里面有 config.json），"
                             f"或者用 --config qwen3-30b-a3b")
        log.info("%s 不存在，用预置 %s", p, preset)
    from ..planner.hf_config import PRESETS

    if preset not in PRESETS:
        raise SystemExit(f"--config 只认 {sorted(PRESETS)}")
    return {"torch_dtype": "bfloat16", **PRESETS[preset]}


def _ints(s: str) -> list[int]:
    """逗号分隔；每项可以是单个数，也可以是区间 起:止:步长（含两端），步长省略时等于起点。

        "128,1024,2048"        → 128, 1024, 2048
        "128:1024:128"         → 128, 256, …, 1024
        "128:1024:128,2048"    → 同上再加 2048
    """
    out: set[int] = set()
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            parts = [int(x) for x in item.split(":")]
            if len(parts) not in (2, 3):
                raise ValueError(f"区间写成 起:止:步长，收到 {item!r}")
            lo, hi = parts[0], parts[1]
            step = parts[2] if len(parts) == 3 else lo
            if step <= 0 or lo <= 0 or hi < lo:
                raise ValueError(f"区间 {item!r} 不合法（要 0 < 起 ≤ 止、步长 > 0）")
            out.update(range(lo, hi + 1, step))
        else:
            out.add(int(item))
    return sorted(out)


def _fmt(xs: Sequence[int]) -> str:
    return ",".join(map(str, xs))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="p2pmoe-layer-bench", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="在本机量（agent 要先停）")
    r.add_argument("--model-dir", default=None, help="读其中的 config.json（不读权重）")
    r.add_argument("--config", default=None, help="没有 --model-dir 时用预置名，如 qwen3-30b-a3b")
    r.add_argument("--node", default=None, help="写进结果的节点 id")
    r.add_argument("--device", default="cuda")
    r.add_argument("--dtype", default=None, help="覆盖 config 的 torch_dtype（默认与服务一致）")
    r.add_argument("--layers", type=int, default=4, help="一次建几层（结果按层平均）")
    r.add_argument("--resident", default="all",
                   help="驻留专家数，逗号分隔，all = 全部；如 all,80")
    r.add_argument("--ctx", default=_fmt(DEFAULT_CTX), help="decode 的上下文长度")
    r.add_argument("--prefill", default=_fmt(DEFAULT_PREFILL),
                   help="prefill 一次喂的 token 数")
    r.add_argument("--steps", type=int, default=32, help="每个上下文长度量多少个 decode 步")
    r.add_argument("--repeats", type=int, default=5, help="每个 prefill 长度量几次")
    r.add_argument("--warmup", type=int, default=3)
    r.add_argument("--miss-policy", default="local_topk")
    r.add_argument("--out", type=Path, default=None)
    rm = sub.add_parser("remote", help="在控制机上一条命令量完所有 agent（不用 ssh、不用停 agent）")
    rm.add_argument("--agents", required=True, help="N01=ip:port,N02=ip:port,…（deploy_15.sh 的 AGENTS）")
    rm.add_argument("--mode", choices=("auto", "loaded", "synthetic"), default="auto",
                    help="auto：装着权重就量已加载的层，没装就用随机权重；loaded / synthetic 强制其一")
    rm.add_argument("--model-dir", default=None, help="控制机上读 config.json（给随机权重模式定形状）")
    rm.add_argument("--config", default=None, help="没有 config.json 时用预置名，如 qwen3-30b-a3b")
    rm.add_argument("--layers", type=int, default=4, help="随机权重模式一次建几层")
    rm.add_argument("--ctx", default=_fmt(DEFAULT_CTX))
    rm.add_argument("--prefill", default=_fmt(DEFAULT_PREFILL))
    rm.add_argument("--steps", type=int, default=32)
    rm.add_argument("--repeats", type=int, default=5)
    rm.add_argument("--warmup", type=int, default=3)
    rm.add_argument("--timeout", type=float, default=900.0, help="每台的等待上限（秒）")
    rm.add_argument("--relay", default=None, help="经中继时给 host:port")
    rm.add_argument("--out-dir", type=Path, default=Path("results/layer_bench"),
                    help="逐台结果存到这里（<节点>.json）")
    rm.add_argument("--out", type=Path, default=Path("results/layer_bench.json"))
    rm.add_argument("--key", default=DEFAULT_KEY)
    m = sub.add_parser("merge", help="合并各节点结果并打印")
    m.add_argument("files", nargs="+", type=Path)
    m.add_argument("--key", default=DEFAULT_KEY, help="规划器默认取哪一列，如 decode@1024")
    m.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if a.cmd == "run":
        cfg = _hf_config(a.model_dir, a.config)
        res = [None if x.strip() == "all" else int(x) for x in a.resident.split(",")]
        from ..runtime.layer_bench import run_bench   # 执行层，只在节点上用到 torch

        from ..runtime.layer_bench import BenchError

        try:
            out = run_bench(cfg, device=a.device, n_layers=a.layers, resident=res,
                            ctx=_ints(a.ctx), prefill=_ints(a.prefill), decode_steps=a.steps,
                            warmup=a.warmup, repeats=a.repeats, miss_policy=a.miss_policy,
                            dtype=a.dtype, node=a.node)
        except BenchError as e:
            raise SystemExit(str(e))
        text = json.dumps(out, ensure_ascii=False, indent=1)
        if a.out:
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(text, encoding="utf-8")
        g = out["gpu"]
        print(f"{a.node or out['host']}  {g.get('name')}  dtype {out['model']['dtype']}")
        for name, v in out["variants"].items():
            print(f"  驻留 {v['resident']} 个专家：")
            for c, s in v["decode"].items():
                print(f"    decode ctx={c:<5} " + (s["error"] if "error" in s else
                      f"{s['median']:.3f} ms/层（mean {s['mean']:.3f} ± {s['std']:.3f}）"))
            for t, s in v["prefill"].items():
                print(f"    prefill T={t:<5} " + (s["error"] if "error" in s else
                      f"{s['median']:.3f} ms/层（每 token {s['per_token_median']:.4f}）"))
        if not a.out:
            print(text)
        return 0

    if a.cmd == "remote":
        from .control import parse_agents

        addrs = parse_agents(a.agents)
        try:
            hf = _hf_config(a.model_dir, a.config) if (a.model_dir or a.config) else None
        except SystemExit:
            if a.mode != "loaded":
                raise
            hf = None
        relay = None
        if a.relay:
            rh, _, rp = a.relay.rpartition(":")
            relay = (rh or "127.0.0.1", int(rp))
        opts = {"ctx": _ints(a.ctx), "prefill": _ints(a.prefill), "decode_steps": a.steps,
                "warmup": a.warmup, "repeats": a.repeats, "n_layers": a.layers}
        a.out_dir.mkdir(parents=True, exist_ok=True)
        print(f"向 {len(addrs)} 个 agent 发逐层计时（mode={a.mode}），各台并行，每台最多等 {a.timeout:.0f}s …")

        def done(node, res, why):
            if res is None:
                print(f"  ✗ {node}：{why}")
                return
            (a.out_dir / f"{node}.json").write_text(
                json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
            v = next(iter(res["variants"].values()))
            src = {"loaded": "已加载权重", "synthetic": "随机权重"}[res["setup"]["source"]]
            lay = f"层 {v['layers'][0]}–{v['layers'][1]}" if "layers" in v else f"{res['setup']['n_layers']} 层"
            d = v["decode"].get(a.key.partition("@")[2]) if a.key.startswith("decode") else None
            print(f"  ✓ {node}  {res['gpu'].get('name')}  {src}（{lay}）"
                  + (f"  {a.key} = {d['median']:.3f} ms/层" if d and "median" in d else ""))

        got, bad = run_remote(addrs, hf_config=hf, mode=a.mode, opts=opts, timeout=a.timeout,
                              relay=relay, on_done=done)
        if not got:
            print("一台都没量成")
            return 1
        merged = merge(list(got.values()), key=a.key)
        _print_table(merged)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(merged, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n→ {a.out}（规划：task_deploy.py --layer-times {a.out} --layer-time-key {a.key}）")
        if bad:
            print(f"⚠ {len(bad)} 台没量成：{sorted(bad)}；规划时它们会退到同型号均值")
        return 0

    rs = [json.loads(f.read_text(encoding="utf-8")) for f in a.files]
    merged = merge(rs, key=a.key)
    _print_table(merged)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(merged, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n→ {a.out}（规划：task_deploy.py --layer-times {a.out} --layer-time-key {a.key}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
