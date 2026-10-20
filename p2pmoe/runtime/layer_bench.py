"""逐层计时的执行部分：在本机 GPU 上建几层随机权重的 Qwen3-MoE，量每层耗时。

命令行、合并与规划器接入在 `p2pmoe/deploy/layer_bench.py`（控制面，不碰 torch）；
这里是唯一 import torch 的部分，属于执行层（与 torch_model.py 同层）。
"""

from __future__ import annotations

import os
import socket
import statistics as st
import time
from typing import Mapping, Sequence

__all__ = ["synth_weights", "run_bench", "bench_loaded", "BenchError",
           "DEFAULT_CTX", "DEFAULT_PREFILL"]

DEFAULT_CTX = (128, 1024, 2048)
DEFAULT_PREFILL = (128, 512, 2048)


# --------------------------------------------------------------------------- #
def _stats(xs: Sequence[float]) -> dict:
    xs = sorted(float(x) for x in xs)
    n = len(xs)
    return {
        "median": round(st.median(xs), 4),
        "mean": round(st.mean(xs), 4),
        "std": round(st.stdev(xs), 4) if n > 1 else 0.0,
        "p90": round(xs[min(n - 1, int(0.9 * (n - 1) + 0.5))], 4),
        "min": round(xs[0], 4),
        "n": n,
    }


def synth_weights(cfg, layers: Sequence[int], experts: Sequence[int], *,
                  device: str, dtype, seed: int = 0) -> dict:
    """同形状的随机权重（std 0.02，norm 取 1）。key 与 checkpoint 一致，0-based。"""
    import torch

    g = torch.Generator(device="cpu").manual_seed(seed)
    d, hd = cfg.d_model, cfg.head_dim
    q, kv, m = cfg.n_heads * hd, cfg.n_kv_heads * hd, cfg.moe_intermediate

    def rnd(*shape):
        return (torch.randn(*shape, generator=g) * 0.02).to(device=device, dtype=dtype)

    def ones(*shape):
        return torch.ones(*shape, device=device, dtype=dtype)

    w: dict = {}
    for l in layers:
        p = f"model.layers.{int(l) - 1}"
        w[f"{p}.self_attn.q_proj.weight"] = rnd(q, d)
        w[f"{p}.self_attn.k_proj.weight"] = rnd(kv, d)
        w[f"{p}.self_attn.v_proj.weight"] = rnd(kv, d)
        w[f"{p}.self_attn.o_proj.weight"] = rnd(d, q)
        w[f"{p}.self_attn.q_norm.weight"] = ones(hd)
        w[f"{p}.self_attn.k_norm.weight"] = ones(hd)
        w[f"{p}.input_layernorm.weight"] = ones(d)
        w[f"{p}.post_attention_layernorm.weight"] = ones(d)
        w[f"{p}.mlp.gate.weight"] = rnd(cfg.n_experts, d)
        for e in experts:
            w[f"{p}.mlp.experts.{e}.gate_proj.weight"] = rnd(m, d)
            w[f"{p}.mlp.experts.{e}.up_proj.weight"] = rnd(m, d)
            w[f"{p}.mlp.experts.{e}.down_proj.weight"] = rnd(d, m)
    return w


def _layer_bytes(cfg, n_resident: int, dtype_bytes: int) -> int:
    d, hd, m = cfg.d_model, cfg.head_dim, cfg.moe_intermediate
    attn = (2 * cfg.n_heads * hd + 2 * cfg.n_kv_heads * hd) * d
    return (attn + cfg.n_experts * d + n_resident * 3 * d * m) * dtype_bytes


def _gpu_info(device: str) -> dict:
    import torch

    info = {"device": device, "torch": torch.__version__,
            "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF")}
    if device.startswith("cuda") and torch.cuda.is_available():
        i = torch.device(device).index or 0
        p = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        info.update(name=p.name, capability=f"{p.major}.{p.minor}",
                    total_gib=round(total / 2**30, 2), free_gib=round(free / 2**30, 2),
                    sm_count=p.multi_processor_count)
    else:
        info.update(name="cpu")
    return info


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        import torch

        torch.cuda.synchronize()


class BenchError(RuntimeError):
    """量不了（显存不够、没有模型可量）。消息里写着该怎么办。"""


def _header(cfg, device: str, node: str | None, setup: dict) -> dict:
    return {"node": node, "host": socket.gethostname(), "gpu": _gpu_info(device),
            "model": {"d_model": cfg.d_model, "n_experts": cfg.n_experts, "top_k": cfg.top_k,
                      "moe_intermediate": cfg.moe_intermediate, "n_heads": cfg.n_heads,
                      "n_kv_heads": cfg.n_kv_heads, "head_dim": cfg.head_dim,
                      "dtype": cfg.dtype},
            "setup": {**setup, "fast_decode": os.environ.get("P2P_FAST_DECODE", "1") != "0"},
            "variants": {}}


def _measure(model, cfg, device: str, *, ctx, prefill, decode_steps: int, warmup: int,
             repeats: int, seed: int = 0, tag: str = "__bench__") -> dict:
    """对一个已经建好的 TorchSegmentModel 计时。每个点单独兜底：某个长度 OOM 不影响别的点。

    输入是随机 hidden state：每 token 激活 top-k 个专家，计算量与输入内容无关。
    """
    import torch

    n_layers = len(model.layers)
    tdt = cfg.torch_dtype
    gen = torch.Generator(device="cpu").manual_seed(seed + 1)

    def x(t: int):
        return torch.randn(t, cfg.d_model, generator=gen).to(device=device, dtype=tdt)

    def oom(e) -> dict:
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        return {"error": f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"}

    v: dict = {"decode": {}, "prefill": {}}
    with torch.inference_mode():
        for c in ctx:
            req = f"{tag}ctx{c}"
            try:
                model.forward(req, x(c))                      # 建上下文，不计时
                for _ in range(warmup):
                    model.forward(req, x(1), decode=True)
                _sync(device)
                ts = []
                for _ in range(decode_steps):
                    xi = x(1)
                    t0 = time.perf_counter()
                    model.forward(req, xi, decode=True)
                    _sync(device)
                    ts.append((time.perf_counter() - t0) * 1e3 / n_layers)
                v["decode"][str(c)] = _stats(ts)
            except (RuntimeError, MemoryError) as e:          # torch.OutOfMemoryError ⊂ RuntimeError
                v["decode"][str(c)] = oom(e)
            finally:
                model.drop_kv(req)
        for t in prefill:
            try:
                for i in range(warmup):
                    model.forward(f"{tag}w{t}-{i}", x(t))
                    model.drop_kv(f"{tag}w{t}-{i}")
                _sync(device)
                ts = []
                for i in range(repeats):
                    xi = x(t)
                    t0 = time.perf_counter()
                    model.forward(f"{tag}p{t}-{i}", xi)
                    _sync(device)
                    ts.append((time.perf_counter() - t0) * 1e3 / n_layers)
                    model.drop_kv(f"{tag}p{t}-{i}")
                s = _stats(ts)
                s["per_token_median"] = round(s["median"] / t, 5)
                v["prefill"][str(t)] = s
            except (RuntimeError, MemoryError) as e:
                v["prefill"][str(t)] = oom(e)
            finally:
                for r in [r for r in model.active_requests() if r.startswith(tag)]:
                    model.drop_kv(r)
    if device.startswith("cuda"):
        v["peak_mem_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    return v


def run_bench(hf_cfg: Mapping, *, device: str = "cuda", n_layers: int = 4,
              resident: Sequence[int | None] = (None,), ctx: Sequence[int] = DEFAULT_CTX,
              prefill: Sequence[int] = DEFAULT_PREFILL, decode_steps: int = 32,
              warmup: int = 3, repeats: int = 5, miss_policy: str = "local_topk",
              dtype: str | None = None, seed: int = 0, node: str | None = None) -> dict:
    """随机权重：在本机建 n_layers 层同形状的模型来量。时间一律是**每层**的毫秒数。"""
    import torch

    from .torch_model import TorchModelConfig, TorchSegmentModel

    cfg = TorchModelConfig.from_hf(hf_cfg)
    if dtype:
        import dataclasses

        cfg = dataclasses.replace(cfg, dtype=dtype)
    tdt = cfg.torch_dtype
    dbytes = torch.tensor([], dtype=tdt).element_size()
    layers = list(range(1, n_layers + 1))
    out = _header(cfg, device, node, {"source": "synthetic", "n_layers": n_layers,
                                      "decode_steps": decode_steps, "warmup": warmup,
                                      "repeats": repeats, "miss_policy": miss_policy})
    free = out["gpu"].get("free_gib")
    for res in resident:
        n_res = cfg.n_experts if res is None else max(cfg.top_k, min(int(res), cfg.n_experts))
        need = n_layers * _layer_bytes(cfg, n_res, dbytes)
        if free is not None and need / 2**30 * 1.3 + 0.5 > free:
            raise BenchError(
                f"显存不够：{n_layers} 层 × {n_res} 个专家约 {need / 2**30:.1f}GiB，"
                f"这张卡只剩 {free:.1f}GiB 空闲 —— agent 已经装着权重的话用 --mode loaded"
                f"（直接量已加载的层），否则先 bash ./deploy_15.sh stop 或把 --layers 调小")
        experts = list(range(n_res))
        w = synth_weights(cfg, layers, experts, device=device, dtype=tdt, seed=seed)
        model = TorchSegmentModel(cfg, {l: experts for l in layers}, w, device=device,
                                  miss_policy=miss_policy)
        v = _measure(model, cfg, device, ctx=ctx, prefill=prefill, decode_steps=decode_steps,
                     warmup=warmup, repeats=repeats, seed=seed)
        v["resident"] = n_res
        out["variants"]["all" if res is None else str(n_res)] = v
        del model, w
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    return out


def bench_loaded(model, *, node: str | None = None, ctx: Sequence[int] = DEFAULT_CTX,
                 prefill: Sequence[int] = DEFAULT_PREFILL, decode_steps: int = 32,
                 warmup: int = 3, repeats: int = 5, seed: int = 0) -> dict:
    """已加载的真实权重：直接量 agent 上这一段（它在清单里分到的那几层、那些专家）。

    不新建任何权重，只多占计时所需的激活与 KV —— agent 装满显存时也量得了。
    """
    if not hasattr(model, "blocks") or not hasattr(model, "cfg"):
        raise BenchError("本节点装的不是 Torch 执行层（toy 模型），没有可量的 GPU 层")
    if model.active_requests():
        raise BenchError(f"本节点还有 {len(model.active_requests())} 条在途请求 —— "
                         f"等测试跑完再量，否则两边都不准")
    device = str(model.device)
    cfg = model.cfg
    out = _header(cfg, device, node, {"source": "loaded", "n_layers": len(model.layers),
                                      "decode_steps": decode_steps, "warmup": warmup,
                                      "repeats": repeats, "miss_policy": model.miss_policy})
    if device.startswith("cuda"):
        import torch

        torch.cuda.reset_peak_memory_stats()
    v = _measure(model, cfg, device, ctx=ctx, prefill=prefill, decode_steps=decode_steps,
                 warmup=warmup, repeats=repeats, seed=seed)
    sizes = [len(b.resident) for b in model.blocks.values()]
    v["resident"] = round(sum(sizes) / len(sizes), 1)
    v["layers"] = [model.layers[0], model.layers[-1]]
    out["variants"]["all"] = v
    return out
