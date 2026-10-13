"""GPU 利用率 / 显存 / 功耗：直接读 NVML（libnvidia-ml.so，驱动自带），不起子进程。

节点侧有一条硬规矩：数据面不 shell out（`test_the_data_plane_never_shells_out`），
所以不用 `nvidia-smi`；也不引入 pynvml 之类的新依赖 —— ctypes 调驱动库就够了。
读不到（没有 NVIDIA 驱动、CPU 节点）时返回 None，调用方照常工作。
"""

from __future__ import annotations

import ctypes
import os
import threading

_LIB = None            # None = 还没试；False = 试过不可用
_LOCK = threading.Lock()
_HANDLE: dict = {}


class _Util(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class _Mem(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong),
                ("used", ctypes.c_ulonglong)]


def _lib():
    global _LIB
    with _LOCK:
        if _LIB is None:
            lib = None
            for name in ("libnvidia-ml.so.1", "libnvidia-ml.so"):
                try:
                    lib = ctypes.CDLL(name)
                    break
                except OSError:
                    continue
            try:
                ok = lib is not None and lib.nvmlInit_v2() == 0
            except Exception:
                ok = False
            _LIB = lib if ok else False
        return _LIB or None


def _handle(lib):
    """本进程用的那块卡：CUDA_VISIBLE_DEVICES 的第一项（下标或 UUID），否则 0 号。"""
    key = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0].strip() or "0"
    if key in _HANDLE:
        return _HANDLE[key]
    h = ctypes.c_void_p()
    if key.isdigit():
        rc = lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(int(key)), ctypes.byref(h))
    else:
        rc = lib.nvmlDeviceGetHandleByUUID(key.encode(), ctypes.byref(h))
    _HANDLE[key] = h if rc == 0 else None
    return _HANDLE[key]


def snapshot() -> dict | None:
    """{util_pct, mem_used_mb, mem_total_mb, power_w, sm_mhz, temp_c}，读不到的项缺省。

    util_pct 是 NVML 的「过去一个采样周期（1/6–1 秒）里有 kernel 在跑的时间占比」，
    与 nvidia-smi 显示的 GPU-Util 是同一个数。"""
    lib = _lib()
    if lib is None:
        return None
    h = _handle(lib)
    if h is None:
        return None
    out: dict = {}
    u, m = _Util(), _Mem()
    p, clk, t = ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint()
    try:
        if lib.nvmlDeviceGetUtilizationRates(h, ctypes.byref(u)) == 0:
            out["util_pct"] = float(u.gpu)
        if lib.nvmlDeviceGetMemoryInfo(h, ctypes.byref(m)) == 0:
            out["mem_used_mb"] = round(m.used / 2**20, 1)
            out["mem_total_mb"] = round(m.total / 2**20, 1)
        if lib.nvmlDeviceGetPowerUsage(h, ctypes.byref(p)) == 0:
            out["power_w"] = round(p.value / 1000.0, 1)
        if lib.nvmlDeviceGetClockInfo(h, ctypes.c_int(1), ctypes.byref(clk)) == 0:  # SM
            out["sm_mhz"] = float(clk.value)
        if lib.nvmlDeviceGetTemperature(h, ctypes.c_int(0), ctypes.byref(t)) == 0:
            out["temp_c"] = float(t.value)
    except Exception:
        return out or None
    return out or None
