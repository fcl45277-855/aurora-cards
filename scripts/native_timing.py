"""Whole-model timing on the device (reviewer point 3, 2026-10-20; FP8 and BF16-resident configs added 2026-10-24): BF16
(Aurora autocast, FP32-resident weights) and bf16_resident (backbone weights stored in BF16) versus TorchAO/mslk NVFP4 and FP8 on
the 187 backbone layers of Q0, W4A16 (weight-only) and W4A4 (NVFP4 activations). Per-6-h-step wall time of model(current)
after warm-up, peak GPU memory, resident weight bytes of the quantised set. The deployable recipe t7_mx (rotation-free
smoothing + low rank + MXFP8 activations) has no native path in this toolchain and is NOT timed here; the paper must say so.

  AQ_MODEL=pretrained AQ_PERCEIVER_CHUNK=16384 python scripts/native_timing.py [--date 2024-02-02 --repeats 10]
Output: results/_native_timing.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cards import harness as H                       # noqa: E402
from scripts.native_w4a16 import tensor_bytes         # noqa: E402


def native_config(cfg_name: str):
    """TorchAO 0.18 configs. w4a16 / w4a4: NVFP4; fp8_w8a16: FP8 E4M3 weight-only (per-row scales);
    fp8_w8a8: FP8 weights (per-row) and dynamic FP8 activations (per-row, i.e. per token); mxfp8_w8a8: MXFP8 weights and
    dynamic MXFP8 activations (E8M0 scale per 32 elements)."""
    from torchao.prototype.mx_formats import (MXDynamicActivationMXWeightConfig, NVFP4DynamicActivationNVFP4WeightConfig,
                                              NVFP4WeightOnlyConfig)
    from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, Float8WeightOnlyConfig, PerRow
    if cfg_name == "w4a16":
        return NVFP4WeightOnlyConfig()
    if cfg_name == "w4a4":
        return NVFP4DynamicActivationNVFP4WeightConfig()
    if cfg_name == "fp8_w8a16":
        return Float8WeightOnlyConfig()
    if cfg_name == "fp8_w8a8":
        return Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())
    if cfg_name == "mxfp8_w8a8":
        return MXDynamicActivationMXWeightConfig()
    raise ValueError(cfg_name)


def time_config(cfg_name: str, date: str, repeats: int, warmup: int, layers: str) -> dict:
    from torchao.quantization import quantize_
    H._paths()
    from cards.compat import prepare
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model = H.load_fp32()
    model.autocast = True
    names = set(H.layer_set(layers))
    mods = dict(model.named_modules())
    w_before = sum(tensor_bytes(mods[n].weight) for n in names)
    if cfg_name != "bf16":
        for n in names:
            mods[n].to(torch.bfloat16)
        if cfg_name != "bf16_resident":          # bf16_resident: same BF16 GEMMs, weights stored in BF16 (no FP32 copy)
            quantize_(model, native_config(cfg_name), filter_fn=lambda m, fqn: fqn in names)
    mods = dict(model.named_modules())
    w_after = sum(tensor_bytes(mods[n].weight) for n in names)
    torch.cuda.synchronize()
    mem_loaded = torch.cuda.memory_allocated() / 2 ** 30
    build_s = time.time() - t0
    batch = H.raw_batch_fast(date)
    with torch.inference_mode():
        current = prepare(model, batch)
        times = []
        for i in range(warmup + repeats):
            torch.cuda.synchronize(); t1 = time.time()
            p = model(current)
            torch.cuda.synchronize(); dt = time.time() - t1
            if i >= warmup:
                times.append(dt)
            del p
        # a full 2-step forecast as the user sees it (includes the CPU copy of the prediction)
        torch.cuda.synchronize(); t2 = time.time()
        preds, _ = H.rollout(model, date, 2, False)
        torch.cuda.synchronize(); roll_s = time.time() - t2
        del preds
    out = {"config": cfg_name, "layers": layers, "n_layers": len(names), "date": date, "warmup": warmup, "repeats": repeats,
           "step_seconds_median": statistics.median(times), "step_seconds_min": min(times), "step_seconds_all": times,
           "rollout2_seconds": roll_s, "peak_gib": torch.cuda.max_memory_allocated() / 2 ** 30, "gpu_mem_after_build_gib": mem_loaded,
           "set_weight_bytes_before": w_before, "set_weight_bytes_after": w_after, "build_seconds": build_s}
    del model, current, batch
    H.free()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2024-02-02")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--layers", default="q0:inc=backbone")
    ap.add_argument("--configs", nargs="+", default=["bf16", "w4a16", "w4a4"])
    ap.add_argument("--out", default=str(ROOT / "results" / "_native_timing.json"))
    a = ap.parse_args()
    import torchao
    frac = float(os.environ.get("AQ_MEM_FRAC", "0"))
    if frac:
        torch.cuda.set_per_process_memory_fraction(frac)
    report = {"model": H.MODEL, "device": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
              "torch": torch.__version__, "torchao": torchao.__version__, "perceiver_chunk": int(os.environ.get("AQ_PERCEIVER_CHUNK", "0")),
              "note": "t7_mx (smoothing + low rank + MXFP8 activations) is simulated only; not timed natively", "configs": {}}
    # the GPU may be shared (MIN_FREE_MIB mode): record what else ran on it, since contention inflates step times
    import subprocess
    q = lambda arg: subprocess.run(["nvidia-smi", arg, "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    report["gpu_at_start"] = {"utilization_memory": q("--query-gpu=utilization.gpu,memory.used"),
                              "compute_apps": q("--query-compute-apps=pid,used_memory")}
    for c in a.configs:
        try:
            r = time_config(c, a.date, a.repeats, a.warmup, a.layers)
        except Exception as e:                  # e.g. a layer shape a TorchAO kernel does not support
            report["configs"][c] = {"config": c, "error": f"{type(e).__name__}: {e}"}
            print(f"[{H.now()}] {c}: FAILED {type(e).__name__}: {e}", flush=True)
            H.free()
            continue
        report["configs"][c] = r
        print(f"[{H.now()}] {c}: step median {r['step_seconds_median']:.3f}s  min {r['step_seconds_min']:.3f}s  "
              f"2-step rollout {r['rollout2_seconds']:.2f}s  peak {r['peak_gib']:.2f} GiB  set weights {r['set_weight_bytes_after'] / 2 ** 20:.0f} MiB", flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    b = report["configs"].get("bf16")
    for c, r in report["configs"].items():
        if b and c != "bf16" and "error" not in r:
            print(f"{c}: step speed-up x{b['step_seconds_median'] / r['step_seconds_median']:.2f}, rollout speed-up x{b['rollout2_seconds'] / r['rollout2_seconds']:.2f}, "
                  f"peak memory {r['peak_gib']:.2f} vs {b['peak_gib']:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
