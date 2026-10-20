"""Native (on-device) port of the deployable recipes (reviewer point 3, user go-ahead 2026-10-20).

The simulated t7 / t7_mx layer is y = Q4(x / s) @ Wq^T * pts + (x @ A^T) @ B^T + b (cards/simlinear.py: smoothing s,
GPTQ-quantised NVFP4 weight Wq * pts, rank-32 low-rank branch on the unsmoothed input, activation Q4 = NVFP4 or MXFP8).
Here every SimNVFP4Linear of a built simulated model is replaced by NativeT7Linear:
  weight      the dequantised NVFP4 weight re-quantised by TorchAO into a real NVFP4Tensor (idempotence is measured and
              written to build_info: fraction of elements whose dequantised value differs);
  activation  --act native : TorchAO/mslk W4A4 kernel (dynamic NVFP4 activations, no scale search) -> fully native t7;
              --act mxfp8_sim : TorchAO W4A16 kernel on MXFP8-rounded BF16 inputs (the MXFP8 x NVFP4 GEMM of t7_mx has
              no kernel in this toolchain, so the activation format stays simulated while weights and GEMM are real);
  smoothing   one BF16 per-channel multiply before the GEMM; low rank: two small BF16 GEMMs; bias added in FP32.
Simulated twins for the paired comparison: rne_t7 (t7 without activation scale search) for --act native, t7_mx for
--act mxfp8_sim. Scored exactly like native_w4a16.py (dev split, FP32 reference cache); per-step timing like native_timing.py.

  AQ_MODEL=pretrained AQ_PERCEIVER_CHUNK=16384 python scripts/native_t7.py --recipe rne_t7 --act native
  AQ_MODEL=pretrained AQ_PERCEIVER_CHUNK=16384 python scripts/native_t7.py --recipe t7_mx --act mxfp8_sim
Outputs results/native_t7_<recipe>_<act>__<split>_h<steps>/{build_info.json, recipe.json, records.jsonl, summary.json}
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
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cards import harness as H                                   # noqa: E402
from cards.recipe import build                                   # noqa: E402
from cards.simlinear import SimNVFP4Linear, qdq_mxfp8            # noqa: E402
from scripts.native_w4a16 import read_jsonl, tensor_bytes        # noqa: E402


class NativeT7Linear(nn.Module):
    def __init__(self, sim: SimNVFP4Linear, cfg, act: str):
        super().__init__()
        assert sim.rotation is None, "native port covers rotation-free recipes only"
        W = (sim.wq.float() * sim.w_pts).to(torch.bfloat16)             # dequantised NVFP4 weight (BF16-exact values)
        self.lin = nn.Linear(sim.in_features, sim.out_features, bias=False, device=W.device, dtype=torch.bfloat16)
        with torch.no_grad():
            self.lin.weight.copy_(W)
        from torchao.quantization import quantize_
        quantize_(self.lin, cfg)
        self.is_quantized = hasattr(self.lin.weight, "qdata")
        # idempotence check: dequantise the NVFP4Tensor and compare with W
        with torch.no_grad():
            try:
                Wd = self.lin.weight.dequantize().to(torch.bfloat16) if hasattr(self.lin.weight, "dequantize") else None
            except Exception:
                Wd = None
            self.mismatch_frac = float((Wd != W).float().mean()) if Wd is not None else float("nan")
            self.mismatch_max_rel = float(((Wd.float() - W.float()).abs().max() / W.float().abs().max().clamp(min=1e-30))) if Wd is not None else float("nan")
        self.register_buffer("smooth_inv", None if sim.smooth is None else (1.0 / sim.smooth.float()).to(torch.bfloat16), persistent=False)
        self.register_buffer("lr_a", None if sim.lr_a is None else sim.lr_a.to(torch.bfloat16), persistent=False)
        self.register_buffer("lr_b", None if sim.lr_b is None else sim.lr_b.to(torch.bfloat16), persistent=False)
        self.register_buffer("bias", None if sim.bias is None else sim.bias.float(), persistent=False)
        self.act = act
        self.in_features, self.out_features = sim.in_features, sim.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, self.in_features)
        xs = flat.to(torch.bfloat16)
        if self.smooth_inv is not None:
            xs = xs * self.smooth_inv
        if self.act == "mxfp8_sim":
            xs = qdq_mxfp8(xs, rounding="rceil").to(torch.bfloat16)     # exact in BF16; GEMM then runs W4A16
        y = self.lin(xs).float()
        if self.lr_a is not None:
            y = y + ((flat.to(torch.bfloat16) @ self.lr_a.t()) @ self.lr_b.t()).float()
        if self.bias is not None:
            y = y + self.bias
        return y.to(x.dtype).reshape(*shape[:-1], self.out_features)


def replace_sim_layers(model: nn.Module, cfg, act: str) -> dict:
    stats = {"n": 0, "n_quantized": 0, "mismatch_frac": [], "mismatch_max_rel": [], "bytes_before": 0, "bytes_after": 0}
    mods = dict(model.named_modules())
    for name, m in list(mods.items()):
        if not isinstance(m, SimNVFP4Linear):
            continue
        stats["bytes_before"] += m.wq.numel() * 2
        nat = NativeT7Linear(m, cfg, act)
        parent_name, _, child = name.rpartition(".")
        setattr(mods[parent_name] if parent_name else model, child, nat)
        stats["n"] += 1; stats["n_quantized"] += int(nat.is_quantized)
        stats["mismatch_frac"].append(nat.mismatch_frac); stats["mismatch_max_rel"].append(nat.mismatch_max_rel)
        stats["bytes_after"] += tensor_bytes(nat.lin.weight) + sum(t.numel() * t.element_size() for t in (nat.smooth_inv, nat.lr_a, nat.lr_b) if t is not None)
    stats["mismatch_frac_mean"] = float(sum(stats["mismatch_frac"]) / max(1, len(stats["mismatch_frac"])))
    stats["mismatch_frac_max"] = float(max(stats["mismatch_frac"])) if stats["mismatch_frac"] else float("nan")
    stats["mismatch_max_rel_max"] = float(max(stats["mismatch_max_rel"])) if stats["mismatch_max_rel"] else float("nan")
    del stats["mismatch_frac"], stats["mismatch_max_rel"]
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", default="rne_t7")
    ap.add_argument("--act", default="native", choices=["native", "mxfp8_sim"])
    ap.add_argument("--split", default="dev", choices=list(H.SPLITS))
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--max-dates", type=int, default=None)
    ap.add_argument("--twin", default=None, help="simulated card to pair with (default: the recipe itself)")
    ap.add_argument("--time-repeats", type=int, default=5)
    ap.add_argument("--results", default=None)
    a = ap.parse_args()
    import torchao
    from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig, NVFP4WeightOnlyConfig
    cfg = NVFP4DynamicActivationNVFP4WeightConfig() if a.act == "native" else NVFP4WeightOnlyConfig()
    frac = float(os.environ.get("AQ_MEM_FRAC", "0"))
    if frac:
        torch.cuda.set_per_process_memory_fraction(frac)
    res = Path(a.results) if a.results else ROOT / ("results" if H.MODEL == "pretrained" else f"results_{H.MODEL}")
    name = f"native_t7_{a.recipe}_{a.act}"
    out = res / f"{name}__{a.split}_h{a.steps}"
    out.mkdir(parents=True, exist_ok=True)
    rec = json.loads((ROOT / "recipes" / f"{a.recipe}.json").read_text(encoding="utf-8"))
    t0 = time.time()
    model, binfo = build(rec)                       # simulated model (calibration, GPTQ, smoothing, low rank)
    t_build_sim = time.time() - t0
    stats = replace_sim_layers(model, cfg, a.act)
    torch.cuda.synchronize(); H.free()
    info = {"name": name, "recipe": a.recipe, "act": a.act, "config": type(cfg).__name__, "torchao": torchao.__version__, "torch": torch.__version__,
            "device": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(), "sim_build_seconds": t_build_sim,
            "convert_seconds": time.time() - t0 - t_build_sim, "gpu_mem_after_build_gib": torch.cuda.memory_allocated() / 2 ** 30,
            "perceiver_chunk": int(os.environ.get("AQ_PERCEIVER_CHUNK", "0")), **stats,
            "note": ("fully native W4A4: NVFP4 weights + mslk dynamic NVFP4 activations; smoothing and low rank in BF16" if a.act == "native"
                     else "NVFP4 weights with the W4A16 kernel; MXFP8 activations rounded in software (no NVFP4 x MXFP8 kernel in this toolchain)")}
    (out / "build_info.json").write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
    (out / "recipe.json").write_text(json.dumps({"name": name, "kind": "native_t7", "recipe": a.recipe, "act": a.act}, indent=1))
    print(json.dumps({k: info[k] for k in ("n", "n_quantized", "mismatch_frac_mean", "mismatch_frac_max", "bytes_before", "bytes_after", "gpu_mem_after_build_gib")}), flush=True)

    # per-step timing (same protocol as native_timing.py)
    H._paths()
    from cards.compat import prepare
    dates = H.SPLITS[a.split][: a.max_dates]
    batch = H.raw_batch_fast(dates[0])
    with torch.inference_mode():
        current = prepare(model, batch)
        times = []
        for i in range(2 + a.time_repeats):
            torch.cuda.synchronize(); t1 = time.time()
            p = model(current)
            torch.cuda.synchronize(); dt = time.time() - t1
            if i >= 2:
                times.append(dt)
            del p
    del current, batch
    H.free()
    info["step_seconds_median"] = statistics.median(times); info["step_seconds_all"] = times
    (out / "build_info.json").write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
    print(f"[{H.now()}] step median {info['step_seconds_median']:.3f}s", flush=True)

    cache = res / f"_ref_fp32_h{a.steps}"
    rec_path = out / "records.jsonl"
    done = {x["date"] for x in read_jsonl(rec_path)}
    truth = H.Truth()
    torch.cuda.reset_peak_memory_stats()
    with rec_path.open("a", encoding="utf-8") as f:
        for d in dates:
            if d in done:
                continue
            t1 = time.time()
            preds, lat = H.rollout(model, d, a.steps, bool(rec.get("surface_fp32", False)))
            sec = time.time() - t1
            ref = H.load_ref(cache, d, a.steps)
            if ref is None:
                sys.exit(f"no FP32 reference cache for {d}")
            for row in H.score(preds, lat, truth, ref):
                row.update(date=d, seconds=sec, peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30)
                f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"[{H.now()}] {name} {d}: {sec:.1f}s", flush=True)
            del preds
            H.free()
    recs = read_jsonl(rec_path)
    base = {}
    for key, nm in (("fp32", "fp32"), ("q0", "q0_sim")):
        p = res / f"{nm}__{a.split}_h{a.steps}" / "records.jsonl"
        if p.exists():
            base[key] = read_jsonl(p)
    s = H.summarize(recs, base)
    s.update(name=name, split=a.split, steps=a.steps, mean_seconds_per_date=sum(x["seconds"] for x in recs) / max(1, len(recs)))
    # paired comparison with the simulated twin
    twin = a.twin or a.recipe
    tp = res / f"{twin}__{a.split}_h{a.steps}" / "records.jsonl"
    if tp.exists():
        tw = {(x["date"], x["var"], x["step"]): x for x in read_jsonl(tp)}
        pairs = [(x["rmse_ref"], tw[(x["date"], x["var"], x["step"])]["rmse_ref"]) for x in recs if (x["date"], x["var"], x["step"]) in tw and "rmse_ref" in x]
        if pairs:
            rel = [(n - t) / t for n, t in pairs if t > 0]
            s["twin"] = twin
            s["twin_pairs"] = len(rel)
            s["twin_rel_diff_mean"] = sum(rel) / len(rel)
            s["twin_rel_diff_max_abs"] = max(abs(r) for r in rel)
            s["twin_qerr_pct"] = json.loads((res / f"{twin}__{a.split}_h{a.steps}" / "summary.json").read_text(encoding="utf-8")).get("qerr_pct") if (res / f"{twin}__{a.split}_h{a.steps}" / "summary.json").exists() else None
    (out / "summary.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
    print(json.dumps({k: s.get(k) for k in ("name", "qerr_pct", "excess_pct", "twin", "twin_qerr_pct", "twin_rel_diff_mean", "twin_rel_diff_max_abs", "mean_seconds_per_date")}), flush=True)


if __name__ == "__main__":
    main()
