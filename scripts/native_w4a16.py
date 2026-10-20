"""On-device NVFP4 weight-only (W4A16) Aurora with TorchAO's real NVFP4Tensor (torchao 0.18, RTX 5090 / SM 12.0),
scored like a card, to validate the simulator and to measure real weight memory and step time.

The FP32 checkpoint is loaded in Aurora's autocast mode (backbone BF16 autocast, encoder/decoder/IO FP32); the chosen
Linear layers (default: Q0's backbone layers, which run under autocast so their inputs are BF16) are replaced by
NVFP4Tensor weights via quantize_(NVFP4WeightOnlyConfig()). Simulator twin: recipes/n_sim_w4a16_bb.json (same layers,
no rotation, RTN, weight source fp32, BF16 activations). Pass criterion written in 09: |qerr(native) - qerr(sim)| small
relative to the date-bootstrap CI; the per-date rmse_ref pairs are also written for a paired comparison.

  AQ_MODEL=pretrained python scripts/native_w4a16.py [--layers q0:inc=backbone --split dev --steps 2]

Outputs results*/native_w4a16__<split>_h<steps>/{recipe.json, build_info.json, records.jsonl, summary.json}.
build_info has: n_layers, weight bytes before/after for those layers, GPU memory after build, torchao version.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cards import harness as H          # noqa: E402


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def tensor_bytes(t) -> int:
    """Storage of a plain tensor or of an NVFP4Tensor (packed codes + E4M3 block scales + FP32 per-tensor scale)."""
    if hasattr(t, "qdata"):
        n = t.qdata.numel() * t.qdata.element_size() + t.scale.numel() * t.scale.element_size()
        if getattr(t, "per_tensor_scale", None) is not None:
            n += t.per_tensor_scale.numel() * t.per_tensor_scale.element_size()
        return n
    return t.numel() * t.element_size()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="q0:inc=backbone")
    ap.add_argument("--act", default="bf16", choices=["bf16", "nvfp4"],
                    help="nvfp4 = NVFP4DynamicActivationNVFP4WeightConfig (real W4A4 GEMM; needs the mslk package)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--split", default="dev", choices=list(H.SPLITS))
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--max-dates", type=int, default=None)
    ap.add_argument("--results", default=None)
    a = ap.parse_args()
    import torchao
    from torchao.quantization import quantize_
    from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig, NVFP4WeightOnlyConfig
    a.name = a.name or ("native_w4a16" if a.act == "bf16" else "native_w4a4")
    cfg = NVFP4WeightOnlyConfig() if a.act == "bf16" else NVFP4DynamicActivationNVFP4WeightConfig()

    res = Path(a.results) if a.results else ROOT / ("results" if H.MODEL == "pretrained" else f"results_{H.MODEL}")
    tag = {"model": H.MODEL, "ckpt": H.CKPT.name}
    marker = res / "_model.json"
    if marker.exists() and json.loads(marker.read_text()) != tag:
        sys.exit(f"{res} belongs to another model")
    out = res / f"{a.name}__{a.split}_h{a.steps}"
    out.mkdir(parents=True, exist_ok=True)
    frac = float(os.environ.get("AQ_MEM_FRAC", "0"))
    if frac:
        torch.cuda.set_per_process_memory_fraction(frac)

    t0 = time.time()
    model = H.load_fp32()
    model.autocast = True
    names = set(H.layer_set(a.layers))
    mods = dict(model.named_modules())
    before = sum(tensor_bytes(mods[n].weight) for n in names)
    torch.cuda.synchronize()
    mem_fp32 = torch.cuda.memory_allocated()
    # Quantize from BF16 weights: an NVFP4Tensor made from an FP32 weight reports dtype float32, and Aurora's autocast
    # then calls aten.to(bfloat16) on it, which NVFP4Tensor does not implement (first run failed in backbone.time_mlp).
    # A BF16-dtype NVFP4Tensor needs no cast. Simulator twin: weight source "bf16" (recipes/n_sim_w4a16_bb16.json).
    for n in names:
        mods[n].to(torch.bfloat16)
    quantize_(model, cfg, filter_fn=lambda m, fqn: fqn in names)
    mods = dict(model.named_modules())
    n_q = sum(1 for n in names if hasattr(mods[n].weight, "qdata"))
    after = sum(tensor_bytes(mods[n].weight) for n in names)
    torch.cuda.synchronize()
    H.free()
    info = {"name": a.name, "layers": a.layers, "act": a.act, "config": type(cfg).__name__, "n_layers": len(names), "n_quantized": n_q, "torchao": torchao.__version__,
            "torch": torch.__version__, "device": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
            "weight_bytes_fp32": before, "weight_bytes_nvfp4": after, "ratio": before / max(after, 1),
            "gpu_mem_after_load_gib": mem_fp32 / 2 ** 30, "gpu_mem_after_quant_gib": torch.cuda.memory_allocated() / 2 ** 30,
            "build_seconds": time.time() - t0, "perceiver_chunk": int(os.environ.get("AQ_PERCEIVER_CHUNK", "0"))}
    if n_q != len(names):
        info["warning"] = f"only {n_q} of {len(names)} layers were replaced"
    (out / "build_info.json").write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
    (out / "recipe.json").write_text(json.dumps({"name": a.name, "kind": "native_torchao_w4a16", "layers": a.layers}, indent=1))
    print(json.dumps({k: info[k] for k in ("n_layers", "n_quantized", "ratio", "gpu_mem_after_load_gib", "gpu_mem_after_quant_gib")}), flush=True)

    dates = H.SPLITS[a.split][: a.max_dates]
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
            preds, lat = H.rollout(model, d, a.steps, False)
            sec = time.time() - t1
            ref = H.load_ref(cache, d, a.steps)
            if ref is None:
                sys.exit(f"no FP32 reference cache for {d}")
            for row in H.score(preds, lat, truth, ref):
                row.update(date=d, seconds=sec, peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30)
                f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"[{H.now()}] {a.name} {d}: {sec:.1f}s", flush=True)
            del preds
            H.free()
    recs = read_jsonl(rec_path)
    base = {}
    for key, nm in (("fp32", "fp32"), ("q0", "q0_sim")):
        p = res / f"{nm}__{a.split}_h{a.steps}" / "records.jsonl"
        if p.exists():
            base[key] = read_jsonl(p)
    s = H.summarize(recs, base)
    s.update(name=a.name, split=a.split, steps=a.steps, mean_seconds_per_date=sum(x["seconds"] for x in recs) / max(1, len(recs)),
             peak_gib=max((x["peak_gib"] for x in recs), default=None), build=info)
    (out / "summary.json").write_text(json.dumps(s, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: s.get(k) for k in ("name", "n_dates", "qerr_pct", "qerr_pct_ci95", "excess_pct", "mean_seconds_per_date", "peak_gib")}), flush=True)


if __name__ == "__main__":
    main()
