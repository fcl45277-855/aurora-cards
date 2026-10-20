"""Where does a 6-h step spend its GPU time? (follow-up to native_timing.py, 2026-10-20; v2 leaf-only attribution)
torch.profiler over a few forward steps for BF16 and native W4A4 (q0 backbone set). Device (CUDA) time is attributed to
LEAF modules only (no nesting, no double counting): quantised linear layers, other backbone linears (non-quantised),
backbone norms/activations, encoder leaves, decoder leaves. The whole forward is wrapped in one range, so
'unattributed' = total - sum(leaves) = functional ops outside leaf modules (window-attention matmuls and softmax,
shifts/rolls, patch merging, residual adds, concatenations). Per-step milliseconds and shares are written to
results/_profile_step.json together with the top kernels.

  AQ_MODEL=pretrained AQ_PERCEIVER_CHUNK=16384 python scripts/profile_step.py [--steps 3 --configs bf16 w4a4]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cards import harness as H                        # noqa: E402

CATS = ("quant_linear", "backbone_linear_other", "backbone_norm_act", "encoder_leaf", "decoder_leaf", "other_leaf", "unattributed")


def category(name: str, m: nn.Module, qnames: set) -> str:
    if name in qnames:
        return "quant_linear"
    if name.startswith("backbone"):
        if isinstance(m, nn.Linear):
            return "backbone_linear_other"
        return "backbone_norm_act"
    if name.startswith("encoder"):
        return "encoder_leaf"
    if name.startswith("decoder"):
        return "decoder_leaf"
    return "other_leaf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--configs", nargs="+", default=["bf16", "w4a4"])
    ap.add_argument("--layers", default="q0:inc=backbone")
    ap.add_argument("--date", default="2024-02-02")
    ap.add_argument("--out", default=str(ROOT / "results" / "_profile_step.json"))
    a = ap.parse_args()
    from torch.profiler import profile, ProfilerActivity, record_function
    frac = float(os.environ.get("AQ_MEM_FRAC", "0"))
    if frac:
        torch.cuda.set_per_process_memory_fraction(frac)
    H._paths()
    from cards.compat import prepare
    report = {"model": H.MODEL, "device": torch.cuda.get_device_name(), "steps": a.steps, "attribution": "leaf modules only; unattributed = total - leaves", "configs": {}}
    for cfg_name in a.configs:
        model = H.load_fp32(); model.autocast = True
        qnames = set(H.layer_set(a.layers)) if cfg_name != "bf16" else set()
        if cfg_name == "w4a4":
            from torchao.quantization import quantize_
            from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig
            mods = dict(model.named_modules())
            for n in qnames:
                mods[n].to(torch.bfloat16)
            quantize_(model, NVFP4DynamicActivationNVFP4WeightConfig(), filter_fn=lambda m, fqn: fqn in qnames)
        handles = []
        n_leaf = 0
        for n, m in model.named_modules():
            if n == "" or len(list(m.children())) > 0:
                continue
            cat = category(n, m, qnames)
            n_leaf += 1
            def pre(mod, inp, cat=cat):
                mod._rf = record_function(f"CAT::{cat}"); mod._rf.__enter__()
            def post(mod, inp, out):
                mod._rf.__exit__(None, None, None)
            handles.append(m.register_forward_pre_hook(pre)); handles.append(m.register_forward_hook(post))
        batch = H.raw_batch_fast(a.date)
        with torch.inference_mode():
            current = prepare(model, batch)
            for _ in range(2):
                p = model(current); del p
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
                for _ in range(a.steps):
                    with record_function("CAT::total"):
                        p = model(current)
                    del p
                torch.cuda.synchronize()
        ev = prof.key_averages()
        cat_ms = {c: 0.0 for c in CATS}; total = 0.0
        for e in ev:
            if e.key == "CAT::total":
                total = e.device_time_total / 1000.0 / a.steps
            elif e.key.startswith("CAT::"):
                cat_ms[e.key[5:]] += e.device_time_total / 1000.0 / a.steps
        leaves = sum(v for k, v in cat_ms.items() if k != "unattributed")
        cat_ms["unattributed"] = max(total - leaves, 0.0)
        top = sorted([e for e in ev if not e.key.startswith("CAT::")], key=lambda e: -e.device_time_total)[:15]
        report["configs"][cfg_name] = {
            "n_leaf_modules": n_leaf, "total_device_ms_per_step": total,
            "category_ms_per_step": cat_ms, "category_share": {c: (v / total if total else 0) for c, v in cat_ms.items()},
            "top_ops": [{"name": e.key[:90], "device_ms_per_step": e.device_time_total / 1000.0 / a.steps, "cpu_ms_per_step": e.cpu_time_total / 1000.0 / a.steps, "calls_per_step": e.count / a.steps} for e in top]}
        print(f"[{H.now()}] {cfg_name}: total {total:.0f} ms/step; " + ", ".join(f"{c} {v:.0f}" for c, v in cat_ms.items()), flush=True)
        for h in handles:
            h.remove()
        del model, current, batch, prof
        H.free()
        Path(a.out).write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
