"""Per-layer activation diagnostics on the BF16 (autocast) model: why are the 6 decoder layers so sensitive?

For every Q0 layer, on the first call of step 1 of each date (first 4 x 16384 tokens), with Q0's activation path (full-width BF16 random
Hadamard, dynamic per-tensor scale per 16384-token chunk), it records:
  under   share of non-zero 16-blocks whose ideal UE4M3 block scale is below 2^-6 (TorchAO clamps those up)
  aerr    relative RMSE of the NVFP4 quantize-dequantized (rotated) input: TorchAO floor 2^-6 / subnormal floor 2^-9
  yerr    relative RMSE of the layer output with NVFP4 input and BF16 weights (same two floors), and FP8 input
  crest   max|x| / rms(x) of the rotated input; chan  max/median of per-channel rms (before rotation)
and, for the decoder layers, how much the input changes between the two dates (0 = state-independent).

  python scripts/diag_layers.py --dates 2024-02-02 2024-02-08 --out results/_diag_layers.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cards import harness as H                      # noqa: E402
from cards import nvfp4 as NV                       # noqa: E402
from cards.recipe import build                      # noqa: E402
from cards.rotation import BlockHadamard            # noqa: E402
from cards.simlinear import qdq_fp8_blocks          # noqa: E402

CHUNK = 16384
MAX_TOKENS = 4 * CHUNK                              # stats on the first 4 chunks (memory: fc1 outputs are 8192 wide)
DEC = ("decoder.level_decoder", "decoder.atmos_levels_embed")


def rel(a, b):
    return float((a.float() - b.float()).pow(2).mean().sqrt() / b.float().pow(2).mean().sqrt().clamp(min=1e-30))


def qdq_chunks(t, floor):
    NV.SCALE_MIN = floor
    out = torch.empty_like(t, dtype=torch.float32)
    for i in range(0, t.shape[0], CHUNK):
        out[i:i + CHUNK] = NV.qdq_blocks(t[i:i + CHUNK])
    NV.SCALE_MIN = NV.E4M3_TINY
    return out


def under_share(t):
    bad = tot = 0
    for i in range(0, t.shape[0], CHUNK):
        c = t[i:i + CHUNK].float()
        xb = c.reshape(-1, NV.BLOCK)
        pts = NV.per_tensor_scale(c).clamp(min=1e-30)
        a = xb.abs().amax(-1)
        nz = a > 0
        bad += int(((a / NV.E2M1_MAX / pts) < NV.E4M3_TINY)[nz].sum())
        tot += int(nz.sum())
    return bad / max(tot, 1)


@torch.no_grad()
def layer_stats(name, lin, x, rot):
    flat = x.reshape(-1, lin.in_features)[:MAX_TOKENS]       # whole 16384-token chunks, so per-chunk scales match
    w = lin.weight.detach()
    xr = rot.apply(flat.to(torch.bfloat16))
    wr = rot.rotate_weight(w.to(torch.bfloat16)).float()
    y_ref = flat.float() @ w.float().t()
    out = {"tokens": int(flat.shape[0]), "in": int(lin.in_features), "out": int(lin.out_features),
           "under": under_share(xr),
           "crest": float(xr.float().abs().max() / xr.float().pow(2).mean().sqrt().clamp(min=1e-30))}
    ch = flat.float().pow(2).mean(0).sqrt()
    out["chan"] = float(ch.max() / ch.median().clamp(min=1e-30))
    for tag, floor in (("tiny", NV.E4M3_TINY), ("sub", NV.E4M3_SUBNORMAL_MIN)):
        q = qdq_chunks(xr, floor)
        out[f"aerr_{tag}"] = rel(q, xr)
        out[f"yerr_{tag}"] = rel(q.to(torch.bfloat16).float() @ wr.t(), y_ref)
    q8 = qdq_fp8_blocks(xr)
    out["yerr_fp8"] = rel(q8.to(torch.bfloat16).float() @ wr.t(), y_ref)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="+", default=["2024-02-02", "2024-02-08"])
    ap.add_argument("--out", default=str(ROOT / "results" / "_diag_layers.json"))
    ap.add_argument("--layers", default="q0", help="layer spec to diagnose (q0 | all314 | ...)")
    a = ap.parse_args()
    model, _ = build({"kind": "bf16"})
    names = H.layer_set(a.layers)
    idx = {n: i for i, n in enumerate(H.manifest_layers(), 1)}
    mods = dict(model.named_modules())
    res, dec_inputs = {}, {}
    for d in a.dates:
        seen = set()
        hooks = []
        for n in names:
            lin = mods[n]
            rot = BlockHadamard(lin.in_features, lin.in_features, seed=H.SEED + idx[n], device="cuda",
                                compute_dtype=torch.bfloat16)

            def hook(m, args, n=n, lin=lin, rot=rot):
                if n in seen:
                    return
                seen.add(n)
                x = args[0]
                res.setdefault(n, {})[d] = layer_stats(n, lin, x, rot)
                if n.startswith(DEC):
                    dec_inputs.setdefault(n, []).append(x.detach().reshape(-1, lin.in_features)[:MAX_TOKENS].float().cpu())
            hooks.append(lin.register_forward_pre_hook(hook))
        H.rollout(model, d, 1, False)
        for h in hooks:
            h.remove()
        print(f"[{H.now()}] {d}: {len(seen)} layers", flush=True)
    for n, xs in dec_inputs.items():
        if len(xs) >= 2 and xs[0].shape == xs[1].shape:
            res[n]["date_change"] = rel(xs[1], xs[0])
    Path(a.out).write_text(json.dumps(res, indent=1))
    d0 = a.dates[0]
    rows = sorted(((n, v[d0]) for n, v in res.items() if d0 in v), key=lambda t: -t[1]["yerr_tiny"])
    print(f"{'layer':58s} tok    in  under  crest  chan  yerr_tiny yerr_sub yerr_fp8  date_change")
    for n, s in rows[:25] + [r for r in rows[25:] if r[0].startswith(DEC)]:
        dc = res[n].get("date_change")
        print(f"{n:58s} {s['tokens']:6d} {s['in']:5d} {s['under']:.3f} {s['crest']:6.1f} {s['chan']:6.1f} "
              f"{s['yerr_tiny']:.4f}   {s['yerr_sub']:.4f}   {s['yerr_fp8']:.4f}   {'' if dc is None else f'{dc:.2e}'}")


if __name__ == "__main__":
    main()
