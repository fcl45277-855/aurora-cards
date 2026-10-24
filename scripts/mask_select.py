"""Same-budget protection mask chosen on the corrected pipeline (selection of the protected layers on the corrected pipeline).

Pool: the 198 layers of t7_a16 plus the 60 Swin and patch layers it inherits as BF16 from Q0 (the encoder surface MLP
stays BF16 as in t7_a16). For every pool layer, the relative output error of its W4A16 weights (smoothing 0.8, rank-32
branch, GPTQ; taken from the cached build of t7a16_all314_noenc, which contains all 198) is measured on the FP32
model's own inputs: e = sum ||(W_eff - W) x||^2 / sum ||W x||^2 over two calibration dates, one 6-h step each.
Layers are protected (kept BF16) in order of decreasing e until the protected parameters reach those of the 60
inherited layers (303.8 M). Writes recipes/t7_a16_maskE_layers.txt (the quantised layers), recipes/t7_a16_maskE.json
and results/_mask_select.json.

  AQ_MODEL=pretrained AQ_PERCEIVER_CHUNK=16384 python scripts/mask_select.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cards import harness as H          # noqa: E402
from cards.recipe import build          # noqa: E402
from cards.simlinear import SimNVFP4Linear   # noqa: E402

CAL_DATES = ["2024-01-04", "2024-01-07"]


def main():
    dims = json.loads(H.MANIFEST.read_text(encoding="utf-8"))["dimensions"]
    npar = lambda n: dims[n]["in_features"] * dims[n]["out_features"]
    t7 = H.layer_set(json.loads((ROOT / "recipes" / "t7_a16.json").read_text())["layers"])
    allv = H.layer_set(json.loads((ROOT / "recipes" / "t7a16_all314_noenc.json").read_text())["layers"])
    inherited = [n for n in allv if n not in set(t7) and not n.startswith("encoder.surf_mlp")]
    pool = list(t7) + inherited
    budget = sum(npar(n) for n in inherited)
    print(f"pool {len(pool)} layers, inherited {len(inherited)} ({budget / 1e6:.1f} M params)", flush=True)

    qmodel, _ = build(json.loads((ROOT / "recipes" / "t7a16_all314_noenc.json").read_text()))
    weff = {}
    for n, m in qmodel.named_modules():
        if n in set(pool) and isinstance(m, SimNVFP4Linear):
            w = m.wq.float() * m.w_pts
            if m.smooth is not None:
                w = w / m.smooth.float()[None, :]
            if m.lr_a is not None:
                w = w + m.lr_b.float() @ m.lr_a.float()
            weff[n] = w.cpu()
    del qmodel
    H.free()
    assert set(weff) == set(pool), set(pool) - set(weff)

    model = H.load_fp32()
    acc = {n: [0.0, 0.0] for n in pool}
    mods = dict(model.named_modules())

    def hook(name):
        def f(mod, inp, out):
            x = inp[0].reshape(-1, inp[0].shape[-1]).float()
            w = mod.weight.float()
            we = weff[name].to(x.device)
            for i in range(0, x.shape[0], 16384):
                xc = x[i:i + 16384]
                y = xc @ w.t()
                acc[name][0] += float(((xc @ we.t() - y) ** 2).sum())
                acc[name][1] += float((y ** 2).sum())
        return f

    hs = [mods[n].register_forward_hook(hook(n)) for n in pool]
    with torch.inference_mode():
        for d in CAL_DATES:
            H.rollout(model, d, 1, False)
            print(f"[{H.now()}] calibration date {d} done", flush=True)
    for h in hs:
        h.remove()
    err = {n: acc[n][0] / max(acc[n][1], 1e-30) for n in pool}
    order = sorted(pool, key=lambda n: -err[n])
    protect, s = [], 0
    for n in order:
        if s >= budget:
            break
        protect.append(n)
        s += npar(n)
    quant = [n for n in pool if n not in set(protect)]
    (ROOT / "recipes" / "t7_a16_maskE_layers.txt").write_text("\n".join(quant) + "\n", encoding="utf-8")
    rec = json.loads((ROOT / "recipes" / "t7_a16.json").read_text())
    rec.update(name="t7_a16_maskE", layers="recipes/t7_a16_maskE_layers.txt",
               _note="t7_a16 with the 60-layer inherited BF16 mask replaced by a same-budget mask chosen by relative "
                     "W4A16 output error on the corrected pipeline (scripts/mask_select.py)")
    (ROOT / "recipes" / "t7_a16_maskE.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    out = {"calibration_dates": CAL_DATES, "budget_params": budget, "protected_params": s, "n_protected": len(protect),
           "n_quantised": len(quant), "overlap_with_inherited": len(set(protect) & set(inherited)),
           "protected": protect, "rel_err": err}
    (ROOT / "results" / "_mask_select.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("budget_params", "protected_params", "n_protected", "n_quantised",
                                          "overlap_with_inherited")}), flush=True)


if __name__ == "__main__":
    main()
