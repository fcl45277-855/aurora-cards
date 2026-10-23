"""B deep-dive evaluation: rank correlation between replay-based predictors and measured attention-only
sensitivity across the 8 modules of one model (perc_enc, perc_dec, enc0..2, dec0..2).

Targets: excess MSE over bf16 of the attention-only cards b_attn_* (qerr^2 - qerr_bf16^2) and, for reference, qerr.
Predictors (alpha = 1, first date of the replay files): B = flip x vgap, flip rate, A, rel_K, share_err_in_flipped,
G, params of the module's attention layers; yerr (mean layer-local error) when a diag file is given.

  python scripts/b_eval.py [--results results_small] [--diag results/_diag_layers_all314.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cards import harness as H          # noqa: E402

MODS = ["perc_enc", "perc_dec", "en0", "en1", "en2", "de0", "de1", "de2"]
STAGE = {"en0": "enc0", "en1": "enc1", "en2": "enc2", "de0": "dec0", "de1": "dec1", "de2": "dec2"}


def spearman(x, y):
    """Spearman rho with average ranks for ties (equal to scipy.stats.spearmanr)."""
    rx, ry = rankdata(x), rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None)
    ap.add_argument("--diag", default=None)
    a = ap.parse_args()
    res = Path(a.results) if a.results else ROOT / ("results" if H.MODEL == "pretrained" else f"results_{H.MODEL}")
    q = lambda n: json.load(open(res / f"{n}__dev_h2" / "summary.json"))["qerr_pct"]
    b = q("bf16")
    target = {m: q(f"b_attn_{m}") ** 2 - b ** 2 for m in MODS}
    swin = json.load(open(res / "_replay_swin.json")); sd = swin["dates"][next(iter(swin["dates"]))]
    pred = {}
    for m in MODS:
        if m in STAGE:
            s = sd[STAGE[m]]
        else:
            r = json.load(open(res / f"_replay_attn_{'decoder' if m == 'perc_dec' else 'encoder'}.json"))
            s = r["dates"][next(iter(r["dates"]))]
            s = {**s, "B_score": s["alpha_1.0"]["flip_rate"] * s["vgap_rel_flipped"]}
        a1 = s["alpha_1.0"]
        pred[m] = {"B": s["B_score"], "flip": a1["flip_rate"], "A": s["A"], "rel_K": s["rel_K"],
                   "err_in_flip": a1["share_err_in_flipped"], "G": a1["G"], "flip_x_errflip": a1["flip_rate"] * a1["share_err_in_flipped"]}
    man = H.manifest_layers()
    dims = json.loads(H.MANIFEST.read_text())["dimensions"] if H.MODEL != "small" else None
    layers = {"perc_enc": [n for n in man if n.startswith("encoder.level_agg") and ".0.0.to_" in n],
              "perc_dec": [n for n in man if n.startswith("decoder.level_decoder") and ".0.0.to_" in n],
              **{m: [n for n in man if f"backbone.{m[:2]}coder_layers.{m[2]}." in n and ".attn." in n] for m in STAGE}}
    if dims:
        for m in MODS:
            pred[m]["params"] = sum(dims[n]["in_features"] * dims[n]["out_features"] for n in layers[m])
    if a.diag and Path(a.diag).exists():
        diag = json.load(open(a.diag)); dd = next(iter(next(iter(diag.values())).keys()))
        for m in MODS:
            ys = [diag[n][dd]["yerr_tiny"] for n in layers[m] if n in diag]
            pred[m]["yerr"] = float(np.mean(ys)) if ys else float("nan")
    keys = sorted({k for p in pred.values() for k in p})
    print(f"model {H.MODEL}; target = excess MSE of attention-only cards over bf16")
    print("module    qerr   target   " + " ".join(f"{k:>12s}" for k in keys))
    for m in MODS:
        print(f"{m:9s} {q(f'b_attn_{m}'):6.2f} {target[m]:8.0f} " + " ".join(f"{pred[m].get(k, float('nan')):12.4f}" for k in keys))
    t = np.array([target[m] for m in MODS])
    print("Spearman vs target:")
    for k in keys:
        x = np.array([pred[m].get(k, np.nan) for m in MODS])
        ok = ~np.isnan(x)
        print(f"  {k:14s} rho = {spearman(x[ok], t[ok]):+.2f}  (n={int(ok.sum())})")
    out = {"model": H.MODEL, "target": target, "pred": pred,
           "spearman": {k: spearman(np.array([pred[m].get(k, np.nan) for m in MODS])[~np.isnan([pred[m].get(k, np.nan) for m in MODS])],
                                     t[~np.isnan([pred[m].get(k, np.nan) for m in MODS])]) for k in keys}}
    (res / "_b_eval.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
