"""Moving-block bootstrap CIs (block = 3 consecutive dates) and bias-removed ERA5 excess, from records.jsonl.

Same headline definitions as cards/harness.summarize (per-record ratios averaged over date x step x var), but the
date bootstrap resamples blocks of consecutive dates because neighbouring dates are correlated (lag-1 0.13-0.39).
Per var-step excess = mean_d rmse_truth(card) / mean_d rmse_truth(fp32) - 1; the bias-removed version uses
sqrt(rmse^2 - bias^2) per record (removes the global lat-weighted mean error of each field).

  python scripts/block_ci.py --split test t7_g02_nodec_fold t5_g02_dec_fp8_fold --ref g02_svdq_a08_r32 --full
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(name, split, steps):
    p = RESULTS / f"{name}__{split}_h{steps}" / "records.jsonl"
    t = {}
    for line in p.read_text().splitlines():
        r = json.loads(line)
        t.setdefault((r["date"], r["step"], r["var"]), r)
    return t


def blocks(dates, rng, L=3):
    n = len(dates)
    out = []
    while len(out) < n:
        s = int(rng.integers(0, n - L + 1))
        out.extend(dates[s:s + L])
    return out[:n]


def metrics(card, fp, ref, ds, vs):
    ks = [(d, s, v) for d in ds for (s, v) in vs]
    f = np.array([fp[k]["rmse_truth"] for k in ks])
    m = {"qerr": float(np.mean([card[k]["rmse_ref"] for k in ks] / f)),
         "excess": float(np.mean(np.array([card[k]["rmse_truth"] for k in ks]) / f - 1))}
    if ref is not None:
        m["red"] = 1 - m["qerr"] / float(np.mean([ref[k]["rmse_ref"] for k in ks] / f))
    for s, v in vs:
        kk = [(d, s, v) for d in ds]
        c = np.mean([card[k]["rmse_truth"] for k in kk])
        b = np.mean([fp[k]["rmse_truth"] for k in kk])
        dbc = np.mean([np.sqrt(max(card[k]["rmse_truth"] ** 2 - card[k]["bias_truth"] ** 2, 0)) for k in kk])
        dbf = np.mean([np.sqrt(max(fp[k]["rmse_truth"] ** 2 - fp[k]["bias_truth"] ** 2, 0)) for k in kk])
        m[f"ex_{v}_s{s}"] = c / b - 1
        m[f"exdb_{v}_s{s}"] = dbc / dbf - 1
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cards", nargs="+")
    ap.add_argument("--split", default="test")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--ref", default=None, help="card for the paired qerr reduction (default none)")
    ap.add_argument("--block", type=int, default=3)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--full", action="store_true", help="print all var-steps")
    ap.add_argument("--results", default=None, help="results dir (default ./results)")
    a = ap.parse_args()
    global RESULTS
    if a.results:
        RESULTS = Path(a.results)
    fp = load("fp32", a.split, a.steps)
    ref = load(a.ref, a.split, a.steps) if a.ref else None
    for name in a.cards:
        card = load(name, a.split, a.steps)
        keys = [k for k in card if k in fp and (ref is None or k in ref)]
        dates = sorted({k[0] for k in keys})
        vs = sorted({(k[1], k[2]) for k in keys})
        main_m = metrics(card, fp, ref, dates, vs)
        rng = np.random.default_rng(0)
        boots = [metrics(card, fp, ref, blocks(dates, rng, a.block), vs) for _ in range(a.n)]
        ci = lambda key: tuple(100 * np.percentile([b[key] for b in boots], p) for p in (2.5, 97.5))
        print(f"== {name} ({a.split}, {len(dates)} dates, block {a.block})")
        print(f"   qerr {100 * main_m['qerr']:.2f} [{ci('qerr')[0]:.2f}, {ci('qerr')[1]:.2f}]   "
              f"ERA5 excess {100 * main_m['excess']:+.2f} [{ci('excess')[0]:+.2f}, {ci('excess')[1]:+.2f}]"
              + (f"   reduction vs {a.ref} {100 * main_m['red']:+.1f} [{ci('red')[0]:+.1f}, {ci('red')[1]:+.1f}]" if ref else ""))
        for tag, lab in (("ex", "raw"), ("exdb", "bias-removed")):
            ks = [f"{tag}_{v}_s{s}" for s, v in vs]
            worst = max(ks, key=lambda k: main_m[k])
            n_over = sum(1 for k in ks if main_m[k] > 0.02)
            n_upper_over = sum(1 for k in ks if ci(k)[1] > 2.0)
            print(f"   {lab:12s} worst {worst[len(tag) + 1:]} {100 * main_m[worst]:+.2f} [{ci(worst)[0]:+.2f}, {ci(worst)[1]:+.2f}]; "
                  f"var-steps > +2%: point {n_over}/{len(ks)}, CI upper {n_upper_over}/{len(ks)}")
            if a.full:
                print("      " + " ".join(f"{k[len(tag) + 1:]}:{100 * main_m[k]:+.1f}" for k in ks))


if __name__ == "__main__":
    main()
