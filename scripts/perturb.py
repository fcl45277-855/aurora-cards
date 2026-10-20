"""Candidate 9: does quantization distort the model's response to physical perturbations?

For each date d and amplitude alpha, the perturbation is delta = alpha * (x_ref(d_next) - x_ref(d)) (the difference
between two real analyses 24 h apart: a flow-dependent, physically shaped perturbation), applied to both history
slots of the initial Batch. One-step paired response R = F(x + delta) - F(x - delta) is computed for every recipe
(the FP32 reference first), and compared per variable with latitude weights:
  resp_err   = ||R_q - R_fp|| / ||R_fp||        (response distortion)
  gain       = ||R_q|| / ||R_fp||               (spread amplification / damping)
  pattern_cc = corr(R_q, R_fp)                  (direction fidelity)
  base_err   = ||F_q(x) - F_fp(x)|| / ||R_fp||  (ordinary one-step error, in units of the response)
Amplitudes default to 0.01, 0.03, 0.1, 0.3, 1.0 of the 24-h analysis difference. Pre-registered reading:
distortion that appears only at alpha <= 0.03 (sub-physical) is not a finding; a recipe is "response-faithful" if
resp_err <= 2 * base_err at alpha >= 0.1.

  AQ_PERCEIVER_CHUNK=16384 AQ_MEM_FRAC=0.57 python scripts/perturb.py --recipes recipes/bf16.json recipes/t7_a16.json \
      recipes/t7_mx.json --dates 2024-03-01 2024-03-05 2024-03-09
"""
import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cards import harness as H                      # noqa: E402
from cards.recipe import build, merged              # noqa: E402

VARS = H.VARS


def perturbed(cur, other, alpha, sign):
    """cur + sign * alpha * (other - cur) on all surf/atmos history slots (static vars untouched)."""
    return dataclasses.replace(
        cur, surf_vars={k: v + sign * alpha * (other.surf_vars[k] - v) for k, v in cur.surf_vars.items()},
        atmos_vars={k: v + sign * alpha * (other.atmos_vars[k] - v) for k, v in cur.atmos_vars.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipes", nargs="+", required=True)
    ap.add_argument("--dates", nargs="+", default=["2024-03-01", "2024-03-05", "2024-03-09"])
    ap.add_argument("--alphas", default="0.01,0.03,0.1,0.3,1.0")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    frac = float(os.environ.get("AQ_MEM_FRAC", "0"))
    if frac:
        torch.cuda.set_per_process_memory_fraction(frac)
    alphas = [float(x) for x in a.alphas.split(",")]
    if a.out is None:
        a.out = str(ROOT / ("results" if H.MODEL == "pretrained" else f"results_{H.MODEL}") / "_perturb.json")
    H._paths()
    from cards.compat import prepare

    def next_day(d):
        return str(np.datetime64(d) + np.timedelta64(1, "D"))

    # initial states (prepared batches) on CPU, once
    fp, _ = build({"kind": "fp32"})
    init = {}
    for d in a.dates:
        b0, b1 = H.raw_batch_fast(d), H.raw_batch_fast(next_day(d))
        init[d] = (prepare(fp, b0).to("cpu"), prepare(fp, b1).to("cpu"), b0.metadata.lat)
    lat_w = H.lat_weights(init[a.dates[0]][2])

    def run(model, cur):
        with torch.inference_mode():
            return H.fields(model(cur.to("cuda")).to("cpu"))

    def responses(model):
        out = {}
        for d in a.dates:
            cur, oth, _ = init[d]
            base = run(model, cur)
            for al in alphas:
                fp_, fm = run(model, perturbed(cur, oth, al, +1)), run(model, perturbed(cur, oth, al, -1))
                out[(d, al)] = {v: (fp_[v] - fm[v])[:720] for v in VARS}
            out[(d, "base")] = {v: base[v][:720] for v in VARS}
        return out

    ref = responses(fp)
    del fp; H.free()
    w = lat_w
    nrm = lambda f: float(np.sqrt(np.mean(w * f ** 2)))
    results = {"dates": a.dates, "alphas": alphas, "recipes": {}}
    for rp in a.recipes:
        r = merged(json.loads(Path(rp).read_text()))
        if r["kind"] == "fp32":
            continue
        m, _ = build(r)
        resp = responses(m)
        del m; H.free()
        rows = {}
        for al in alphas:
            per_var = {}
            for v in VARS:
                e = g = c = b = 0.0
                for d in a.dates:
                    R, Rq = ref[(d, al)][v], resp[(d, al)][v]
                    nR = nrm(R)
                    e += nrm(Rq - R) / nR; g += nrm(Rq) / nR
                    c += float(np.sum(w * R * Rq) / np.sqrt(np.sum(w * R * R) * np.sum(w * Rq * Rq)))
                    b += nrm(resp[(d, "base")][v] - ref[(d, "base")][v]) / nR
                n = len(a.dates)
                per_var[v] = {"resp_err": e / n, "gain": g / n, "pattern_cc": c / n, "base_err": b / n}
            rows[al] = {"per_var": per_var,
                        "mean": {k: float(np.mean([per_var[v][k] for v in VARS])) for k in ("resp_err", "gain", "pattern_cc", "base_err")}}
            mm = rows[al]["mean"]
            print(f"[{H.now()}] {r['name']:22s} alpha {al:<5}: resp_err {mm['resp_err']:.3f}  gain {mm['gain']:.3f}  "
                  f"cc {mm['pattern_cc']:.3f}  base_err {mm['base_err']:.3f}", flush=True)
        results["recipes"][r["name"]] = rows
    results["peak_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
    Path(a.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
