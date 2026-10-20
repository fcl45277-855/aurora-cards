"""Candidate 8: do cross-step terms matter in how per-step quantization residuals reach the end of a rollout?

Along the FP32 reference trajectory x_0..x_T (T steps of 6 h) the per-step residual of a quantized model is
eps_t = F_q(x_t) - F(x_t). Injecting only eps_t into an otherwise FP32 rollout and reading the terminal difference
gives d_t (the propagated contribution of step t). First order:  e_T ~= sum_t d_t, and
||e_T||^2 = sum_t ||d_t||^2 + 2 sum_{s<t} <d_s, d_t>.  This script measures, per date:
  - d_t for every t (single-residual injections), d_all (all residuals injected), e_true (real quantized rollout)
  - linearity: ||sum_t d_t - d_all|| / ||d_all||   (expansion valid only if small)
  - cross share: (||d_all||^2 - sum_t ||d_t||^2) / ||d_all||^2
  - sign controls: all residuals negated (-eps_t) and alternating signs (+,-,+,-): norm-preserving counterfactuals
  - ERA5 RMSE of every terminal state, so chance cancellation of FP32 bias is visible
Norm M: latitude-weighted MSE per variable, divided by the FP32 vs ERA5 MSE of that variable at step T (same
normalisation as qerr), summed over the 9 variables.

  AQ_PERCEIVER_CHUNK=16384 AQ_MEM_FRAC=0.57 python scripts/crossterm.py --recipe recipes/t7_mx.json --steps 4 \
      --dates 2024-03-01 2024-03-05 2024-03-09
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


def to_cpu(b):
    return b.to("cpu")


def step(model, cur):
    with torch.inference_mode():
        p = model(cur.to("cuda"))
    nxt = dataclasses.replace(p, surf_vars={k: torch.cat([cur.surf_vars[k][:, 1:].to("cuda"), v], 1) for k, v in p.surf_vars.items()},
                              atmos_vars={k: torch.cat([cur.atmos_vars[k][:, 1:].to("cuda"), v], 1) for k, v in p.atmos_vars.items()})
    return to_cpu(p), to_cpu(nxt)


def add_eps(p, eps, sign=1.0):
    return dataclasses.replace(p, surf_vars={k: v + sign * eps["surf"][k] for k, v in p.surf_vars.items()},
                               atmos_vars={k: v + sign * eps["atmos"][k] for k, v in p.atmos_vars.items()})


def advance(p, cur):
    return dataclasses.replace(p, surf_vars={k: torch.cat([cur.surf_vars[k][:, 1:], v], 1) for k, v in p.surf_vars.items()},
                               atmos_vars={k: torch.cat([cur.atmos_vars[k][:, 1:], v], 1) for k, v in p.atmos_vars.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--dates", nargs="+", default=["2024-03-01", "2024-03-05", "2024-03-09"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    frac = float(os.environ.get("AQ_MEM_FRAC", "0"))
    if frac:
        torch.cuda.set_per_process_memory_fraction(frac)
    r = merged(json.loads(Path(a.recipe).read_text()))
    T = a.steps
    if a.out is None:
        a.out = str(ROOT / ("results" if H.MODEL == "pretrained" else f"results_{H.MODEL}") / f"_crossterm_{r['name']}_h{T}.json")
    H._paths()
    from cards.compat import prepare
    truth = H.Truth()

    # ---- pass 1: FP32 reference trajectories ---------------------------------------------------------
    fp, _ = build({"kind": "fp32"})
    ref = {}
    for d in a.dates:
        batch = H.raw_batch_fast(d)
        cur = to_cpu(prepare(fp, batch))
        states, preds = [cur], []
        for t in range(T):
            p, cur = step(fp, cur)
            preds.append(p); states.append(cur)
        ref[d] = {"states": states, "preds": preds, "lat": batch.metadata.lat}
        print(f"[{H.now()}] fp32 reference {d}", flush=True)
    lat_w = H.lat_weights(ref[a.dates[0]]["lat"])
    sw = np.sqrt(np.clip(lat_w, 0, None))        # float32 cos(90 deg) is -7e-8: clip before sqrt
    del fp; H.free()

    # ---- pass 2: quantized model: residuals on the reference states + the true quantized rollout -------
    q, _ = build(r)
    for d in a.dates:
        R = ref[d]
        R["eps"] = []
        for t in range(T):
            with torch.inference_mode():
                pq = to_cpu(q(R["states"][t].to("cuda")))
            pf = R["preds"][t]
            R["eps"].append({"surf": {k: pq.surf_vars[k] - pf.surf_vars[k] for k in pf.surf_vars},
                             "atmos": {k: pq.atmos_vars[k] - pf.atmos_vars[k] for k in pf.atmos_vars}})
        cur = R["states"][0]
        for t in range(T):
            p, cur = step(q, cur)
        R["true_terminal"] = p
        print(f"[{H.now()}] quantized residuals + rollout {d}", flush=True)
    del q; H.free()

    # ---- pass 3: FP32 again: injections --------------------------------------------------------------
    fp, _ = build({"kind": "fp32"})

    def inject(R, signs):
        """signs[t] in {0, +1, -1}: inject signs[t] * eps_t at step t, FP32 elsewhere. Returns terminal prediction."""
        cur = R["states"][0]
        for t in range(T):
            p, _ = step(fp, cur)
            if signs[t]:
                p = add_eps(p, R["eps"][t], signs[t])
            cur = advance(p, cur)
        return p

    def fields(p):
        return H.fields(p)

    def normed_diff(pa, pb, scale):
        """per-var lat-weighted field difference divided by FP32 ERA5 RMSE at step T -> dict of 2-D arrays"""
        fa, fb = fields(pa), fields(pb)
        return {v: (fa[v] - fb[v])[:720] * sw / scale[v] for v in VARS}

    def norm2(dd):
        return float(sum(np.mean(dd[v] ** 2) for v in VARS))

    def inner(da, db):
        return float(sum(np.mean(da[v] * db[v]) for v in VARS))

    def era5_rmse(p, d):
        when = np.datetime64(d) + np.timedelta64(6 * T, "h")
        f = fields(p)
        return {v: float(np.sqrt(np.mean(lat_w * (f[v][:720] - truth.get(when, v)) ** 2))) for v in VARS}

    out = {"recipe": r["name"], "model": H.MODEL, "steps": T, "dates": {}}
    for d in a.dates:
        R = ref[d]
        pT = R["preds"][T - 1]
        scale = era5_rmse(pT, d)                                  # FP32 ERA5 RMSE per var at step T
        singles = [inject(R, [1 if s == t else 0 for s in range(T)]) for t in range(T)]
        d_all = inject(R, [1] * T)
        d_neg = inject(R, [-1] * T)
        d_alt = inject(R, [1 if t % 2 == 0 else -1 for t in range(T)])
        D = [normed_diff(p, pT, scale) for p in singles]
        Dall, Dneg, Dalt = (normed_diff(p, pT, scale) for p in (d_all, d_neg, d_alt))
        Dtrue = normed_diff(R["true_terminal"], pT, scale)
        Dsum = {v: sum(Dt[v] for Dt in D) for v in VARS}
        diag = [norm2(Dt) for Dt in D]
        cross = sum(2 * inner(D[s], D[t]) for s in range(T) for t in range(s + 1, T))
        n_all = norm2(Dall)
        res = {"diag_terms": diag, "sum_diag": sum(diag), "cross_term_from_singles": cross,
               "norm2_sum_singles": norm2(Dsum), "norm2_all": n_all,
               "cross_share_all": (n_all - sum(diag)) / n_all,
               "linearity_err": float(np.sqrt(norm2({v: Dsum[v] - Dall[v] for v in VARS}) / n_all)),
               "norm2_true": norm2(Dtrue),
               "true_vs_all_err": float(np.sqrt(norm2({v: Dtrue[v] - Dall[v] for v in VARS}) / norm2(Dtrue))),
               "norm2_neg": norm2(Dneg), "norm2_alt": norm2(Dalt),
               "era5_rmse_ratio": {k: {v: era5_rmse(p, d)[v] / scale[v] - 1 for v in VARS}
                                   for k, p in (("fp32", pT), ("all", d_all), ("neg", d_neg), ("alt", d_alt), ("true", R["true_terminal"]))}}
        res["era5_excess_mean"] = {k: float(np.mean(list(v.values()))) for k, v in res["era5_rmse_ratio"].items()}
        out["dates"][d] = res
        print(f"[{H.now()}] {d}: diag {sum(diag):.4f} all {n_all:.4f} cross_share {res['cross_share_all']:+.3f} "
              f"lin_err {res['linearity_err']:.3f} true/all err {res['true_vs_all_err']:.3f} neg {norm2(Dneg):.4f} alt {norm2(Dalt):.4f}", flush=True)
    out["peak_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({d: {k: v for k, v in res.items() if k != "era5_rmse_ratio"} for d, res in out["dates"].items()}, indent=1))


if __name__ == "__main__":
    main()
