"""PhysMetrics.Weather-style metrics (Kasteleyn et al., arXiv:2606.10642; code github.com/Emmakast/PhysMetrics.Weather,
MIT) recomputed with their definitions on our rollouts, so the paper can report them next to our own diagnostics:

  hydrostatic_rmse   area-weighted RMSE over the whole grid of [Phi(500) - Phi(850)] - R_d * Tv_mean * ln(850/500),
                     Tv = T (1 + (R_v/R_d - 1) q), Tv_mean = mean of the two levels (their compute_hydrostatic_imbalance)
  geostrophic_rmse   500 hPa: u_g = -(dPhi/dphi)/(f a), v_g = (dPhi/dlambda)/(f a cos phi); dPhi/dphi by central
                     differences in latitude (degrees -> radians), dPhi/dlambda by central differences with wrap-around;
                     RMSE of |(u,v) - (u_g,v_g)| over 20 <= |lat| < 89.9, area weighted (their compute_geostrophic_imbalance)
  L_eff (km)         spherical-harmonic KE spectrum E(l) at 500 hPa (pyshtools SHExpandDH, Driscoll-Healy 720x1440,
                     lmax = 359); first l >= 10 where E_pred/E_ref < 0.5 for 5 consecutive l; L = 2 pi a / l
                     (their _find_effective_resolution); also the small-scale energy ratio, spectral divergence (W1) and
                     spectral residual (log RMSE) of their compute_spectral_scores
Excess values are model minus reference (ERA5 at the same valid time), as in their summary table. We add the same
quantities with the FP32 forecast as the reference, which is the quantisation-specific question.

  AQ_MODEL=pretrained python scripts/physmetrics_fields.py --recipes q0_sim t7_mx m01_q0_decoder_only m02_q0_no_decoder --split test --steps 2
Output: results*/_physmetrics_<split>_h<steps>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyshtools as pysh
import torch
from scipy.stats import wasserstein_distance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cards import harness as H          # noqa: E402
from cards.recipe import build          # noqa: E402
from scripts.diag_fields import from_pred, truth_full, field   # noqa: E402

R_DRY, R_V, EARTH_RADIUS, OMEGA = 287.05, 461.5, 6.371e6, 7.2921e-5
P_TOP, P_BOT, GEO_LEVEL, LAT_CUTOFF = 500.0, 850.0, 500.0, 20.0
K_MIN, THRESH, N_CONSEC = 10, 0.5, 5


class Grid:
    def __init__(self, lat: np.ndarray):
        self.lat = lat[:720].astype(np.float64)
        self.area = np.cos(np.deg2rad(self.lat)).clip(min=0.0)[:, None] * np.ones((1, 1440))   # cell area ∝ cos(lat)
        self.w_all = self.area / self.area.sum()
        self.f = 2.0 * OMEGA * np.sin(np.deg2rad(self.lat))
        self.cos = np.cos(np.deg2rad(self.lat))
        self.geo_mask = (np.abs(self.lat) >= LAT_CUTOFF) & (np.abs(self.lat) < 89.9)
        wg = np.where(self.geo_mask[:, None], self.area, 0.0)
        self.w_geo = wg / wg.sum()
        self.dlat = np.deg2rad(np.gradient(self.lat))                 # radians per row (negative when lat decreases)
        self.dlon = np.deg2rad(0.25)


def hydrostatic_rmse(F: dict, g: Grid) -> float:
    i_top, i_bot = H.LEVELS.index(int(P_TOP)), H.LEVELS.index(int(P_BOT))
    phi_top, phi_bot = F["atmos"]["z"][i_top].astype(np.float64), F["atmos"]["z"][i_bot].astype(np.float64)
    t_top, t_bot = F["atmos"]["t"][i_top].astype(np.float64), F["atmos"]["t"][i_bot].astype(np.float64)
    q_top, q_bot = F["atmos"]["q"][i_top].astype(np.float64), F["atmos"]["q"][i_bot].astype(np.float64)
    tv_top = t_top * (1.0 + (R_V / R_DRY - 1.0) * q_top)
    tv_bot = t_bot * (1.0 + (R_V / R_DRY - 1.0) * q_bot)
    err = (phi_top - phi_bot) - R_DRY * 0.5 * (tv_top + tv_bot) * np.log(P_BOT / P_TOP)
    return float(np.sqrt(np.sum(g.w_all * err ** 2)))


def geostrophic_rmse(F: dict, g: Grid) -> float:
    i = H.LEVELS.index(int(GEO_LEVEL))
    phi = F["atmos"]["z"][i].astype(np.float64)
    u, v = F["atmos"]["u"][i].astype(np.float64), F["atmos"]["v"][i].astype(np.float64)
    dphi_dlat = np.gradient(phi, axis=0) / g.dlat[:, None]                 # d/d(phi) in radians, like differentiate*(180/pi)
    dphi_dlon = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2.0 * g.dlon)
    f = np.where(g.f == 0.0, 1e-10, g.f)[:, None]
    cos = np.where(g.cos == 0.0, 1e-10, g.cos)[:, None]
    u_g = -dphi_dlat / (f * EARTH_RADIUS)
    v_g = dphi_dlon / (f * EARTH_RADIUS * cos)
    err2 = (u - u_g) ** 2 + (v - v_g) ** 2
    err2 = np.where(g.geo_mask[:, None], np.nan_to_num(err2), 0.0)
    return float(np.sqrt(np.sum(g.w_geo * err2)))


def ke_spectrum(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Their _ke_spectrum_spharm on a 720 x 1440 Driscoll-Healy grid (sampling 2, lmax 359)."""
    u = np.nan_to_num(np.asarray(u, dtype=np.float64)[:720]); v = np.nan_to_num(np.asarray(v, dtype=np.float64)[:720])
    lmax = 720 // 2 - 1
    uc = pysh.expand.SHExpandDH(u, sampling=2, lmax_calc=lmax)
    vc = pysh.expand.SHExpandDH(v, sampling=2, lmax_calc=lmax)
    pw = uc ** 2 + vc ** 2                                   # (2, l, m)
    e = 0.5 * (pw[0].sum(1) + pw[1].sum(1))                  # m = 0 of the sine part is zero
    return e


def effective_resolution(e_pred: np.ndarray, e_true: np.ndarray):
    k = np.arange(len(e_pred))
    mask = (k >= K_MIN) & (e_true > 1e-12)
    k_sel, ratio = k[mask], e_pred[mask] / e_true[mask]
    below = ratio < THRESH
    idx, run = None, 0
    for i in range(len(ratio)):
        if below[i]:
            run += 1
            if run >= N_CONSEC:
                idx = i - N_CONSEC + 1
                break
        else:
            run = 0
    if idx is None:
        return 2.0 * np.pi * EARTH_RADIUS / float(k_sel[-1]) / 1000.0, float(np.mean(ratio)), None
    k_c = float(k_sel[idx])
    return 2.0 * np.pi * EARTH_RADIUS / k_c / 1000.0, float(np.mean(ratio[idx:])), k_c


def spectral_scores(e_pred: np.ndarray, e_true: np.ndarray, eps: float = 1e-12):
    k = np.arange(len(e_pred))
    div = float(wasserstein_distance(u_values=k, v_values=k, u_weights=e_true, v_weights=e_pred))
    res = float(np.sqrt(np.mean((np.log(e_pred + eps) - np.log(e_true + eps)) ** 2)))
    return div, res


def metrics_of(F: dict, g: Grid) -> dict:
    i = H.LEVELS.index(500)
    return {"hydrostatic_rmse": hydrostatic_rmse(F, g), "geostrophic_rmse": geostrophic_rmse(F, g),
            "ke500": ke_spectrum(F["atmos"]["u"][i], F["atmos"]["v"][i])}


def relative(a: dict, b: dict) -> dict:
    L, ssr, kc = effective_resolution(a["ke500"], b["ke500"])
    div, res = spectral_scores(a["ke500"], b["ke500"])
    return {"L_eff_km": L, "small_scale_ratio": ssr, "k_cut": kc, "spectral_divergence": div, "spectral_residual": res,
            "hydrostatic_excess": a["hydrostatic_rmse"] - b["hydrostatic_rmse"],
            "geostrophic_excess": a["geostrophic_rmse"] - b["geostrophic_rmse"]}


def rollout_fields(model, date, steps, surface_fp32):
    preds, lat = H.rollout(model, date, steps, surface_fp32)
    out = [(np.datetime64(p.metadata.time[0], "s"), from_pred(p)) for p in preds]
    del preds
    H.free()
    return out, lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipes", nargs="+", required=True)
    ap.add_argument("--split", default="test", choices=list(H.SPLITS))
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--max-dates", type=int, default=None)
    ap.add_argument("--dates", nargs="*", default=None)
    ap.add_argument("--results", default=None)
    a = ap.parse_args()
    res = Path(a.results) if a.results else ROOT / ("results" if H.MODEL == "pretrained" else f"results_{H.MODEL}")
    dates = a.dates or H.SPLITS[a.split][: a.max_dates]
    out_json = res / f"_physmetrics_{a.split}_h{a.steps}.json"
    report = {"model": H.MODEL, "split": a.split, "steps": a.steps, "dates": dates, "definitions": "PhysMetrics.Weather (arXiv:2606.10642) hydrostatic 500-850 hPa with virtual temperature, geostrophic 500 hPa |lat|>=20, L_eff from spherical-harmonic KE(500) with threshold 0.5 over 5 consecutive l >= 10",
              "truth": {}, "fp32": {}, "recipes": {}}
    t0 = time.time()
    # Phase 1 (2026-10-20, shared GPU): FP32 and ERA5 metrics for every date with only the FP32 model resident.
    # The metrics are small (two scalars + a 360-float KE spectrum), so the FP32 model is freed before any card is built;
    # building a GPTQ/smoothed card next to a resident FP32 model ran out of memory on the shared 32 GB card.
    fp_model, _ = build(json.loads((ROOT / "recipes" / "fp32.json").read_text()))
    g = None
    M = {}
    for d in dates:
        fp_steps, lat = rollout_fields(fp_model, d, a.steps, False)
        g = g or Grid(lat.float().cpu().numpy())
        for s, (when, Ffp) in enumerate(fp_steps, 1):
            T = truth_full(when)
            mt, mf = metrics_of(T, g), metrics_of(Ffp, g)
            M[(s, d)] = (mt, mf)
            report["truth"].setdefault(str(s), {})[d] = {k: mt[k] for k in ("hydrostatic_rmse", "geostrophic_rmse")}
            report["fp32"].setdefault(str(s), {})[d] = {**{k: mf[k] for k in ("hydrostatic_rmse", "geostrophic_rmse")}, "vs_truth": relative(mf, mt),
                                                      "ke500": [float(x) for x in mf["ke500"]], "ke500_truth": [float(x) for x in mt["ke500"]]}
            del T
        del fp_steps
        print(f"[{H.now()}] fp32+truth {d} done ({time.time() - t0:.0f}s)", flush=True)
    del fp_model
    H.free()
    for ri, name in enumerate(a.recipes):
        rec = json.loads((ROOT / "recipes" / f"{name}.json").read_text(encoding="utf-8"))
        model, _ = build(rec)
        R = report["recipes"][name] = {"per_date": {}}
        for d in dates:
            steps, _ = rollout_fields(model, d, a.steps, bool(rec.get("surface_fp32", False)))
            for s, (when, Fc) in enumerate(steps, 1):
                mt, mf = M[(s, d)]
                mc = metrics_of(Fc, g)
                R["per_date"].setdefault(str(s), {})[d] = {**{k: mc[k] for k in ("hydrostatic_rmse", "geostrophic_rmse")},
                                                            "vs_truth": relative(mc, mt), "vs_fp32": relative(mc, mf),
                                                            "ke500": [float(x) for x in mc["ke500"]]}
            del steps
            print(f"[{H.now()}] {name} {d} done ({time.time() - t0:.0f}s)", flush=True)
        del model
        H.free()
        # per-step means
        R["per_step"] = {}
        for s, per in R["per_date"].items():
            ds_ = list(per)
            agg = {"hydrostatic_rmse": float(np.mean([per[d]["hydrostatic_rmse"] for d in ds_])),
                   "geostrophic_rmse": float(np.mean([per[d]["geostrophic_rmse"] for d in ds_])),
                   "truth": {k: float(np.mean([report["truth"][s][d][k] for d in ds_])) for k in ("hydrostatic_rmse", "geostrophic_rmse")},
                   "fp32": {k: float(np.mean([report["fp32"][s][d][k] for d in ds_])) for k in ("hydrostatic_rmse", "geostrophic_rmse")}}
            for rel in ("vs_truth", "vs_fp32"):
                agg[rel] = {k: float(np.mean([per[d][rel][k] for d in ds_])) for k in ("L_eff_km", "small_scale_ratio", "spectral_divergence", "spectral_residual", "hydrostatic_excess", "geostrophic_excess")}
            agg["fp32_vs_truth"] = {k: float(np.mean([report["fp32"][s][d]["vs_truth"][k] for d in ds_])) for k in ("L_eff_km", "small_scale_ratio", "spectral_divergence", "spectral_residual", "hydrostatic_excess", "geostrophic_excess")}
            R["per_step"][s] = agg
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"[{H.now()}] wrote {out_json.name}", flush=True)
    print(f"[{H.now()}] all done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
