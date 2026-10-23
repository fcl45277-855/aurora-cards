"""Candidate 13: meteorological fingerprints of quantization error, computed online (no fields are stored).

The FP32 model (recipes/fp32.json) and one card model at a time share the GPU; per date both are rolled out and per
step the card is compared with FP32 and with ERA5 (FP32 vs ERA5 is the yardstick):
  spectra      zonal power spectra (rFFT along longitude, 20-70 deg both hemispheres, cos-lat weighted, log2 wavenumber
               bins) of msl, 2t, z500, t850, u850 for truth / fp32 / card and of the errors fp32-truth, card-fp32;
               small-scale share of the error power (k > 50) says noise-like vs large-scale. This scalar cut (k > 50)
               differs from the binned small-scale ratio of scripts/extended_analyses.py section D (bins with k >= 64);
               both are reported as defined
  extremes     10 m wind q99 / q99.9 / max, msl min, 2t q0.1 / q99.9 (|lat| <= 80); RMSE of card-fp32 restricted to the
               truth-wind top 1 % cells over the global RMSE ("does the error concentrate in extremes")
  vertical     hydrostatic residual RMS over the 12 layers (dz - R T ln(p1/p2)), superadiabatic-layer share (p >= 300),
               vertical correlation of adjacent-level errors (t and z) for card-fp32 and fp32-truth
  geostrophic  500 hPa ageostrophic residual RMS (25-70 deg); the zonal gradient is one-sided at the first and last
               longitude (numpy.gradient), as for the paper's values; --wrap-lon uses periodic central differences
               there, as scripts/physmetrics_fields.py does
  geography    2-deg maps of squared error (card-fp32, fp32-truth) for z500 / 2t / msl, zonal-mean squared error,
               spatial correlation of the two maps (flow dependence proxy)
  mass         lat-weighted global mean msl

  AQ_MODEL=pretrained python scripts/diag_fields.py --recipes q0_sim t7_mx c08_weight_only_gptq --split dev --steps 2 [--max-dates 7]
  AQ_MODEL=pretrained python scripts/diag_fields.py --recipes q0_sim t7_mx --split test --steps 20 --dates 2024-03-05 2024-03-15

Outputs results*/_diag_fields_<split>_h<steps>.json (scalars per recipe/step, mean over dates and per date) and
results*/_diag_fields_<split>_h<steps>_maps.npz (2-deg maps and zonal profiles).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cards import harness as H          # noqa: E402
from cards.recipe import build          # noqa: E402

R_D = 287.05
G0 = 9.80665
OMEGA = 7.2921e-5
M_PER_DEG = 111195.0
SPEC_FIELDS = ("msl", "2t", "z500", "t850", "u850")
MAP_FIELDS = ("z500", "2t", "msl")
K_EDGES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 721)       # wavenumber bins [edge_i, edge_{i+1})
K_SMALL = 50                                                 # "small scale" = wavenumber > 50 (~560 km at 45 deg)
SURF_PRED = {"2t": "2t", "10u": "10u", "10v": "10v", "msl": "msl"}


# ---- fields ----------------------------------------------------------------------------------------

def from_pred(pred) -> dict:
    return {"surf": {k: pred.surf_vars[k][0, 0].float().cpu().numpy() for k in SURF_PRED},
            "atmos": {v: pred.atmos_vars[v][0, 0].float().cpu().numpy() for v in ("z", "t", "u", "v", "q")}}


def truth_full(when: np.datetime64) -> dict:
    day = str(when.astype("datetime64[D]"))
    hour = int((when - when.astype("datetime64[D]")) / np.timedelta64(1, "h"))
    i = H.hour_index(day, hour)
    surf = {k: np.asarray(H._npz_member(H.DATA / f"surface_{day}.npz", H.SURF[k])[i], np.float32)[:720] for k in H.SURF}
    atmos = {}
    for v in ("z", "t", "u", "v", "q"):
        arr = H._npz_member(H.DATA / f"air_{day}.npz", v)
        atmos[v] = np.stack([np.asarray(arr[i, H.level_index(day, lev)], np.float32)[:720] for lev in H.LEVELS])
    return {"surf": surf, "atmos": atmos}


def field(F: dict, name: str) -> np.ndarray:
    if name in F["surf"]:
        return F["surf"][name]
    v, lev = name[0], int(name[1:])
    return F["atmos"][v][H.LEVELS.index(lev)]


# ---- diagnostics -----------------------------------------------------------------------------------

class Geo:
    wrap_lon = False                 # periodic zonal gradient in geostrophic_residual (--wrap-lon)

    def __init__(self, lat: np.ndarray):
        self.lat = lat[:720].astype(np.float64)
        self.w = np.cos(np.deg2rad(self.lat)).clip(min=0.0)
        self.w2 = (self.w / self.w.mean())[:, None]                               # lat weights, mean 1
        self.mid = (np.abs(self.lat) >= 20) & (np.abs(self.lat) <= 70)
        self.geo = (np.abs(self.lat) >= 25) & (np.abs(self.lat) <= 70)
        self.ext = np.abs(self.lat) <= 80
        self.dlat = float(self.lat[1] - self.lat[0])                               # negative if lat runs 90 -> -90
        self.f = 2 * OMEGA * np.sin(np.deg2rad(self.lat))

    def wmean(self, a: np.ndarray) -> float:
        return float(np.mean(self.w2 * a))

    def wrmse(self, a: np.ndarray, mask: np.ndarray | None = None) -> float:
        if mask is None:
            return float(np.sqrt(self.wmean(a * a)))
        w = np.broadcast_to(self.w2, a.shape)[mask]
        return float(np.sqrt(np.sum(w * a[mask] ** 2) / np.sum(w)))

    def spectrum(self, a: np.ndarray) -> np.ndarray:
        """cos-lat weighted mean zonal power per wavenumber bin over the midlatitude bands (both hemispheres)."""
        rows = a[self.mid].astype(np.float64)
        p = np.abs(np.fft.rfft(rows, axis=1)) ** 2 / rows.shape[1] ** 2                # (rows, 721)
        w = self.w[self.mid][:, None]
        pw = (p * w).sum(0) / w.sum()
        return np.array([pw[K_EDGES[i]:K_EDGES[i + 1]].sum() for i in range(len(K_EDGES) - 1)])

    def small_share(self, a: np.ndarray) -> float:
        rows = a[self.mid].astype(np.float64)
        p = np.abs(np.fft.rfft(rows, axis=1)) ** 2
        w = self.w[self.mid][:, None]
        pw = (p * w).sum(0) / w.sum()
        return float(pw[K_SMALL + 1:].sum() / max(pw[1:].sum(), 1e-30))

    def geostrophic_residual(self, F: dict) -> float:
        z = field(F, "z500").astype(np.float64)
        u, v = field(F, "u500").astype(np.float64), field(F, "v500").astype(np.float64)
        dy = self.dlat * M_PER_DEG
        dx = 0.25 * M_PER_DEG * np.cos(np.deg2rad(self.lat))[:, None]
        dz_dy = np.gradient(z, axis=0) / dy
        if self.wrap_lon:            # central differences everywhere (longitude is periodic)
            dz_dx = (np.roll(z, -1, axis=1) - np.roll(z, 1, axis=1)) / (2.0 * dx)
        else:                        # one-sided at the two edge columns (paper values)
            dz_dx = np.gradient(z, axis=1) / dx
        f = np.where(self.geo, self.f, np.nan)[:, None]                        # rows outside 25-70 deg are not used
        with np.errstate(divide="ignore", invalid="ignore"):
            ug, vg = -dz_dy / f, dz_dx / f
        m = self.geo
        return float(np.sqrt(np.mean((self.w2 * ((u - ug) ** 2 + (v - vg) ** 2))[m]) / np.mean(self.w2[m])))

    def coarse(self, a: np.ndarray) -> np.ndarray:
        return a.reshape(90, 8, 180, 8).mean((1, 3))


def absolute(F: dict, g: Geo) -> dict:
    """Quantities of one state (truth, fp32 or card)."""
    out = {"spectra": {k: g.spectrum(field(F, k)).tolist() for k in SPEC_FIELDS}}
    ws = np.hypot(F["surf"]["10u"], F["surf"]["10v"])
    m = g.ext
    out["ws_q99"], out["ws_q999"], out["ws_max"] = [float(x) for x in np.quantile(ws[m], (0.99, 0.999, 1.0))]
    t2 = F["surf"]["2t"]
    out["t2m_q001"], out["t2m_q999"] = [float(x) for x in np.quantile(t2[m], (0.001, 0.999))]
    out["msl_min"] = float(F["surf"]["msl"].min())
    out["msl_global_mean"] = g.wmean(F["surf"]["msl"])
    T, Z = F["atmos"]["t"].astype(np.float64), F["atmos"]["z"].astype(np.float64)
    lv = np.array(H.LEVELS, dtype=np.float64)
    res2, n = 0.0, 0
    sup, nsup = 0, 0
    for i in range(len(lv) - 1):                              # i: lower pressure (higher), i+1: higher pressure
        dz = Z[i] - Z[i + 1]
        est = R_D * 0.5 * (T[i] + T[i + 1]) * np.log(lv[i + 1] / lv[i])
        res2 += g.wmean((dz - est) ** 2); n += 1
        if lv[i] >= 300:
            lapse = -(T[i] - T[i + 1]) / np.maximum(dz / G0, 1.0)         # K/m, positive when T falls with height
            sup += float(np.mean(g.w2 * (lapse > 0.0098))); nsup += 1
    out["hydrostatic_residual_rms"] = float(np.sqrt(res2 / n))
    out["superadiabatic_share"] = sup / max(nsup, 1)
    out["geostrophic_residual_rms"] = g.geostrophic_residual(F)
    return out


def pair(A: dict, B: dict, g: Geo, truth: dict) -> dict:
    """Error quantities of A relative to B (card-fp32 or fp32-truth); conditional statistics use the truth wind."""
    out = {"err_spectra": {}, "err_small_share": {}, "rmse": {}}
    for k in SPEC_FIELDS:
        e = field(A, k) - field(B, k)
        out["err_spectra"][k] = g.spectrum(e).tolist()
        out["err_small_share"][k] = g.small_share(e)
        out["rmse"][k] = g.wrmse(e)
    ws_t = np.hypot(truth["surf"]["10u"], truth["surf"]["10v"])
    top = ws_t >= np.quantile(ws_t[g.ext], 0.99)
    e_ws = np.hypot(A["surf"]["10u"], A["surf"]["10v"]) - np.hypot(B["surf"]["10u"], B["surf"]["10v"])
    out["ws_rmse_all"], out["ws_rmse_top1pct"] = g.wrmse(e_ws), g.wrmse(e_ws, top)
    out["ws_top1pct_ratio"] = out["ws_rmse_top1pct"] / max(out["ws_rmse_all"], 1e-12)
    corr = {}
    for v in ("t", "z"):
        e = (A["atmos"][v] - B["atmos"][v]).astype(np.float64)
        cs = []
        for i in range(e.shape[0] - 1):
            a, b = e[i] - g.wmean(e[i]), e[i + 1] - g.wmean(e[i + 1])
            cs.append(g.wmean(a * b) / np.sqrt(max(g.wmean(a * a) * g.wmean(b * b), 1e-30)))
        corr[v] = float(np.mean(cs))
        out[f"rmse_{v}_by_level"] = [g.wrmse(e[i]) for i in range(e.shape[0])]
    out["vertical_err_corr"] = corr
    out["maps"] = {k: g.coarse((field(A, k) - field(B, k)) ** 2) for k in MAP_FIELDS}
    out["zonal"] = {k: (g.w2[:, 0] * 0 + np.mean((field(A, k) - field(B, k)) ** 2, axis=1)).tolist() for k in MAP_FIELDS}
    return out


def map_corr(m1: np.ndarray, m2: np.ndarray) -> float:
    a, b = np.log(m1 + 1e-30).ravel(), np.log(m2 + 1e-30).ravel()
    return float(np.corrcoef(a, b)[0, 1])


# ---- driver ----------------------------------------------------------------------------------------

def rollout_fields(model, date: str, steps: int, surface_fp32: bool):
    preds, lat = H.rollout(model, date, steps, surface_fp32)
    out = [(np.datetime64(p.metadata.time[0], "s"), from_pred(p)) for p in preds]
    del preds
    H.free()
    return out, lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipes", nargs="+", required=True)
    ap.add_argument("--split", default="dev", choices=list(H.SPLITS))
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--max-dates", type=int, default=None)
    ap.add_argument("--dates", nargs="*", default=None)
    ap.add_argument("--results", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--wrap-lon", action="store_true",
                    help="periodic zonal gradient for the geostrophic residual (default off: one-sided edges, as in the paper)")
    a = ap.parse_args()
    Geo.wrap_lon = a.wrap_lon
    res = Path(a.results) if a.results else ROOT / ("results" if H.MODEL == "pretrained" else f"results_{H.MODEL}")
    res.mkdir(parents=True, exist_ok=True)
    dates = a.dates or H.SPLITS[a.split][: a.max_dates]
    out_json = res / f"_diag_fields_{a.split}_h{a.steps}{a.tag}.json"
    out_maps = res / f"_diag_fields_{a.split}_h{a.steps}{a.tag}_maps.npz"
    report = {"model": H.MODEL, "split": a.split, "steps": a.steps, "dates": dates, "k_edges": K_EDGES, "k_small": K_SMALL,
              "spec_fields": SPEC_FIELDS, "truth": {}, "fp32": {}, "recipes": {}}
    maps = {}

    # The FP32 model stays on the GPU; each card model is built next to it and both are rolled out date by date, so
    # only one date's fields (fp32 + card, ~0.6 GB per step) live in RAM at a time. The first version kept all dates'
    # FP32 fields and truths in RAM and was OOM-killed on the 20-step run (container limit 90 GiB, 45 GiB of it /dev/shm).
    t0 = time.time()
    fp_model, _ = build(json.loads((ROOT / "recipes" / "fp32.json").read_text()))
    g = None
    for ri, name in enumerate(a.recipes):
        rec = json.loads((ROOT / "recipes" / f"{name}.json").read_text(encoding="utf-8"))
        t1 = time.time()
        # 2026-10-20 (shared GPU): building a GPTQ/smoothed card next to the resident FP32 model ran out of memory
        # in the calibration pass (t7_mx in the inits7to12 and y2023 runs); free FP32, build the card, reload FP32 (~20 s).
        del fp_model
        H.free()
        model, info = build(rec)
        fp_model, _ = build(json.loads((ROOT / "recipes" / "fp32.json").read_text()))
        print(f"[{H.now()}] built {name} in {time.time() - t1:.0f}s", flush=True)
        R = report["recipes"][name] = {"per_step": {}, "per_date": {}}
        for d in dates:
            fp_steps, lat = rollout_fields(fp_model, d, a.steps, False)
            g = g or Geo(lat.float().cpu().numpy())
            steps, _ = rollout_fields(model, d, a.steps, bool(rec.get("surface_fp32", False)))
            for s, (when, Fc) in enumerate(steps, 1):
                Ffp = fp_steps[s - 1][1]
                T = truth_full(when)
                if ri == 0:                                     # truth and fp32 statistics once, with the first recipe
                    report["truth"].setdefault(str(s), {})[d] = absolute(T, g)
                    pr = pair(Ffp, T, g, T)
                    maps[f"fp32_vs_truth/s{s}/{d}"] = pr.pop("maps")
                    report["fp32"].setdefault(str(s), {})[d] = {**absolute(Ffp, g), "vs_truth": pr}
                pf = pair(Fc, Ffp, g, T)
                pt = pair(Fc, T, g, T)
                m_q, m_f = pf.pop("maps"), pt.pop("maps")
                maps[f"{name}_vs_fp32/s{s}/{d}"] = m_q
                maps[f"{name}_vs_truth/s{s}/{d}"] = m_f
                fp_t = maps[f"fp32_vs_truth/s{s}/{d}"]
                R["per_date"].setdefault(str(s), {})[d] = {
                    **absolute(Fc, g), "vs_fp32": pf, "vs_truth": pt,
                    "map_corr_qerr_vs_fp32err": {k: map_corr(m_q[k], fp_t[k]) for k in MAP_FIELDS}}
                del T
            del fp_steps, steps
            print(f"[{H.now()}] {name} {d} done ({time.time() - t0:.0f}s)", flush=True)
        del model
        H.free()
        # means over dates for the headline scalars
        for s, per in R["per_date"].items():
            keys = ["ws_q99", "ws_q999", "ws_max", "t2m_q001", "t2m_q999", "msl_min", "msl_global_mean",
                    "hydrostatic_residual_rms", "superadiabatic_share", "geostrophic_residual_rms"]
            agg = {k: float(np.mean([per[d][k] for d in per])) for k in keys}
            agg["fp32"] = {k: float(np.mean([report["fp32"][s][d][k] for d in per])) for k in keys}
            agg["truth"] = {k: float(np.mean([report["truth"][s][d][k] for d in per])) for k in keys}
            for rel in ("vs_fp32", "vs_truth"):
                agg[rel] = {"rmse": {k: float(np.mean([per[d][rel]["rmse"][k] for d in per])) for k in SPEC_FIELDS},
                            "err_small_share": {k: float(np.mean([per[d][rel]["err_small_share"][k] for d in per])) for k in SPEC_FIELDS},
                            "ws_top1pct_ratio": float(np.mean([per[d][rel]["ws_top1pct_ratio"] for d in per])),
                            "vertical_err_corr": {v: float(np.mean([per[d][rel]["vertical_err_corr"][v] for d in per])) for v in ("t", "z")},
                            "rmse_t_by_level": np.mean([per[d][rel]["rmse_t_by_level"] for d in per], 0).tolist(),
                            "rmse_z_by_level": np.mean([per[d][rel]["rmse_z_by_level"] for d in per], 0).tolist()}
            agg["fp32_vs_truth"] = {
                "err_small_share": {k: float(np.mean([report["fp32"][s][d]["vs_truth"]["err_small_share"][k] for d in per])) for k in SPEC_FIELDS},
                "ws_top1pct_ratio": float(np.mean([report["fp32"][s][d]["vs_truth"]["ws_top1pct_ratio"] for d in per])),
                "vertical_err_corr": {v: float(np.mean([report["fp32"][s][d]["vs_truth"]["vertical_err_corr"][v] for d in per])) for v in ("t", "z")}}
            agg["map_corr_qerr_vs_fp32err"] = {k: float(np.mean([per[d]["map_corr_qerr_vs_fp32err"][k] for d in per])) for k in MAP_FIELDS}
            R["per_step"][s] = agg
        out_json.write_text(json.dumps(report, indent=1), encoding="utf-8")
        np.savez_compressed(out_maps, **{k.replace("/", "__") + "__" + v: m[v] for k, m in maps.items() for v in m})
        print(f"[{H.now()}] wrote {out_json.name}", flush=True)
    print(f"[{H.now()}] all done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
