"""Hurricane Beryl case (reviewer point 4, 2026-10-20): track and intensity of the ERA5 vortex in the summer week under
FP32, Q0 (q0_sim) and the deployable recipe (t7_mx), 20 steps (5 days) from two initial dates.
ERA5 truth (laptop, 2026-10-20): box 8-32N / 100-50W, min msl 990.0 hPa (07-08 12Z), max 10 m wind 27.6 m/s (07-03 12Z);
the vortex moves from 11N 57W (07-01 00Z) to 30N 95W (07-08 18Z). ERA5 underestimates Beryl's real intensity (934 hPa),
so the case tests whether quantisation changes the model's own ERA5-like vortex, not the real storm.
Per step and recipe: min msl and its position, max 10 m wind in the box, track error (km) vs the ERA5 vortex at the same
valid time, intensity errors; the box fields msl and ws are saved for a figure.

  AQ_MODEL=pretrained AQ_PERCEIVER_CHUNK=16384 python scripts/beryl_case.py [--recipes fp32 q0_sim t7_mx --dates 2024-07-01 2024-07-03 --steps 20]
Output: results/_beryl_case.json, results/_beryl_case_fields.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cards import harness as H                                  # noqa: E402
from cards.recipe import build                                  # noqa: E402
from scripts.diag_fields import from_pred, truth_full            # noqa: E402

BOX = {"lat": (8.0, 32.0), "lon": (260.0, 310.0)}   # 8-32N, 100W-50W
R_EARTH = 6371.0


def box_index(lat: np.ndarray):
    lon = np.arange(0, 360, 0.25)
    la = np.where((lat >= BOX["lat"][0]) & (lat <= BOX["lat"][1]))[0]
    lo = np.where((lon >= BOX["lon"][0]) & (lon <= BOX["lon"][1]))[0]
    return la, lo, lat[la], lon[lo]


def vortex(F: dict, la, lo, lat_b, lon_b) -> dict:
    msl = F["surf"]["msl"][np.ix_(la, lo)] / 100.0
    ws = np.hypot(F["surf"]["10u"], F["surf"]["10v"])[np.ix_(la, lo)]
    j, k = np.unravel_index(np.argmin(msl), msl.shape)
    return {"msl_min_hpa": float(msl.min()), "lat": float(lat_b[j]), "lon": float(lon_b[k]) - 360.0, "ws_max": float(ws.max()),
            "_msl": msl.astype(np.float32), "_ws": ws.astype(np.float32)}


def gc_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2, dl = np.radians(lat1), np.radians(lat2), np.radians(lon2 - lon1)
    return float(R_EARTH * np.arccos(np.clip(np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl), -1, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipes", nargs="+", default=["fp32", "q0_sim", "t7_mx"])
    ap.add_argument("--dates", nargs="+", default=["2024-07-01", "2024-07-03"])
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--results", default=str(ROOT / "results"))
    a = ap.parse_args()
    frac = float(os.environ.get("AQ_MEM_FRAC", "0"))
    if frac:
        torch.cuda.set_per_process_memory_fraction(frac)
    res = Path(a.results); res.mkdir(parents=True, exist_ok=True)
    report = {"model": H.MODEL, "box": BOX, "steps": a.steps, "dates": a.dates, "recipes": {}, "truth": {}}
    fields = {}
    t0 = time.time()
    la = lo = lat_b = lon_b = None
    for name in a.recipes:
        rec = json.loads((ROOT / "recipes" / f"{name}.json").read_text(encoding="utf-8"))
        model, _ = build(rec)
        R = report["recipes"][name] = {}
        for d in a.dates:
            preds, lat = H.rollout(model, d, a.steps, bool(rec.get("surface_fp32", False)))
            if la is None:
                la, lo, lat_b, lon_b = box_index(lat.float().cpu().numpy()[:720])
            rows = []
            for s, p in enumerate(preds, 1):
                when = np.datetime64(p.metadata.time[0], "s")
                v = vortex(from_pred(p), la, lo, lat_b, lon_b)
                key = str(when)
                if key not in report["truth"]:
                    T = truth_full(when)
                    vt = vortex(T, la, lo, lat_b, lon_b)
                    fields[f"truth__{key}__msl"] = vt.pop("_msl"); fields[f"truth__{key}__ws"] = vt.pop("_ws")
                    report["truth"][key] = vt
                    del T
                vt = report["truth"][key]
                fields[f"{name}__{d}__s{s}__msl"] = v.pop("_msl"); fields[f"{name}__{d}__s{s}__ws"] = v.pop("_ws")
                v.update(step=s, valid=key, track_err_km=gc_km(v["lat"], v["lon"], vt["lat"], vt["lon"]),
                         msl_err_hpa=v["msl_min_hpa"] - vt["msl_min_hpa"], ws_err=v["ws_max"] - vt["ws_max"])
                rows.append(v)
            R[d] = rows
            print(f"[{H.now()}] {name} {d}: step20 msl {rows[-1]['msl_min_hpa']:.1f} hPa (truth {report['truth'][rows[-1]['valid']]['msl_min_hpa']:.1f}), "
                  f"track err {rows[-1]['track_err_km']:.0f} km, ws max {rows[-1]['ws_max']:.1f} ({time.time() - t0:.0f}s)", flush=True)
            del preds
            H.free()
            (res / "_beryl_case.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
            np.savez_compressed(res / "_beryl_case_fields.npz", lat=lat_b, lon=lon_b - 360.0, **fields)
        del model
        H.free()
    # paired summary vs FP32: mean |track err| and intensity differences over steps 1..20, per date
    if "fp32" in report["recipes"]:
        summ = {}
        for name in a.recipes:
            if name == "fp32":
                continue
            diffs = []
            for d in a.dates:
                for r, f in zip(report["recipes"][name][d], report["recipes"]["fp32"][d]):
                    diffs.append({"step": r["step"], "track_vs_fp32_km": gc_km(r["lat"], r["lon"], f["lat"], f["lon"]),
                                  "msl_vs_fp32_hpa": r["msl_min_hpa"] - f["msl_min_hpa"], "ws_vs_fp32": r["ws_max"] - f["ws_max"],
                                  "track_err_km": r["track_err_km"], "fp32_track_err_km": f["track_err_km"]})
            summ[name] = {"mean_track_vs_fp32_km": float(np.mean([x["track_vs_fp32_km"] for x in diffs])),
                          "max_track_vs_fp32_km": float(np.max([x["track_vs_fp32_km"] for x in diffs])),
                          "mean_msl_vs_fp32_hpa": float(np.mean([x["msl_vs_fp32_hpa"] for x in diffs])),
                          "mean_ws_vs_fp32": float(np.mean([x["ws_vs_fp32"] for x in diffs])),
                          "mean_track_err_km": float(np.mean([x["track_err_km"] for x in diffs])),
                          "fp32_mean_track_err_km": float(np.mean([x["fp32_track_err_km"] for x in diffs])), "n": len(diffs)}
        report["summary_vs_fp32"] = summ
        (res / "_beryl_case.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(json.dumps(summ, indent=1), flush=True)


if __name__ == "__main__":
    main()
