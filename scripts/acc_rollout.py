"""Anomaly correlation coefficient (ACC) of 20-step rollouts against ERA5 (Table 9, Table S17c; descriptive).

For each recipe and initial date, the model is rolled out from ERA5 (00 and 06 UTC inputs, 06 UTC initialisation) and
z500, t850, 2t and msl are compared with ERA5 at every 6-h lead. Anomalies are taken against the ERA5 1991-2020
monthly climatology for the valid hour (scripts/clim_download.py). ACC is the uncentred, cos-latitude-weighted
correlation of forecast and ERA5 anomalies (as in WeatherBench 2), computed on the 1-degree points of the 0.25-degree
grid (every fourth point; latitudes 90..-89). Output rows: results/_acc_<split>_h<steps>/<recipe>.jsonl, resumable.

  python scripts/acc_rollout.py --recipes fp32 t7_a16 --split test --steps 20 --max-dates 25
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
from cards import harness as H  # noqa: E402
from cards.recipe import build  # noqa: E402

ACC_VARS = ("z500", "t850", "2t", "msl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipes", nargs="+", required=True)
    ap.add_argument("--split", default="test", choices=list(H.SPLITS))
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--max-dates", type=int, default=None)
    ap.add_argument("--clim", default=str(ROOT / "data_clim" / "era5_clim_1991_2020_1deg.npz"))
    a = ap.parse_args()
    clim = np.load(a.clim)
    frac = float(os.environ.get("AQ_MEM_FRAC", "0"))
    if frac:
        torch.cuda.set_per_process_memory_fraction(frac)
    out_dir = ROOT / "results" / f"_acc_{a.split}_h{a.steps}"
    out_dir.mkdir(parents=True, exist_ok=True)
    dates = H.SPLITS[a.split][: a.max_dates]
    truth = H.Truth()
    for name in a.recipes:
        path = out_dir / f"{name}.jsonl"
        done = set()
        if path.exists():
            done = {json.loads(l)["date"] for l in path.open(encoding="utf-8") if l.strip()}
        todo = [d for d in dates if d not in done]
        if not todo:
            print(f"skip {name}", flush=True)
            continue
        recipe = json.loads((ROOT / "recipes" / f"{name}.json").read_text(encoding="utf-8"))
        model, _ = build(recipe)
        with path.open("a", encoding="utf-8") as f:
            for d in todo:
                t0 = time.time()
                preds, lat = H.rollout(model, d, a.steps, False)
                w = np.cos(np.deg2rad(lat.float().cpu().numpy()[:720][::4]))[:, None]
                rows = []
                for s, p in enumerate(preds, 1):
                    when = np.datetime64(p.metadata.time[0], "s")
                    month = int(str(when)[5:7])
                    hour = int(str(when)[11:13])
                    fc = H.fields(p)
                    for var in ACC_VARS:
                        c = clim[f"{var}_m{month:02d}_h{hour:02d}"][:180]
                        fa = fc[var][::4, ::4] - c
                        ta = truth.get(when, var)[::4, ::4] - c
                        acc = float(np.sum(w * fa * ta) / np.sqrt(np.sum(w * fa * fa) * np.sum(w * ta * ta)))
                        rows.append({"date": d, "step": s, "var": var, "valid": str(when), "acc": acc})
                for r in rows:
                    f.write(json.dumps(r) + "\n")
                f.flush()
                print(f"[{H.now()}] {name} {d}: {time.time() - t0:.1f}s", flush=True)
                del preds
                H.free()
        del model
        H.free()


if __name__ == "__main__":
    main()
