"""Skill-hours lost: for a quantized card at lead time T, the extra lead time D such that FP32's ERA5 RMSE at T + D
equals the card's ERA5 RMSE at T (linear interpolation of FP32's RMSE-vs-lead curve, per variable, mean over dates).
D is in hours (6 h per step). Negative = the card is better than FP32 at that lead time.

Rule: RMSE is averaged over initial dates first, per variable and lead. Where FP32's curve decreases with lead (2t has
diurnal dips after 90 h), it is replaced by its running maximum, so the interpolation is well defined. Below the first
lead the curve is extrapolated linearly (D >= -T); above the last lead D is undefined and excluded from the variable
mean. D is a point estimate at the 6-h output resolution, not an uncertainty bound.

  python scripts/skill_hours.py --split test --steps 20 t7_mx c08_weight_only_gptq t7_g02_nodec_fold bf16
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VARS = ["2t", "10u", "10v", "msl", "z500", "t850", "u850", "v850", "q700"]


def curve(name, split, steps, results):
    by = defaultdict(list)
    for line in (results / f"{name}__{split}_h{steps}" / "records.jsonl").read_text().splitlines():
        r = json.loads(line)
        by[(r["var"], r["step"])].append(r["rmse_truth"])
    return {k: float(np.mean(v)) for k, v in by.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cards", nargs="+")
    ap.add_argument("--split", default="test")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--at", default="2,4,8,12,16,20", help="steps at which to report")
    a = ap.parse_args()
    res = Path(a.results)
    fp = curve("fp32", a.split, a.steps, res)
    at = [int(s) for s in a.at.split(",")]
    hours = np.arange(1, a.steps + 1) * 6.0
    print(f"skill-hours lost (h); columns = lead time; rows = variable; '>' = beyond the {6 * a.steps:.0f} h curve")
    for name in a.cards:
        c = curve(name, a.split, a.steps, res)
        print(f"== {name}")
        mean_d = defaultdict(list)
        for v in VARS:
            f = np.maximum.accumulate(np.array([fp[(v, s)] for s in range(1, a.steps + 1)]))   # monotone envelope
            row = []
            for s in at:
                target = c[(v, s)]
                if target <= f[0]:
                    d = (target - f[0]) / max(f[1] - f[0], 1e-12) * 6.0 + 6.0 - 6.0 * s   # extrapolate below the curve
                    d = max(d, -6.0 * s)
                elif target >= f[-1]:
                    row.append(">"); mean_d[s].append(np.nan); continue
                else:
                    t_eq = float(np.interp(target, f, hours))        # f is non-decreasing (envelope)
                    d = t_eq - 6.0 * s
                row.append(f"{d:+.1f}")
                mean_d[s].append(d)
            print(f"  {v:5s} " + " ".join(f"{x:>7s}" for x in row))
        print("  mean  " + " ".join(f"{np.nanmean(mean_d[s]):+7.1f}" for s in at))
        print("  lead  " + " ".join(f"{6 * s:6d}h" for s in at))


if __name__ == "__main__":
    main()
