"""Numbers quoted in the revised manuscript that come from existing per-record files (no new runs):
unrounded native-vs-simulated qerr (Sect. 4.5), the debiased excess of t7_a16 without the low-rank branch (Table 8)
and the FP32 reference skill on the 25 March rollout dates (Sect. 2.3, Table S7).
  python scripts/r2_numbers.py
"""
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = Path(os.environ.get("AQ_RESULTS", str(ROOT / "results")))


def recs(d):
    return [json.loads(l) for l in (RES / d / "records.jsonl").open(encoding="utf-8") if l.strip()]


def summ(d):
    return json.load(open(RES / d / "summary.json"))


# 1. unrounded qerr / excess, simulated vs native
for sim, nat in (("t7_a16__test_h2", "native_t7_t7_a16_w4a16__test_h2"),
                 ("w16_t7_gptq_sq__test_h2", "native_t7_w16_t7_gptq_sq_w4a16_exact_bf16rest__test_h2")):
    for d in (sim, nat):
        p = RES / d / "summary.json"
        if p.exists():
            s = summ(d)
            print("%-60s qerr %.4f excess %.4f" % (d, s["qerr_pct"], s["excess_pct"]))
for d in sorted(x.name for x in RES.glob("native_t7_t7_a16*__test_h2")):
    s = summ(d)
    print("%-60s qerr %.4f excess %.4f" % (d, s["qerr_pct"], s["excess_pct"]))

# 2. debiased excess (global-mean error removed) of w16_t7_gptq_sq and t7_a16, mean over dates and var-steps
f = {(r["date"], r["var"], r["step"]): r for r in recs("fp32__test_h2")}
for card in ("t7_a16", "w16_t7_gptq_sq"):
    vals = []
    for r in recs(f"{card}__test_h2"):
        b = f[(r["date"], r["var"], r["step"])]
        q = np.sqrt(max(r["rmse_truth"] ** 2 - r["bias_truth"] ** 2, 0))
        p = np.sqrt(max(b["rmse_truth"] ** 2 - b["bias_truth"] ** 2, 0))
        vals.append(q / p - 1)
    print(card, "debiased excess %.3f %%" % (100 * np.mean(vals)))

# 3. FP32 reference skill against ERA5 over the 25 March rollout dates
rr = [r for r in recs("fp32__test_h20") if r["date"] <= "2024-03-25"]
print("fp32 h20 dates:", len({r["date"] for r in rr}))
for var in ("z500", "t850", "2t", "10u", "msl"):
    row = []
    for step in (2, 4, 12, 20):
        v = [r["rmse_truth"] ** 2 for r in rr if r["var"] == var and r["step"] == step]
        row.append(np.sqrt(np.mean(v)))
    print(var, " ".join("%.3f" % x for x in row))
