"""Summaries of the revision analyses (Table 5 lower block and S17b; Table 9 and S17c) from the per-record files.

    python scripts/r1_summary.py > results/_r1_summary.txt      (also writes results/_r1_summary.json)

Excess (per cent) for a lead: mean over the nine variables of RMSE(card vs ERA5) / RMSE(FP32 vs ERA5) - 1 per date, then
mean over dates; qerr: mean over variables of RMSE(card vs FP32) / RMSE(FP32 vs ERA5). Intervals: 3-day non-circular
moving-block bootstrap over initial dates (2000 resamples, seed 0), paired by date, percentile 95 %.
"""
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = Path(os.environ.get("AQ_RESULTS", str(ROOT / "results")))
RNG = np.random.default_rng(0)


def records(card, split, steps):
    p = RES / f"{card}__{split}_h{steps}" / "records.jsonl"
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()] if p.exists() else []


def block_idx(n, block=3, reps=2000):
    starts = np.arange(n - block + 1)
    k = int(np.ceil(n / block))
    out = []
    for _ in range(reps):
        s = RNG.choice(starts, size=k)
        idx = np.concatenate([np.arange(a, a + block) for a in s])[:n]
        out.append(idx)
    return np.array(out)


def ci(per_date, idx):
    m = per_date[idx].mean(1)
    return float(per_date.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def per_date_metric(card, split, steps, step, dates, metric):
    rc = {(r["date"], r["var"]): r for r in records(card, split, steps) if r["step"] == step}
    rf = {(r["date"], r["var"]): r for r in records("fp32", split, steps) if r["step"] == step}
    vals = []
    for d in dates:
        v = []
        for (dd, var), r in rc.items():
            if dd != d or (dd, var) not in rf:
                continue
            if metric == "excess":
                v.append(r["rmse_truth"] / rf[(dd, var)]["rmse_truth"] - 1)
            else:
                v.append(r["rmse_ref"] / rf[(dd, var)]["rmse_truth"])
        vals.append(100 * np.mean(v))
    return np.array(vals)


def common_dates(cards, split, steps):
    sets = [set(r["date"] for r in records(c, split, steps)) for c in cards + ["fp32"]]
    return sorted(set.intersection(*sets))


out = {}
# ---------------- E3: rollouts
for split, cards in (("test", ["t7_a16", "w16_t7_gptq_sq", "u8_w8a16_mx_prot", "c08_t7set"]),
                     ("oct", ["t7_a16", "w16_t7_gptq_sq"])):
    dates = common_dates(cards, split, 20)
    if not dates:
        continue
    idx = block_idx(len(dates))
    print(f"== rollouts {split}: {len(dates)} common dates {dates[0]}..{dates[-1]}")
    for c in cards:
        row = {}
        for step in (2, 4, 12, 20):
            e = per_date_metric(c, split, 20, step, dates, "excess")
            q = per_date_metric(c, split, 20, step, dates, "qerr")
            row[f"{6 * step}h"] = {"excess": ci(e, idx), "qerr": ci(q, idx)}
        out.setdefault("rollouts", {}).setdefault(split, {})[c] = row
        print(c, {k: "exc %+.2f [%+.2f, %+.2f] qerr %.1f" % (*v["excess"], v["qerr"][0]) for k, v in row.items()})

# ---------------- E3: ACC
for split in ("test", "oct"):
    d = RES / f"_acc_{split}_h20"
    if not d.exists():
        continue
    rows = {}
    for f in d.glob("*.jsonl"):
        rows[f.stem] = [json.loads(l) for l in f.open(encoding="utf-8") if l.strip()]
    if "fp32" not in rows:
        continue
    dates = sorted(set.intersection(*[set(r["date"] for r in v) for v in rows.values()]))
    idx = block_idx(len(dates))
    print(f"== ACC {split}: {len(dates)} dates; recipes {sorted(rows)}")

    def arr(name, var, step):
        m = {(r["date"]): r["acc"] for r in rows[name] if r["var"] == var and r["step"] == step}
        return np.array([m[x] for x in dates])

    for name in sorted(rows):
        res = {}
        for var in ("z500", "t850", "2t", "msl"):
            for step in (4, 12, 20):
                a = arr(name, var, step)
                entry = {"acc": float(a.mean())}
                if name != "fp32":
                    diff = a - arr("fp32", var, step)
                    entry["dacc"] = ci(diff, idx)
                res[f"{var}_{6 * step}h"] = entry
        out.setdefault("acc", {}).setdefault(split, {})[name] = res
        print(name, {k: ("%.4f" % v["acc"]) + ("" if "dacc" not in v else " d%+.4f [%+.4f, %+.4f]" % v["dacc"])
                     for k, v in res.items() if k.endswith("120h") or k.endswith("72h")})

# ---------------- E2: groups (paired, test, 2 steps)
dates =[f"2024-03-{d:02d}" for d in range(1, 31)]
idx = block_idx(len(dates))


def h2(card, metric):
    rc = [r for r in records(card, "test", 2)]
    rf = {(r["date"], r["var"], r["step"]): r for r in records("fp32", "test", 2)}
    vals = []
    for d in dates:
        v = [(r["rmse_truth"] / rf[(d, r["var"], r["step"])]["rmse_truth"] - 1) if metric == "excess"
             else r["rmse_ref"] / rf[(d, r["var"], r["step"])]["rmse_truth"] for r in rc if r["date"] == d]
        vals.append(100 * np.mean(v) if v else np.nan)
    return np.array(vals)


base = {m: h2("t7_a16", m) for m in ("qerr", "excess")}
print("== E2 groups vs t7_a16 (test, 2 steps)")
for card in ["t7a16_all314_noenc"] + [f"w16_t7_inh_{g}" for g in ("surf_mlp", "patch", "enc0", "enc1", "enc2", "dec0", "dec1", "dec2")]:
    if not records(card, "test", 2):
        print(card, "missing")
        continue
    r = {}
    for m in ("qerr", "excess"):
        a = h2(card, m)
        r[m] = ci(a, idx)
        r["d" + m] = ci(a - base[m], idx)
    out.setdefault("groups", {})[card] = r
    print("%-22s qerr %.2f exc %+.2f | dq %+.2f [%+.2f, %+.2f] dexc %+.3f [%+.3f, %+.3f]" % (
        card, r["qerr"][0], r["excess"][0], *r["dqerr"], *r["dexcess"]))

(RES / "_r1_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
