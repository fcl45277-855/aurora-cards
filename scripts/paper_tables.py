"""Reproduce the headline numbers of the paper's tables from the released result summaries (no GPU, no data).

    python scripts/paper_tables.py [--results results_summary]

Prints the recipe path (Table 2, development split), the test-month evaluation (Table 3), the seasonal and
cross-year checks (Sect. 4.5) and the cross-model localisation ratios (Table S10), each next to the value printed in
the paper, and exits with status 1 if any value differs by more than the rounding of the paper. Re-running the
experiments (docs/REPRODUCE.md) writes new summaries into results/; use --results results to compare a re-run.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (card, split, paper qerr %, decimals, paper ERA5 excess % or None, decimals)
TABLE2 = [("q0_sim", "dev", 147.5, 1, 84.8, 1), ("c04_no_rot", "dev", 164.0, 1, 101.3, 1), ("c11_norot_gptq", "dev", 156.9, 1, 95.2, 1),
          ("c09_smooth05_gptq", "dev", 103.8, 1, 50.6, 1), ("g01_svdq_a05_r32", "dev", 32.8, 1, 6.8, 1),
          ("g02_svdq_a08_r32", "dev", 33.7, 1, 7.3, 1), ("g03_svdq_a08_r64", "dev", 28.6, 1, 5.9, 1),
          ("t7_g02_nodec_fold", "dev", 17.3, 1, 1.3, 1), ("t7_mx", "dev", 10.5, 1, 0.65, 2),
          ("c08_weight_only_gptq", "dev", 14.1, 1, 0.9, 1), ("g02_mx", "dev", 16.2, 1, 1.8, 1)]
TABLE3 = [("bf16", "test", 1.75, 2, 0.00, 2), ("q0_sim", "test", 147.24, 2, 84.50, 2), ("m01_q0_decoder_only", "test", 129.03, 2, 75.89, 2),
          ("m02_q0_no_decoder", "test", 46.36, 2, 10.53, 2), ("c08_weight_only_gptq", "test", 14.20, 2, 1.02, 2),
          ("g02_svdq_a08_r32", "test", 33.62, 2, 7.16, 2), ("t7_g02_nodec_fold", "test", 17.32, 2, 1.27, 2),
          ("t7_mx", "test", 10.56, 2, 0.61, 2)]
ROBUST = [("t7_mx_cal2", "test", 10.40, 2, None, 2), ("t7_mx_calsum", "test", 10.60, 2, 0.60, 2),
          ("t7_mx", "summer", 10.85, 2, 0.67, 2), ("q0_sim", "summer", 144.7, 1, None, 1),
          ("t7_mx", "y2023", 11.00, 2, 0.51, 2), ("c08_weight_only_gptq", "y2023", 14.89, 2, 1.11, 2),
          ("q0_sim", "y2023", 156.4, 1, 92.3, 1), ("bf16", "y2023", 1.81, 2, None, 2)]
CROSS = {"pretrained": ("m01_q0_decoder_only", 0.52, 0.54), "finetuned": ("x_dec_only", 0.42, 0.68), "small": ("x_dec_only", 0.70, 0.35)}


def load(res: Path, model: str, card: str, split: str, steps: int = 2):
    for base in (res / model, res):
        p = base / f"{card}__{split}_h{steps}" / "summary.json"
        if p.exists():
            return json.load(open(p, encoding="utf-8"))
    return None


def close(value, paper, dec):
    return value is not None and abs(round(value, dec) - paper) <= 0.5 * 10 ** -dec + 1e-9


def table(title, rows, res, bad):
    print(f"\n{title}\n{'card':26s} {'split':7s} {'qerr %':>9s} {'paper':>8s} {'excess %':>9s} {'paper':>8s}  95% CI of excess")
    for card, split, pq, dq, pe, de in rows:
        s = load(res, "pretrained", card, split)
        if s is None:
            print(f"{card:26s} {split:7s}   (not found)"); bad.append(f"{card}/{split}"); continue
        q, e, ci = s.get("qerr_pct"), s.get("excess_pct"), s.get("excess_pct_ci95")
        ok = close(q, pq, dq) and (pe is None or close(e, pe, de))
        if not ok:
            bad.append(f"{card}/{split}")
        e_s = f"{e:+9.{de}f}" if e is not None else " " * 9
        pe_s = f"{pe:+8.{de}f}" if pe is not None else " " * 8
        ci_s = f"[{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci else ""
        print(f"{card:26s} {split:7s} {q:9.{dq}f} {pq:8.{dq}f} {e_s} {pe_s}  {ci_s}{'' if ok else '   <-- differs'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results_summary"))
    a = ap.parse_args()
    res = Path(a.results)
    bad = []
    table("Table 2: path to the deployable recipe (development split, 12 dates x 2 steps)", TABLE2, res, bad)
    table("Table 3: test month (2024-03-01..30, 30 dates x 2 steps)", TABLE3, res, bad)
    table("Sect. 4.5: calibration week, season, calibration season and year", ROBUST, res, bad)
    print("\nTable S10, criterion 1 (dev): qerr(one resampler at W4A4) / qerr(all 314 layers at W4A4)")
    for model, (dec_card, pd, pe) in CROSS.items():
        f, d, e = (load(res, model, c, "dev") for c in ("x_all_q0num", dec_card, "m1enc"))
        if not (f and d and e):
            print(f"  {model:10s} (summaries not found)"); bad.append(f"cross/{model}"); continue
        rd, re_ = d["qerr_pct"] / f["qerr_pct"], e["qerr_pct"] / f["qerr_pct"]
        ok = close(rd, pd, 2) and close(re_, pe, 2)
        if not ok:
            bad.append(f"cross/{model}")
        print(f"  {model:10s} decoder {rd:.2f} (paper {pd:.2f})   encoder {re_:.2f} (paper {pe:.2f}){'' if ok else '   <-- differs'}")
    print("\nall values match the paper" if not bad else f"\nmismatches or missing: {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
