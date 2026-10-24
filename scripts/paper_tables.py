"""Reproduce the headline numbers of the paper's tables from the released result summaries (no GPU, no data).

    python scripts/paper_tables.py [--results results_summary]

Table and section numbers follow manuscript version v17. Prints the recipe path (Table 3, development split), the
test-month evaluation (Table 4), the protected configurations and 8-bit baselines (Table 6), the calibration-week,
seasonal and cross-year checks (Sect. 4.6), native t7_a16 by weight construction (Table S14j), the native timing and peak memory (Table 8) and the cross-model
localisation ratios (Table S10), each next to the value printed in the paper, and exits with status 1 if any value
differs by more than the rounding of the paper. The intervals in summary.json are i.i.d. date-bootstrap intervals;
the paper's 3-day block-bootstrap intervals need the per-record archive (scripts/block_ci.py,
scripts/extended_analyses.py). Re-running the experiments (docs/REPRODUCE.md) writes new summaries into results/;
use --results results to compare a re-run.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (card, split, paper qerr %, decimals, paper ERA5 excess % or None, decimals)
TABLE3 = [("q0_sim", "dev", 147.5, 1, 84.8, 1), ("c04_no_rot", "dev", 164.0, 1, 101.3, 1), ("c11_norot_gptq", "dev", 156.9, 1, 95.2, 1),
          ("c09_smooth05_gptq", "dev", 103.8, 1, 50.6, 1), ("g01_svdq_a05_r32", "dev", 32.8, 1, 6.8, 1),
          ("g02_svdq_a08_r32", "dev", 33.7, 1, 7.3, 1), ("g03_svdq_a08_r64", "dev", 28.6, 1, 5.9, 1),
          ("t7_g02_nodec_fold", "dev", 17.3, 1, 1.3, 1), ("t7_mx", "dev", 10.5, 1, 0.65, 2), ("t7_a16", "dev", 9.4, 1, 0.54, 2),
          ("c08_weight_only_gptq", "dev", 14.1, 1, 0.9, 1), ("g02_mx", "dev", 16.2, 1, 1.8, 1)]
TABLE4 = [("bf16", "test", 1.75, 2, 0.00, 2), ("q0_sim", "test", 147.24, 2, 84.50, 2), ("m01_q0_decoder_only", "test", 129.03, 2, 75.89, 2),
          ("m02_q0_no_decoder", "test", 46.36, 2, 10.53, 2), ("c08_weight_only_gptq", "test", 14.20, 2, 1.02, 2),
          ("g02_svdq_a08_r32", "test", 33.62, 2, 7.16, 2), ("t7_g02_nodec_fold", "test", 17.32, 2, 1.27, 2),
          ("t7_mx", "test", 10.56, 2, 0.61, 2), ("t7_a16", "test", 9.46, 2, 0.50, 2)]
TABLE6 = [("c08_t7set", "test", 10.61, 2, 0.50, 2), ("t7mx_all314_noenc", "test", 17.09, 2, 1.46, 2),
          ("t7a16_all314_noenc", "test", 14.38, 2, 1.03, 2), ("u8_w8a16_mx_t7", "test", 5.12, 2, 0.14, 2),
          ("u8_w8a16_mx", "test", 18.46, 2, 0.39, 2), ("u8_w8a16_ch", "test", 17.80, 2, 3.02, 2),
          ("u8_w8a16_mx_prot", "test", 10.40, 2, 0.66, 2), ("u8_w8a8_mx", "test", 107.29, 2, 47.84, 2),
          ("u8_w8a8_ch", "test", 97.71, 2, 41.53, 2), ("u8_w8a8_mx_prot", "test", 25.09, 2, 3.54, 2)]
ROBUST = [("t7_a16_cal2", "test", 9.34, 2, 0.40, 2), ("t7_a16", "summer", 9.81, 2, 0.57, 2), ("t7_a16", "y2023", 9.84, 2, 0.42, 2),
          ("t7_mx_cal2", "test", 10.40, 2, 0.51, 2), ("t7_mx_calsum", "test", 10.60, 2, 0.60, 2),
          ("t7_mx", "summer", 10.85, 2, 0.67, 2), ("q0_sim", "summer", 144.7, 1, None, 1),
          ("t7_mx", "y2023", 11.00, 2, 0.51, 2), ("c08_weight_only_gptq", "y2023", 14.89, 2, 1.11, 2),
          ("q0_sim", "y2023", 156.4, 1, 92.3, 1), ("bf16", "y2023", 1.81, 2, None, 2)]
# Table S14j: native t7_a16 by construction of the TorchAO weights (re-quantised, FP32 tensor scale, from codes)
TABLE_S14J = [("native_t7_t7_a16_w4a16_bf16rest", "test", 9.50, 2, 0.52, 2), ("native_t7_t7_a16_w4a16_exact_bf16rest", "test", 9.52, 2, 0.51, 2),
              ("native_t7_t7_a16_w4a16_codes_bf16rest", "test", 9.52, 2, 0.51, 2)]
# Table 8: (timing file in _aggregates, config, paper step time s, paper peak GiB)
TABLE8 = [("_native_timing_v14_backbone.json", "bf16", 1.143, 14.92), ("_native_timing_v14_backbone.json", "bf16_resident", 1.141, 13.29),
          ("_native_timing_v14_backbone.json", "w4a16", 1.184, 12.12), ("_native_timing_v14_backbone.json", "w4a4", 1.154, 14.00),
          ("_native_timing_v14_backbone_fp8.json", "fp8_w8a16", 1.201, 12.49), ("_native_timing_v14_backbone_mx.json", "mxfp8_w8a8", 1.430, 15.49)]
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
    print(f"\n{title}\n{'card':26s} {'split':7s} {'qerr %':>9s} {'paper':>8s} {'excess %':>9s} {'paper':>8s}  i.i.d. date-bootstrap 95% interval of excess")
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
    table("Table 3: development of the protected configuration (development split, 12 dates x 2 steps)", TABLE3, res, bad)
    table("Table 4: test month (2024-03-01..30, 30 dates x 2 steps)", TABLE4, res, bad)
    table("Table 6: protected configurations and 8-bit baselines (test month)", TABLE6, res, bad)
    table("Sect. 4.6: calibration week, season, calibration season and year", ROBUST, res, bad)
    table("Table S14j: native t7_a16 (TorchAO W4A16) by weight construction", TABLE_S14J, res, bad)
    print("\nTable 8: native timing (median step s) and peak memory (GiB), RTX 5090, GPU otherwise idle")
    agg = res / "pretrained" / "_aggregates"
    for fname, cfg, pt, pm in TABLE8:
        f = agg / fname
        if not f.exists():
            print(f"  {cfg:14s} ({fname} not found)"); bad.append(f"timing/{cfg}"); continue
        r = json.load(open(f, encoding="utf-8"))["configs"][cfg]
        ok = close(r["step_seconds_median"], pt, 3) and close(r["peak_gib"], pm, 2)
        if not ok:
            bad.append(f"timing/{cfg}")
        print(f"  {cfg:14s} step {r['step_seconds_median']:.3f} (paper {pt:.3f})  peak {r['peak_gib']:.2f} (paper {pm:.2f}){'' if ok else '   <-- differs'}")
    nat = res / "pretrained" / "native_t7_t7_a16_w4a16_bf16rest__test_h2" / "build_info.json"
    if nat.exists():
        st = json.load(open(nat, encoding="utf-8"))["step_seconds_median"]
        ok = close(st, 1.838, 3)
        bad += [] if ok else ["timing/native_t7_a16"]
        print(f"  {'native t7_a16':14s} step {st:.3f} (paper 1.838){'' if ok else '   <-- differs'}")
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
