#!/usr/bin/env bash
# Example 3 (GPU >= 24 GB, about 30-40 min including downloads): the full pipeline on one test date.
#
#   1. downloads the Aurora 0.25 pretrained checkpoint (5 GB, Hugging Face) and two ERA5 days (2 x 1.1 GB, CDS)
#   2. runs the FP32 reference, naive full quantisation (q0_sim) and the deployable recipe (t7_mx, calibrated here on
#      a single January day instead of the paper's week, recipe example_t7_mx_calib1) on 2024-03-01, 2 steps of 6 h
#   3. prints qerr and the ERA5 excess of each card
#
# Expected: q0_sim qerr ~ 140-150 % (paper, test-month mean: 147.2), t7_mx ~ 10-11 % (paper: 10.56; one date and a
# one-day calibration add noise). Full reproduction of the tables: docs/REPRODUCE.md.
#
#   bash examples/03_one_date.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export AQ_CKPT_DIR=${AQ_CKPT_DIR:-$PWD/checkpoints}
export AQ_ERA5_DIR=${AQ_ERA5_DIR:-$PWD/data/era5}
export AQ_PERCEIVER_CHUNK=${AQ_PERCEIVER_CHUNK:-16384}   # resamplers in 16384-patch chunks: peak GPU memory ~15 GiB
mkdir -p "$AQ_CKPT_DIR" "$AQ_ERA5_DIR"

[ -f "$AQ_CKPT_DIR/aurora-0.25-pretrained.ckpt" ] || \
  huggingface-cli download microsoft/aurora aurora-0.25-pretrained.ckpt --local-dir "$AQ_CKPT_DIR"
python scripts/download_static.py --out "$AQ_ERA5_DIR"
python scripts/download_era5_day.py 2024-01-04 2024-03-01 --out "$AQ_ERA5_DIR"
python scripts/check_env.py

for r in fp32 q0_sim example_t7_mx_calib1; do
  python -u scripts/run_card.py --recipe recipes/$r.json --split test --steps 2 --max-dates 1
done
python - <<'EOF'
import json
for r in ("q0_sim", "example_t7_mx_calib1"):
    s = json.load(open(f"results/{r}__test_h2/summary.json"))
    print(f"{r:22s} qerr {s['qerr_pct']:7.2f} %   ERA5 excess {s['excess_pct']:+6.2f} %   (1 date)")
EOF
