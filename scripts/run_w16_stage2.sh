#!/usr/bin/env bash
# W4A16 ablation, stage 2: run only the branches whose prespecified trigger was met in stage 1 (Online Resource 1, S15).
#   bash scripts/run_w16_stage2.sh a1        # criterion A-1 passed (GPTQ-only W4A16 close to t7_a16); not triggered in the paper
#   bash scripts/run_w16_stage2.sh b1        # criterion B-1 "decisive" or "intermediate" (run in the paper; Table S15c)
# Timing (branch a1) needs an otherwise idle GPU, as in Table 12.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
export AQ_MODEL=${AQ_MODEL:-pretrained} AQ_PERCEIVER_CHUNK=${AQ_PERCEIVER_CHUNK:-16384} PYTORCH_ALLOC_CONF=expandable_segments:True
ROLL="2024-03-01 2024-03-02 2024-03-03 2024-03-04 2024-03-05 2024-03-06"

wait_gpu() { while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 60; done; }
run() { wait_gpu; echo "== $(date '+%F %T') $*"; "$@" || echo "!! FAILED $*"; }

for branch in "$@"; do
  case "$branch" in
  a1)
    run "$PY" -u scripts/run_card.py --recipe recipes/w16_t7_gptq.json --split oct --steps 2
    run "$PY" -u scripts/run_card.py --recipe recipes/w16g_t7_dec.json --split dev --steps 2
    run "$PY" -u scripts/run_card.py --recipe recipes/w16g_t7_dec.json --split test --steps 2
    run "$PY" -u scripts/run_card.py --recipe recipes/w16_t7_gptq.json --split test --steps 20 --max-dates 12
    run "$PY" -u scripts/native_t7.py --recipe w16_t7_gptq --act w4a16 --split test --steps 2 --exact-weights --bf16-rest q0:inc=backbone
    run "$PY" -u scripts/run_card.py --recipe recipes/fp32.json --split dev --steps 2 --max-dates 1 --results results_timing_w16
    for r in t7_a16 w16_t7_gptq; do
      run "$PY" -u scripts/native_t7.py --recipe "$r" --act w4a16 --split dev --steps 2 --max-dates 1 --time-repeats 10 \
        --exact-weights --bf16-rest q0:inc=backbone --results results_timing_w16
    done
    ;;
  b1)
    run "$PY" -u scripts/diag_fields.py --recipes w16_t7_dec --split test --steps 20 --dates $ROLL --tag w16
    run "$PY" -u scripts/physmetrics_fields.py --recipes w16_t7_dec --split test --steps 20 --dates $ROLL --tag w16
    ;;
  *) echo "unknown branch $branch (use a1 and/or b1)"; exit 2 ;;
  esac
done
echo "== $(date '+%F %T') stage 2 done: $*"
