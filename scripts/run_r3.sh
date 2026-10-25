#!/usr/bin/env bash
# Residency of the BF16-resident memory reference (Table 12, last row of the first block; Table S17a, both blocks).
# All timings in one session with the GPU otherwise idle; dev 2024-02-02, 3 warm-up + 10 timed steps.
#   bash scripts/run_r3.sh > logs/r3.log 2>&1
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
export AQ_MODEL=${AQ_MODEL:-pretrained} AQ_PERCEIVER_CHUNK=${AQ_PERCEIVER_CHUNK:-16384} AQ_MEM_FRAC=${AQ_MEM_FRAC:-0.9} PYTORCH_ALLOC_CONF=expandable_segments:True
idle() { while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 60; done; }
run() { idle; echo "== $(date '+%F %T') $*"; "$@" || echo "!! FAILED $*"; }
OUT=results_timing_r3
mkdir -p "$OUT" "${OUT}_allbb" logs

# (1) references on the 187-layer Q0 backbone set (all other layers at Aurora's default FP32 residency)
run "$PY" -u scripts/native_timing.py --layers q0:inc=backbone --configs bf16 bf16_resident w4a16 --out "$OUT/_native_timing_q0.json"
# (2) reference with all 296 backbone linear layers BF16-resident
run "$PY" -u scripts/native_timing.py --layers all314:inc=backbone --configs bf16_resident --out "$OUT/_native_timing_allbackbone.json"
# (3) t7 builds: 138 layers NVFP4, the other 49 layers of the 187-layer set BF16, all other layers FP32 (as in (1))
run "$PY" -u scripts/run_card.py --recipe recipes/fp32.json --split dev --steps 2 --max-dates 1 --results "$OUT"
for r in t7_a16 w16_t7_gptq_sq w16_t7_gptq; do
  run "$PY" -u scripts/native_t7.py --recipe "$r" --act w4a16 --split dev --steps 2 --max-dates 1 --time-repeats 10 \
    --weights fp32-tensor-scale --bf16-rest q0:inc=backbone --results "$OUT"
done
# (4) t7 builds with every other backbone linear layer BF16-resident (as in (2))
run "$PY" -u scripts/run_card.py --recipe recipes/fp32.json --split dev --steps 2 --max-dates 1 --results "${OUT}_allbb"
for r in t7_a16 w16_t7_gptq_sq; do
  run "$PY" -u scripts/native_t7.py --recipe "$r" --act w4a16 --split dev --steps 2 --max-dates 1 --time-repeats 10 \
    --weights fp32-tensor-scale --bf16-rest all314:inc=backbone --results "${OUT}_allbb"
done
echo "== $(date '+%F %T') r3 done"
