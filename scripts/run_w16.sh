#!/usr/bin/env bash
# Prespecified W4A16 ablation of t7_a16, stage 1 (Tables 5 and 7, upper blocks; Online Resource 1, Sect. S15).
#   bash scripts/run_w16.sh > logs/w16_stage1.log 2>&1
# Needs the fp32 and t7_a16 cards on the dev and test splits (Table 4) and the layer lists in recipes/sets/
# (released; regenerate with scripts/make_w16_sets.py). Cards with a summary.json are skipped, so the script can be
# restarted. Before each card it waits until the GPU has no compute process; MIN_FREE_MIB=N instead starts a card
# whenever N MiB are free (shared GPU). It never signals other processes.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
export AQ_MODEL=${AQ_MODEL:-pretrained} AQ_PERCEIVER_CHUNK=${AQ_PERCEIVER_CHUNK:-16384} PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p logs

wait_gpu() {
  if [ -n "${MIN_FREE_MIB:-}" ]; then
    while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt "$MIN_FREE_MIB" ]; do sleep 60; done
  else
    while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 60; done
  fi
}
card() {  # card <recipe> <split>
  local out="results/$1__$2_h2"
  if [ -f "$out/summary.json" ]; then echo "skip $out"; return; fi
  wait_gpu
  echo "== $(date '+%F %T') $1 $2"
  "$PY" -u scripts/run_card.py --recipe "recipes/$1.json" --split "$2" --steps 2 || echo "!! FAILED $1 $2"
}

# S0: a copy of t7_a16 under a new name, rebuilt without the build cache (calibration + GPTQ from scratch), must
# reproduce the dev result of t7_a16 (qerr 9.38 / excess +0.54 %); the script stops otherwise
AQ_BUILD_CACHE=0 card w16_ref_t7_a16 dev
"$PY" - <<'PYEOF' || { echo "!! S0 failed: environment does not reproduce t7_a16, stopping"; exit 1; }
import json
a = json.load(open("results/w16_ref_t7_a16__dev_h2/summary.json"))
b = json.load(open("results/t7_a16__dev_h2/summary.json"))
print("S0 qerr", a["qerr_pct"], b["qerr_pct"], "excess", a["excess_pct"], b["excess_pct"])
assert abs(a["qerr_pct"] - b["qerr_pct"]) <= 0.02 and abs(a["excess_pct"] - b["excess_pct"]) <= 0.005, "S0 mismatch"
PYEOF

# Block A (recipe components) and Block B (protections): dev first (screening only), then the test month (judged)
A="w16_t7_gptq w16_t7_gptq_sq w16_t7_gptq_lr w16_t7_rtn_sq_lr w16_t7_rtn"
B="w16_t7_dec w16_t7_enc w16_t7_res2 w16_t7_static"
for r in $A $B; do card "$r" dev; card "$r" test; done
card g02_a16 test                       # B5: t7_a16 numerics on the full Q0 set

# Physical diagnostics on the test month (Table 10 and Table S9 definitions; Table S15b)
wait_gpu
"$PY" -u scripts/physmetrics_fields.py --recipes w16_t7_dec w16_t7_enc w16_t7_static g02_a16 w16_t7_gptq --split test --steps 2 --tag w16
wait_gpu
"$PY" -u scripts/diag_fields.py --recipes w16_t7_dec w16_t7_enc w16_t7_static g02_a16 w16_t7_gptq --split test --steps 2 --tag w16

# Intervals and paired differences (3-day moving blocks, 2000 resamples)
"$PY" scripts/block_ci.py --split test $A $B g02_a16 --ref t7_a16 --full > results/_w16_block_ci.txt
"$PY" scripts/extended_analyses.py --sections A --pairs \
  w16_t7_gptq:t7_a16 w16_t7_gptq_sq:t7_a16 w16_t7_gptq_lr:t7_a16 w16_t7_rtn_sq_lr:t7_a16 w16_t7_rtn:t7_a16 \
  w16_t7_dec:t7_a16 w16_t7_enc:t7_a16 w16_t7_res2:t7_a16 w16_t7_static:t7_a16 g02_a16:t7_a16 \
  > results/_w16_paired.txt
echo "== $(date '+%F %T') stage 1 done"
