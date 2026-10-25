#!/usr/bin/env bash
# Analyses added in the first revision of the Earth Science Informatics manuscript (all descriptive):
#   E1 native timing and peak memory without the low-rank branch (Table S17a, Table 11),
#   E2 group attribution of the 62 inherited BF16 layers (Table 5 lower block, Table S17b),
#   E3 20-step rollouts on 25 March and 9 October dates with ACC (Table 9, Table S17c).
#   bash scripts/run_r1.sh > logs/r1.log 2>&1
# Needs the fp32, t7_a16, c08_t7set and u8_w8a16_mx_prot cards (Tables 4 and 8), the oct split on disk and a CDS
# account for the climatology (scripts/clim_download.py runs in parallel; ACC waits for it).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
export AQ_MODEL=${AQ_MODEL:-pretrained} AQ_PERCEIVER_CHUNK=${AQ_PERCEIVER_CHUNK:-16384} AQ_MEM_FRAC=${AQ_MEM_FRAC:-0.6} PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p logs

wait_gpu() {   # MIN_FREE_MIB=N: share the GPU; IDLE=1: require a GPU without other processes (timing)
  if [ "${IDLE:-0}" = 1 ] || [ -z "${MIN_FREE_MIB:-}" ]; then
    while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 60; done
  else
    while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt "$MIN_FREE_MIB" ]; do sleep 60; done
  fi
}
run() { wait_gpu; echo "== $(date '+%F %T') $*"; "$@" || echo "!! FAILED $*"; }
card() { [ -f "results/$1__$2_h$3/summary.json" ] && [ "${4:-}" = "" ] && { echo "skip $1 $2 h$3"; return; }
         run "$PY" -u scripts/run_card.py --recipe "recipes/$1.json" --split "$2" --steps "$3" ${4:+--max-dates $4}; }

# climatology for ACC (background, CPU only)
CLIM_PID=""
if [ ! -f data_clim/era5_clim_1991_2020_1deg.npz ]; then
  "$PY" -u scripts/clim_download.py > logs/r1_clim.log 2>&1 &
  CLIM_PID=$!
fi

# E1: native timing and peak memory with an otherwise idle GPU, same session (Table S17a, upper block)
IDLE=1 run "$PY" -u scripts/run_card.py --recipe recipes/fp32.json --split dev --steps 2 --max-dates 1 --results results_timing_w16
for r in t7_a16 w16_t7_gptq_sq w16_t7_gptq; do
  IDLE=1 run "$PY" -u scripts/native_t7.py --recipe "$r" --act w4a16 --split dev --steps 2 --max-dates 1 --time-repeats 10 \
    --weights fp32-tensor-scale --bf16-rest q0:inc=backbone --results results_timing_w16
done
run "$PY" -u scripts/native_t7.py --recipe w16_t7_gptq_sq --act w4a16 --split test --steps 2 --weights fp32-tensor-scale --bf16-rest q0:inc=backbone

# E2: t7_a16 plus one group of the 62 layers that the all-layer configuration adds (lists in recipes/sets/)
G="surf_mlp patch enc0 enc1 enc2 dec0 dec1 dec2"
for g in $G; do card "w16_t7_inh_$g" test 2; done

# E3: 20-step rollouts on all usable March dates (25; the test data end on 03-30) and October dates (9)
card fp32 test 20 25
for r in t7_a16 w16_t7_gptq_sq u8_w8a16_mx_prot c08_t7set; do card "$r" test 20 25; done
card fp32 oct 20 9
for r in t7_a16 w16_t7_gptq_sq; do card "$r" oct 20 9; done

# ACC (needs the climatology)
[ -n "$CLIM_PID" ] && wait "$CLIM_PID"
if [ -f data_clim/era5_clim_1991_2020_1deg.npz ]; then
  run "$PY" -u scripts/acc_rollout.py --recipes fp32 t7_a16 w16_t7_gptq_sq u8_w8a16_mx_prot c08_t7set --split test --steps 20 --max-dates 25
  run "$PY" -u scripts/acc_rollout.py --recipes fp32 t7_a16 w16_t7_gptq_sq --split oct --steps 20 --max-dates 9
else
  echo "!! no climatology: ACC skipped (see logs/r1_clim.log)"
fi

# paired summaries (Table 5 lower block) and the rollout / ACC / group tables (Tables 9, S17b, S17c)
P="t7a16_all314_noenc:t7_a16"; for g in $G; do P="$P w16_t7_inh_$g:t7_a16"; done
"$PY" scripts/extended_analyses.py --sections A --pairs $P > results/_r1_groups_paired.txt
"$PY" scripts/r1_summary.py > results/_r1_summary.txt
for s in test oct; do "$PY" scripts/skill_hours.py --split $s --steps 20 t7_a16 w16_t7_gptq_sq; done > results/_r1_skill_hours.txt
echo "== $(date '+%F %T') r1 done"
