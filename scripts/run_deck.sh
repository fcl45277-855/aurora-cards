#!/usr/bin/env bash
# Run cards (recipes) in order on one GPU; resumable: dates already scored are skipped by run_card.py.
#   bash scripts/run_deck.sh <split> <steps> <recipe> [<recipe> ...]
#   e.g. bash scripts/run_deck.sh test 2 fp32 bf16 q0_sim t7_mx
# fp32 must run first on a split: it writes the FP32 reference cache that every other card is scored against.
# Logs: logs/deck_<split>.log; results under results/<recipe>__<split>_h<steps>/.
set -uo pipefail
cd "$(dirname "$0")/.."
SPLIT=$1; STEPS=$2; shift 2
PY=${PY:-python}
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p logs
for c in "$@"; do
  echo "== $(date '+%F %T') $c" | tee -a logs/deck_${SPLIT}.log
  $PY -u scripts/run_card.py --recipe recipes/$c.json --split $SPLIT --steps $STEPS >> logs/deck_${SPLIT}.log 2>&1
  tail -1 logs/deck_${SPLIT}.log
done
echo "DECK_DONE $(date '+%F %T')" | tee -a logs/deck_${SPLIT}.log
