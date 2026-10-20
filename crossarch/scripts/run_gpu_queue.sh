#!/bin/bash
# GPU part of the ClimaX cross-architecture check, after the CPU preparation in crossarch/README.md (steps 1-4):
#   full-model check -> fine-tuning with the persistence gate -> evaluation config -> plumbing, January predictor,
#   freeze, March confirmation, July season (W4A4), March W4A32, fixed-criterion analysis.
# Completed steps are skipped on a re-run; any failure stops the queue. Logs go to $CROSSARCH_ROOT/queue.log.
#
#   CROSSARCH_ROOT=/path/to/work AQ_ERA5_DIR=/path/to/era5 bash crossarch/scripts/run_gpu_queue.sh
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
: "${CROSSARCH_ROOT:?set CROSSARCH_ROOT to the work directory}"
PY=${PYTHON:-python}
CFG=$CROSSARCH_ROOT/configs/climax_config.json
R=$CROSSARCH_ROOT/results
LOG=$CROSSARCH_ROOT/queue.log
mkdir -p "$R"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
step() { local name=$1; shift; log "START $name"; "$@" >> "$LOG" 2>&1; local rc=$?; log "END $name rc=$rc"; return $rc; }

$PY -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || { log "no CUDA device"; exit 1; }
[ -f "$CROSSARCH_ROOT/finetune_era5_20261020/training_data_manifest.json" ] || { log "training data incomplete: run prepare_training.py"; exit 1; }
cd "$HERE" || exit 1

step full_model_check $PY check_full_model.py || exit 1
if [ ! -f "$CROSSARCH_ROOT/climax_finetune_run/provenance.json" ]; then
    step train_check $PY train_climax_era5.py --check || exit 1
    step train $PY train_climax_era5.py
    rc=$?
    [ $rc -eq 3 ] && { log "fine-tuned ClimaX did not beat persistence: quantisation runs not authorised"; exit 3; }
    [ $rc -eq 0 ] || exit 1
fi
[ -f "$CFG" ] || step config $PY make_climax_config.py --output "$CFG" || exit 1
step gate $PY run_climax.py --config "$CFG" --check || exit 1
[ -f "$R/climax_plumbing_W4A4.json" ] || step plumbing $PY run_climax.py --config "$CFG" --split plumbing || exit 1
[ -f "$R/climax_predictor_W4A4.json" ] || step predictor $PY run_climax.py --config "$CFG" --split predictor || exit 1
[ -f "$CROSSARCH_ROOT/metadata/frozen_predictor.json" ] || step freeze $PY freeze_predictor.py "$R/climax_predictor_W4A4.json" || exit 1
[ -f "$R/climax_confirmation_W4A4.json" ] || step confirmation $PY run_climax.py --config "$CFG" --split confirmation || exit 1
[ -f "$R/climax_summer_W4A4.json" ] || step summer $PY run_climax.py --config "$CFG" --split summer || exit 1
[ -f "$R/climax_confirmation_W4A32.json" ] || step confirmation_w4a32 $PY run_climax.py --config "$CFG" --split confirmation --mode W4A32 || exit 1
[ -f "$R/climax_analysis.json" ] || step analysis $PY analyze_climax.py "$R/climax_confirmation_W4A4.json" --output "$R/climax_analysis.json" || exit 1
log "ClimaX queue finished"
