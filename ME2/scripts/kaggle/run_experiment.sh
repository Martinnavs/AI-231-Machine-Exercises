#!/usr/bin/env bash
# Full experiment: train -> eval (clean + fixed-seed noisy gate) -> INT8 export/bench
# -> end-to-end latency -> RESULTS.md -> packaged tarball. Mirrors the recipe in
# docs/archive/QUARTZNET-STUDENT.md "Reproducing".
#
# Stages are resumable at stage granularity (a stage that finished is skipped on re-run;
# training itself has no mid-run resume, so an interrupted `train` restarts from scratch).
#
# Env (all optional):
#   PRESET=quartznet5x3  SEED=0  MAX_MINUTES=135  MAX_EPOCHS=150  P_RIR=0.7  BATCH_SIZE=16
#   RUN_NAME=<preset>-seed<seed>-<minutes>m   OUT_ROOT=out/vcm
#   STAGES="train eval export e2e report package"   FORCE=1 (ignore done-markers)
#   SMOKE=1   tiny data + ~1 min budgets: validates the whole pipeline before a long run
#   TRAIN_MANIFEST / EVAL_MANIFEST / GRAMMAR=optionb / BEAM=50 / NOISY_SEED=0
#   DEVICE=auto|cuda:0|cpu   NUM_WORKERS=<min(nproc,8)>   PY="python"
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=${PY:-python}
export PYTHONPATH=${PYTHONPATH:-src}
PRESET=${PRESET:-quartznet5x3}
SEED=${SEED:-0}
MAX_MINUTES=${MAX_MINUTES:-135}
MAX_EPOCHS=${MAX_EPOCHS:-150}
P_RIR=${P_RIR:-0.7}
BATCH_SIZE=${BATCH_SIZE:-16}
GRAMMAR=${GRAMMAR:-optionb}
BEAM=${BEAM:-50}
NOISY_SEED=${NOISY_SEED:-0}
TRAIN_MANIFEST=${TRAIN_MANIFEST:-out/conversions/v2/optionb-v3-vcmx-fil50-ambient/manifest.csv}
EVAL_MANIFEST=${EVAL_MANIFEST:-out/conversions/v2/optionb-v3-vcmx-fil50/manifest.csv}
OUT_ROOT=${OUT_ROOT:-out/vcm}
STAGES=${STAGES:-"train eval export e2e report package"}
CALIB=32; BENCH_ITERS=100; E2E_CLIPS=200; EXTRA_EVAL=""
SMOKE=${SMOKE:-0}

if [ "$SMOKE" = 1 ]; then
  MAX_MINUTES=2; MAX_EPOCHS=2; CALIB=4; BENCH_ITERS=10; E2E_CLIPS=10
  EXTRA_EVAL="--noisy-eval-rir-pool-size 10"
  # smoke is a wiring check: don't fan out over every core of a shared box
  export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
  NUM_WORKERS=${NUM_WORKERS:-2}
  RUN_NAME=${RUN_NAME:-smoke-${PRESET}}
else
  RUN_NAME=${RUN_NAME:-${PRESET}-seed${SEED}-${MAX_MINUTES}m}
fi
RUN_DIR=$OUT_ROOT/$RUN_NAME
MARK=$RUN_DIR/.stages
mkdir -p "$RUN_DIR" "$MARK"

if [ "${DEVICE:-auto}" = auto ]; then
  DEVICE=$($PY -c "import torch; print('cuda:0' if torch.cuda.is_available() else 'cpu')")
fi
if [ -z "${NUM_WORKERS:-}" ]; then
  n=$(nproc); NUM_WORKERS=$(( n < 8 ? n : 8 ))
fi

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
log "run=$RUN_NAME preset=$PRESET seed=$SEED device=$DEVICE workers=$NUM_WORKERS smoke=$SMOKE stages: $STAGES"

if [ "$SMOKE" = 1 ]; then
  $PY scripts/kaggle/smoke_manifest.py --manifest "$TRAIN_MANIFEST" --out "$RUN_DIR/smoke/train.csv" --seed "$SEED"
  $PY scripts/kaggle/smoke_manifest.py --manifest "$EVAL_MANIFEST" --out "$RUN_DIR/smoke/eval.csv" --seed "$SEED"
  TRAIN_MANIFEST=$RUN_DIR/smoke/train.csv; EVAL_MANIFEST=$RUN_DIR/smoke/eval.csv
fi

CKPT=$RUN_DIR/checkpoints/checkpoint.pt

stage_train() {
  $PY -m me2_voicegen.vcm.train --manifest "$TRAIN_MANIFEST" --out-dir "$RUN_DIR" \
    --preset "$PRESET" --device "$DEVICE" --max-minutes "$MAX_MINUTES" --max-epochs "$MAX_EPOCHS" \
    --seed "$SEED" --p-rir "$P_RIR" --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS"
}
stage_eval() {
  $PY -m me2_voicegen.vcm.evaluate --manifest "$EVAL_MANIFEST" --checkpoint "$CKPT" \
    --out-dir "$RUN_DIR/noisy_eval" --device "$DEVICE" --beam-width "$BEAM" --grammar "$GRAMMAR" \
    --noisy-eval-seed "$NOISY_SEED" $EXTRA_EVAL
}
stage_export() {
  $PY -m me2_voicegen.vcm.benchmark --checkpoint "$CKPT" --out-dir "$RUN_DIR" --manifest "$TRAIN_MANIFEST" \
    --calibration-samples "$CALIB" --n-iters "$BENCH_ITERS"
}
stage_e2e() {
  $PY -m me2_voicegen.vcm.e2e_benchmark --run-dir "$RUN_DIR" --manifest "$EVAL_MANIFEST" --split test \
    --n-clips "$E2E_CLIPS" --window-s 2.5 --beam-width "$BEAM" --grammar "$GRAMMAR" --seed 0 \
    --out "$RUN_DIR/metadata/e2e_benchmark.json"
}
stage_report() {
  {
    echo "# $RUN_NAME"
    echo
    echo "- preset \`$PRESET\`, seed $SEED, budget ${MAX_MINUTES} min, p_rir $P_RIR, device $DEVICE"
    echo "- code \`$(git rev-parse --short HEAD 2>/dev/null || echo unknown)\`$(git diff --quiet 2>/dev/null || echo ' (+uncommitted changes)')"
    echo "- train manifest \`$TRAIN_MANIFEST\`, eval manifest \`$EVAL_MANIFEST\`, grammar $GRAMMAR, beam $BEAM, noisy seed $NOISY_SEED"
    echo
    echo "## Training (tail of train.log)"; echo '```'; tail -n 8 "$RUN_DIR/train.log" 2>/dev/null || true; echo '```'
    for f in noisy_eval/metadata/eval_report.md metadata/vcm_benchmark.md; do
      [ -f "$RUN_DIR/$f" ] && { echo; echo "## $f"; echo; cat "$RUN_DIR/$f"; }
    done
    [ -f "$RUN_DIR/metadata/e2e_benchmark.json" ] && { echo; echo "## e2e_benchmark.json"; echo '```json'; cat "$RUN_DIR/metadata/e2e_benchmark.json"; echo '```'; }
    echo; echo "## INT8 export size"; ls -l "$RUN_DIR"/export/*.int8.onnx 2>/dev/null || echo "(no export)"
  } > "$RUN_DIR/RESULTS.md"
  cat "$RUN_DIR/RESULTS.md"
}
stage_package() {
  # Kaggle keeps /kaggle/working as the notebook output -> drop the tarball there.
  dest=${PACKAGE_DIR:-$([ -d /kaggle/working ] && echo /kaggle/working || echo "$OUT_ROOT")}
  mkdir -p "$dest"
  tar -czf "$dest/$RUN_NAME.tar.gz" -C "$OUT_ROOT" --exclude='smoke' --exclude='.stages' "$RUN_NAME"
  log "packaged $dest/$RUN_NAME.tar.gz ($(du -h "$dest/$RUN_NAME.tar.gz" | cut -f1))"
}

# stage -> log file; train.log keeps the name existing runs use.
for s in $STAGES; do
  if [ -f "$MARK/$s.done" ] && [ "${FORCE:-0}" != 1 ]; then log "skip $s (done; FORCE=1 to redo)"; continue; fi
  if [ "$s" != train ] && [ "$s" != report ] && [ "$s" != package ] && [ ! -f "$CKPT" ]; then
    log "stage $s needs $CKPT -- run the train stage first"; exit 1
  fi
  log "=== $s ==="; t0=$SECONDS
  if [ "$s" = report ] || [ "$s" = package ]; then
    "stage_$s"
  else
    lf=$RUN_DIR/$s.log; [ "$s" = eval ] && lf=$RUN_DIR/noisy_eval.log
    # PIPESTATUS: a failing python must fail the stage even though tee succeeds
    set +e; "stage_$s" 2>&1 | tee "$lf"; rc=${PIPESTATUS[0]}; set -e
    if [ "$rc" -ne 0 ]; then
      log "stage $s FAILED (rc=$rc); log: $lf"
      stage_report >/dev/null 2>&1 || true; stage_package || true
      exit "$rc"
    fi
  fi
  echo "$(date -Is) $((SECONDS - t0))s" > "$MARK/$s.done"
  log "$s done in $((SECONDS - t0))s"
done
log "all stages complete: $RUN_DIR"
