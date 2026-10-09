#!/usr/bin/env bash
# Ambient babble-noise overlay for the VCM + sesame wakeword datasets
# (feature-engineering/ambient-noise-overlay/SPEC.md).
#
# Materializes ASR-gated ambient-noise siblings (p_mix per row, 2 attempts
# with distinct noise chunks, SNR uniform in [SNR_MIN_DB, SNR_MAX_DB] dB)
# into derived manifests the existing train/eval targets consume via
# manifest override:
#   VCM:      out/conversions/v2/optionb-v3-vcmx-fil50-ambient/manifest.csv
#             (base: optionb-v3-vcmx-fil50 -- the manifest option-d-fil50 trained on)
#   sesame:   out/conversions/v2/wakeword-sesame/manifest.ambient.csv
# Every stored mix passed the transcriber gate; the realized SNR histogram
# and the license block are in each out root's summary.md.
#
# Usage:
#   GPUS="1 7" scripts/ambient_noise.sh                # all stages (0 corpus, 1 mix, 2 transcribe,
#                                                      # 3 finalize, 4 train, 5 eval)
#   GPUS="1" PILOT=50 scripts/ambient_noise.sh         # stages 0-3 on the first 50 eligible rows per model, then STOP for listening
#   STAGES="4 5" GPUS="1 7" scripts/ambient_noise.sh   # resume; finished stages skip unless FORCE=1
#
# Stage 2 (transcribe) needs a GPU and runs the external
# ~/simple-audio-transcriber under its own venv (accent_balance_fil50.sh
# convention): v1-pair QA over each model's shim dir, plus a single-process
# free-decode of the sesame `_unknown_` dir (scripts/ambient_free_decode.py).
#
# Notes:
#   - The corpus (stage 0) is non-idempotent by design (unseeded chunk
#     boundaries, like every sibling corpus): re-running stage 0 replaces
#     out/conversions/v2/ambient_noise/ with a new-but-consistent chunk set.
#     Skip it (STAGES="1 2 3") to reuse an existing one.
#   - PILOT outputs get pilot-suffixed names; the pilot and full run share
#     the sesame staging dir (it is transient and cleared at each mix).
#
# Detach for long runs:  nohup scripts/ambient_noise.sh > ambient.log 2>&1 & disown
# Each stage writes $ROOT/.done/<stage>; a finished stage is skipped unless FORCE=1.
set -euo pipefail

cd "$(dirname "$0")/.."

UV=${UV:-$(command -v uv || echo /opt/uv/uv)}
STAGES=${STAGES:-"0 1 2 3 4 5"}
GPUS=${GPUS:-}
PILOT=${PILOT:-}
FORCE=${FORCE:-0}
SEED=${SEED:-0}
P_MIX=${P_MIX:-0.5}
SNR_MIN_DB=${SNR_MIN_DB:-0}
SNR_MAX_DB=${SNR_MAX_DB:-30}
QA_MODEL=${QA_MODEL:-small}
QA_THRESHOLD=${QA_THRESHOLD:-0.80}
QA_REPO=${QA_REPO:-$HOME/simple-audio-transcriber}
# Stage 4 time budgets, matching the original runs for a fair before/after
# (out/vcm/option-d-fil50: max_minutes=90 optiond; out/wakeword-sesame: 30 default).
VCM_MINUTES=${VCM_MINUTES:-90}
WAKEWORD_MINUTES=${WAKEWORD_MINUTES:-30}

CORPUS=out/conversions/v2/ambient_noise
# VCM base = the fil50 treatment manifest (what trained out/vcm/option-d-fil50):
# base optionb-v3-vcmx + 9,958 clean fil50_persona rows + 4,435 _noisy variants
# (the noisy variants are ineligible, the clean ones get ambient siblings too).
VCM_BASE=out/conversions/v2/optionb-v3-vcmx-fil50/manifest.csv
WW_BASE=out/conversions/v2/wakeword-sesame/manifest.csv

# Run state (.done markers, logs, pilot report + listen dir) lives in its own
# root; the derived data lives in the per-model out roots the SPEC names.
ROOT=out/ambient-noise${PILOT:+-pilot$PILOT}

if [ -n "$PILOT" ]; then
    VCM_OUT=out/conversions/v2/optionb-v3-vcmx-fil50-ambient-pilot$PILOT
    WW_MAN_SUFFIX="-pilot$PILOT"
    WW_AUDIO_DIR=out/conversions/v2/wakeword-sesame/ambient-pilot$PILOT/audio
    WW_SUMMARY=out/conversions/v2/wakeword-sesame/summary.ambient-pilot$PILOT.md
else
    VCM_OUT=out/conversions/v2/optionb-v3-vcmx-fil50-ambient
    WW_MAN_SUFFIX=""
    WW_AUDIO_DIR=out/conversions/v2/wakeword-sesame/ambient/audio
    WW_SUMMARY=out/conversions/v2/wakeword-sesame/summary.md
fi
WW_OUT=out/conversions/v2/wakeword-sesame
VCM_OUT_MANIFEST=$VCM_OUT/manifest.csv
WW_OUT_MANIFEST=$WW_OUT/manifest.ambient$WW_MAN_SUFFIX.csv

# Train/eval run dirs (stage 4/5, accent_balance_fil50.sh convention): new runs
# live next to the baselines they are compared against.
VCM_RUN=out/vcm/option-d-fil50-ambient${PILOT:+-pilot$PILOT}
VCM_OLD_RUN=out/vcm/option-d-fil50
WW_RUN=out/wakeword-sesame-ambient${PILOT:+-pilot$PILOT}
WW_OLD_RUN=out/wakeword-sesame

VCM_STAGING=$VCM_OUT/.ambient.staging
WW_STAGING=$WW_OUT/.ambient.staging

if [ -n "$PILOT" ]; then
    # The trailing `true` matters under `set -e`: without it, the last command
    # run inside the $(...) is whichever `[ "$s" -le 3 ]` test happened to run
    # last (false for the final non-pilot stage), and that exit status becomes
    # this assignment's own exit status -- killing the whole script right here.
    STAGES=$(for s in $STAGES; do [ "$s" -le 3 ] && printf '%s ' "$s"; done; true)
fi

mkdir -p "$ROOT/.done" "$ROOT/logs"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

need_gpus() {
    [ -n "$GPUS" ] || { echo "GPUS must be set explicitly (check nvidia-smi for idle ones)" >&2; exit 1; }
}

first_gpu() { set -- $GPUS; echo "$1"; }

run_stage() {
    local n=$1 name=$2; shift 2
    case " $STAGES " in *" $n "*) ;; *) return 0 ;; esac
    if [ -f "$ROOT/.done/$n" ] && [ "$FORCE" != 1 ]; then
        log "stage $n ($name): already done, skipping"
        return 0
    fi
    log "stage $n ($name): start"
    "$@"
    touch "$ROOT/.done/$n"
    log "stage $n ($name): done"
}

stage0_corpus() {
    "$UV" run --with soundfile --with numpy python "$CORPUS/build_corpus.py"
}

mix_model() {
    local model=$1 manifest=$2 out_root=$3
    "$UV" run python -m me2_voicegen.dataset_tools.mix_ambient_noise \
        --model "$model" --stage mix \
        --manifest "$manifest" --corpus-root "$CORPUS" --out-root "$out_root" \
        --p-mix "$P_MIX" --snr-min-db "$SNR_MIN_DB" --snr-max-db "$SNR_MAX_DB" \
        --seed "$SEED" ${PILOT:+--max-rows "$PILOT"}
}

stage1_mix() {
    mix_model vcm "$VCM_BASE" "$VCM_OUT"
    mix_model wakeword "$WW_BASE" "$WW_OUT"
}

v1_pair_qa() {
    local staging=$1 unit=$2
    local shim=$staging/qa/shim/$unit reports=$staging/qa/reports
    [ -d "$shim" ] || { log "no $unit shim dir at $shim (no such jobs planned); skipping QA"; return 0; }
    # Stale reports from a prior mix of the same staging must never gate this
    # run's mixes: clear them, then QA (re)writes one report per unit.
    rm -f "$reports"/*.md 2>/dev/null || true
    (cd "$QA_REPO" && CUDA_VISIBLE_DEVICES=$(first_gpu) "$UV" run python -m audio_transcript_parser.qa \
        "$OLDPWD/$shim" --naming v1-pair --model "$QA_MODEL" \
        --device cuda --threshold "$QA_THRESHOLD" --out-dir "$OLDPWD/$reports")
}

stage2_transcribe() {
    need_gpus
    v1_pair_qa "$VCM_STAGING" vcm
    v1_pair_qa "$WW_STAGING" wakeword_sesame
    if [ -d "$WW_STAGING/qa/free_decode" ]; then
        (cd "$QA_REPO" && CUDA_VISIBLE_DEVICES=$(first_gpu) "$UV" run python \
            "$OLDPWD/scripts/ambient_free_decode.py" \
            --in-dir "$OLDPWD/$WW_STAGING/qa/free_decode" \
            --out-csv "$OLDPWD/$WW_STAGING/qa/decisions.csv" \
            --model "$QA_MODEL" --device cuda)
    else
        log "no sesame free_decode dir (no _unknown_ jobs planned); skipping free-decode"
    fi
}

stage3_finalize() {
    "$UV" run python -m me2_voicegen.dataset_tools.mix_ambient_noise \
        --model vcm --stage finalize \
        --manifest "$VCM_BASE" --corpus-root "$CORPUS" --out-root "$VCM_OUT" \
        --reports-dir "$VCM_STAGING/qa/reports" \
        --audio-dir "$VCM_OUT/audio" \
        --manifest-out "$VCM_OUT/manifest.csv" \
        --summary-out "$VCM_OUT/summary.md" \
        --p-mix "$P_MIX" --snr-min-db "$SNR_MIN_DB" --snr-max-db "$SNR_MAX_DB" --seed "$SEED"

    local neg_flags=()
    [ -f "$WW_STAGING/qa/decisions.csv" ] && neg_flags=(--negatives-decisions "$WW_STAGING/qa/decisions.csv")
    "$UV" run python -m me2_voicegen.dataset_tools.mix_ambient_noise \
        --model wakeword --stage finalize \
        --manifest "$WW_BASE" --corpus-root "$CORPUS" --out-root "$WW_OUT" \
        --reports-dir "$WW_STAGING/qa/reports" \
        ${neg_flags[@]+"${neg_flags[@]}"} \
        --audio-dir "$WW_AUDIO_DIR" \
        --manifest-out "$WW_OUT/manifest.ambient$WW_MAN_SUFFIX.csv" \
        --summary-out "$WW_SUMMARY" \
        --p-mix "$P_MIX" --snr-min-db "$SNR_MIN_DB" --snr-max-db "$SNR_MAX_DB" --seed "$SEED"

    if [ -n "$PILOT" ]; then
        "$UV" run python scripts/ambient_pilot_report.py \
            --vcm-manifest "$VCM_OUT/manifest.csv" --vcm-summary "$VCM_OUT/summary.md" \
            --ww-manifest "$WW_OUT/manifest.ambient$WW_MAN_SUFFIX.csv" --ww-summary "$WW_SUMMARY" \
            --out-root "$ROOT"
    fi
}

stage4_train() {
    need_gpus
    local g g2
    g=$(first_gpu)
    g2=$(set -- $GPUS; echo "${2:-$1}")
    [ -f "$VCM_OLD_RUN/checkpoints/checkpoint.pt" ] || { echo "baseline checkpoint missing: $VCM_OLD_RUN/checkpoints/checkpoint.pt" >&2; exit 1; }
    [ -f "$WW_OLD_RUN/checkpoints/checkpoint.pt" ] || { echo "baseline checkpoint missing: $WW_OLD_RUN/checkpoints/checkpoint.pt" >&2; exit 1; }
    log "stage 4: VCM on cuda:$g (optiond, ${VCM_MINUTES} min), wakeword on cuda:$g2 (${WAKEWORD_MINUTES} min), seed $SEED"
    make optionb-train OPTIONB_MANIFEST="$VCM_OUT_MANIFEST" OPTIONB_OUT_DIR="$VCM_RUN" \
        VCM_PRESET=optiond VCM_DEVICE=cuda:"$g" VCM_MAX_MINUTES="$VCM_MINUTES" VCM_SEED="$SEED" \
        > "$ROOT/logs/train_vcm.log" 2>&1 &
    local vcm_pid=$!
    make wakeword-train WAKEWORD_TRAIN_MANIFEST="$WW_OUT_MANIFEST" WAKEWORD_TRAIN_OUT_DIR="$WW_RUN" \
        WAKEWORD_DEVICE=cuda:"$g2" WAKEWORD_MAX_MINUTES="$WAKEWORD_MINUTES" WAKEWORD_TRAIN_SEED="$SEED" \
        > "$ROOT/logs/train_wakeword.log" 2>&1
    wait "$vcm_pid"
    log "stage 4: train logs in $ROOT/logs/train_vcm.log + train_wakeword.log"
}

stage5_eval() {
    need_gpus
    local g
    g=$(first_gpu)
    # New checkpoint on the ambient manifest (slot-eval subset = base optionb rows,
    # generated at train time; ambient rows excluded from slot/speaker diagnostics by design).
    make optionb-eval OPTIONB_MANIFEST="$VCM_OUT_MANIFEST" OPTIONB_OUT_DIR="$VCM_RUN" \
        VCM_DEVICE=cuda:"$g" > "$ROOT/logs/eval_vcm_new.log" 2>&1
    # Same-test before/after: the shipped baselines on the same ambient manifest,
    # written to separate out-dirs so the committed baseline reports stay untouched.
    "$UV" run python -m me2_voicegen.vcm.evaluate --manifest "$VCM_OUT_MANIFEST" \
        --checkpoint "$VCM_OLD_RUN/checkpoints/checkpoint.pt" --out-dir "$VCM_RUN/baseline_old_ckpt" \
        --device cuda:"$g" --beam-width 50 --grammar optionb \
        > "$ROOT/logs/eval_vcm_old.log" 2>&1
    "$UV" run python -m me2_voicegen.wakeword.train --manifest "$WW_OUT_MANIFEST" \
        --out-dir "$WW_RUN/eval_report" --eval-only --checkpoint "$WW_RUN/checkpoints/checkpoint.pt" \
        --device cuda:"$g" > "$ROOT/logs/eval_wakeword_new.log" 2>&1
    "$UV" run python -m me2_voicegen.wakeword.train --manifest "$WW_OUT_MANIFEST" \
        --out-dir "$WW_RUN/baseline_old_ckpt" --eval-only --checkpoint "$WW_OLD_RUN/checkpoints/checkpoint.pt" \
        --device cuda:"$g" > "$ROOT/logs/eval_wakeword_old.log" 2>&1
    make wakeword-export WAKEWORD_TRAIN_MANIFEST="$WW_OUT_MANIFEST" WAKEWORD_TRAIN_OUT_DIR="$WW_RUN" \
        > "$ROOT/logs/export_wakeword.log" 2>&1
    make wakeword-bench WAKEWORD_TRAIN_MANIFEST="$WW_OUT_MANIFEST" WAKEWORD_TRAIN_OUT_DIR="$WW_RUN" \
        > "$ROOT/logs/bench_wakeword.log" 2>&1
    make vcmx-export VCMX_TRAINED_OUT_DIR="$VCM_RUN" VCMX_SERVE_MANIFEST="$VCM_OUT_MANIFEST" \
        > "$ROOT/logs/export_vcm.log" 2>&1
    log "stage 5: evals in $VCM_RUN/ + $VCM_RUN/baseline_old_ckpt/, $WW_RUN/eval_report/ + $WW_RUN/baseline_old_ckpt/ (logs in $ROOT/logs/)"
}

run_stage 0 corpus     stage0_corpus
run_stage 1 mix        stage1_mix
run_stage 2 transcribe stage2_transcribe
run_stage 3 finalize   stage3_finalize
run_stage 4 train      stage4_train
run_stage 5 eval       stage5_eval

if [ -n "$PILOT" ]; then
    log "pilot complete (PILOT=$PILOT) -- review $ROOT/pilot_report.md + $ROOT/listen/ BEFORE the full run"
else
    log "all requested stages finished: $STAGES"
    case " $STAGES " in
        *" 5 "*) log "before/after: compare $VCM_RUN/metadata/eval_report.md vs $VCM_RUN/baseline_old_ckpt/metadata/eval_report.md (VCM); $WW_RUN/eval_report/metadata/eval_report.md vs $WW_RUN/baseline_old_ckpt/metadata/eval_report.md (wakeword)" ;;
    esac
fi
