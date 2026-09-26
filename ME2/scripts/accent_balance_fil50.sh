#!/usr/bin/env bash
# 50/50 Filipino/non-Filipino rebalance of the VCM + wakeword positive classes,
# then retrain + evaluate both models. Plan: .scratch/accent-balance-fil50/tickets/00-RECAP.md
#
 # Usage:
 #   GPUS="4 5 6 7" scripts/accent_balance_fil50.sh             # all stages
 #   GPUS="3 4" PILOT=60 scripts/accent_balance_fil50.sh        # pilot: stages 0-3 + report, then stop
 #   STAGES="4 5 6" GPUS="7" scripts/accent_balance_fil50.sh    # resume
 #
 # Stage 3 (QA) transcriber knobs:
 #   QA_BACKEND=faster-whisper (default)  -> exactly today's flags: --model "$QA_MODEL"
 #   QA_BACKEND=faster-whisper-dual       -> --backend faster-whisper-dual
 #                                           --opt model_a="$QA_MODEL_A" --opt model_b="$QA_MODEL_B"
 #                                           (no --model). QA_MODEL_A defaults to
 #                                           models/faster-whisper-small and QA_MODEL_B to
 #                                           models/faster-whisper-small-tagalog; both are
 #                                           relative to $QA_REPO, which stage 3 cds into
 #                                           before invoking the transcriber.
 #   Any other QA_BACKEND value aborts the script (set -e + explicit exit).
 #
 # VOICE_SOURCES (stage 1, default unset -> today's 'all' pool, flag omitted):
 #   references -> plan from the references voices only, usable in EVERY split
 #                 (Phase 1); sapinsapin -> only each split's own fsc_ voices.
 #
 # Detach for long runs:  nohup scripts/accent_balance_fil50.sh > fil50.log 2>&1 & disown
 # Each stage writes $ROOT/.done/<stage>; a finished stage is skipped unless FORCE=1.
set -euo pipefail

cd "$(dirname "$0")/.."

UV=${UV:-$(command -v uv || echo /opt/uv/uv)}
STAGES=${STAGES:-"0 1 2 3 4 5 6"}
GPUS=${GPUS:-}
PILOT=${PILOT:-}
FORCE=${FORCE:-0}
SEED=${SEED:-0}
OVERGEN=${OVERGEN:-1.15}
VOICE_SOURCES=${VOICE_SOURCES:-}
QA_THRESHOLD=${QA_THRESHOLD:-0.80}
QA_MODEL=${QA_MODEL:-small}
QA_BACKEND=${QA_BACKEND:-faster-whisper}
QA_MODEL_A=${QA_MODEL_A:-models/faster-whisper-small}
QA_MODEL_B=${QA_MODEL_B:-models/faster-whisper-small-tagalog}
QA_REPO=${QA_REPO:-$HOME/simple-audio-transcriber}
VCM_MINUTES=${VCM_MINUTES:-60}
WAKEWORD_MINUTES=${WAKEWORD_MINUTES:-30}

if [ -n "$PILOT" ]; then
    # The trailing `true` matters under `set -e`: without it, the last command
    # run inside the $(...) is whichever `[ "$s" -le 3 ]` test happened to run
    # last (false for the final non-pilot stage), and that exit status becomes
    # this assignment's own exit status -- killing the whole script right here.
    STAGES=$(for s in $STAGES; do [ "$s" -le 3 ] && printf '%s ' "$s"; done; true)
    STAGES="$STAGES 3b"  # run_stage's membership check needs 3b explicitly listed, it isn't numeric
fi

ROOT=out/conversions/v2/fil50${PILOT:+-pilot$PILOT}
FSC_MANIFEST=out/conversions/v2/filipino_speech_corpus/manifest.csv
OLD_REFS_DIR=${REFS_DIR:-$HOME/cosy-voice-data/References}
VCM_BASE=out/conversions/v2/optionb-v3-vcmx/manifest.csv
WW_BASE=out/conversions/v2/wakeword/manifest.csv
VCM_OUT_MANIFEST=out/conversions/v2/optionb-v3-vcmx-fil50${PILOT:+-pilot$PILOT}/manifest.csv
WW_OUT_MANIFEST=out/conversions/v2/wakeword/manifest.fil50${PILOT:+-pilot$PILOT}.csv
VCM_RUN=out/vcm/option-d-fil50
VCM_OLD_RUN=out/vcm/option-d-dataset-v2
WW_RUN=out/wakeword-fil50
WW_OLD_RUN=out/wakeword

mkdir -p "$ROOT/.done" "$ROOT/logs"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

need_gpus() {
    [ -n "$GPUS" ] || { echo "GPUS must be set explicitly (check nvidia-smi for idle ones)" >&2; exit 1; }
}

first_gpu() { set -- $GPUS; echo "$1"; }
gpu_count() { set -- $GPUS; echo "$#"; }

# Stage 3 transcriber model flags, per QA_BACKEND (see header). Result goes
# in the global QA_FLAGS array, expanded quoted at the call site.
# faster-whisper (default): byte-identical to today's `--model "$QA_MODEL"`.
# faster-whisper-dual: two-model confidence backend; model paths are
# relative to $QA_REPO (stage3_qa cds into it before invoking).
qa_backend_flags() {
    case "$QA_BACKEND" in
        faster-whisper)
            QA_FLAGS=(--model "$QA_MODEL")
            ;;
        faster-whisper-dual)
            QA_FLAGS=(--backend faster-whisper-dual --opt "model_a=$QA_MODEL_A" --opt "model_b=$QA_MODEL_B")
            ;;
        *)
            echo "QA_BACKEND must be faster-whisper or faster-whisper-dual, got: $QA_BACKEND" >&2
            exit 1
            ;;
    esac
}

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

stage0_refs() {
    need_gpus  # whisper transcription of the References cuts benefits from a pinned GPU
    CUDA_VISIBLE_DEVICES=$(first_gpu) "$UV" run python -m me2_voicegen.accent_balance.build_refs \
        --fsc-manifest "$FSC_MANIFEST" --old-refs-dir "$OLD_REFS_DIR" \
        --out-dir "$ROOT/refs" --seed "$SEED"
}

stage1_plan() {
    "$UV" run python -m me2_voicegen.accent_balance.plan_jobs \
        --voices "$ROOT/refs/voices.csv" \
        --vcm-manifest "$VCM_BASE" --wakeword-manifest "$WW_BASE" \
        --overgen "$OVERGEN" --seed "$SEED" ${PILOT:+--pilot "$PILOT"} \
        ${VOICE_SOURCES:+--voice-sources "$VOICE_SOURCES"} \
        --out "$ROOT/jobs/jobs.csv"
}

# generate.py's own --out-dir contract is "$OUT" such that it writes
# $OUT/audio/<model>/<job_id>.wav and merges to $OUT/gen_manifest.csv -- pass
# $ROOT itself here, not $ROOT/audio (that would double-nest audio/audio/).
stage2_generate() {
    need_gpus
    local gpus=($GPUS) n=${#gpus[@]} i pids=()
    for i in "${!gpus[@]}"; do
        CUDA_VISIBLE_DEVICES=${gpus[$i]} nohup "$UV" run python -m me2_voicegen.accent_balance.generate \
            --jobs "$ROOT/jobs/jobs.csv" --out-dir "$ROOT" --refs-dir "$ROOT/refs" \
            --shard "$i/$n" --device cuda \
            > "$ROOT/logs/generate.shard$i.log" 2>&1 &
        pids+=($!)
    done
    local failed=0
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}" || { echo "generate shard $i failed, see $ROOT/logs/generate.shard$i.log" >&2; failed=1; }
    done
    [ "$failed" = 0 ]
    "$UV" run python -m me2_voicegen.accent_balance.generate --merge-shards --out-dir "$ROOT"
}

stage3_qa() {
    need_gpus
    qa_backend_flags
    "$UV" run python -m me2_voicegen.accent_balance.qa build-shim \
        --gen-manifest "$ROOT/gen_manifest.csv" --shim-dir "$ROOT/qa/shim"
    local subset
    for subset in "$ROOT"/qa/shim/*/; do
        (cd "$QA_REPO" && CUDA_VISIBLE_DEVICES=$(first_gpu) "$UV" run python -m audio_transcript_parser.qa \
            "$OLDPWD/$subset" --naming v1-pair "${QA_FLAGS[@]}" --device cuda \
            --threshold "$QA_THRESHOLD" --out-dir "$OLDPWD/$ROOT/qa/reports")
    done
    "$UV" run python -m me2_voicegen.accent_balance.qa parse \
        --gen-manifest "$ROOT/gen_manifest.csv" --shim-dir "$ROOT/qa/shim" \
        --reports-dir "$ROOT/qa/reports" --out "$ROOT/qa/qa_pass.csv"
}

stage3b_pilot_report() {
    "$UV" run python -m me2_voicegen.accent_balance.pilot_report \
        --root "$ROOT" --vcm-manifest "$VCM_BASE" --wakeword-manifest "$WW_BASE" \
        --overgen "$OVERGEN" --gpus "$(gpu_count)" --seed "$SEED"
    log "pilot report: $ROOT/pilot_report.md (listen dir: $ROOT/listen/)"
}

stage4_collate() {
    "$UV" run python -m me2_voicegen.accent_balance.collate \
        --gen-manifest "$ROOT/gen_manifest.csv" --qa-pass "$ROOT/qa/qa_pass.csv" \
        --vcm-manifest "$VCM_BASE" --wakeword-manifest "$WW_BASE" \
        --noise-root out/conversions/v2/background_noise --seed "$SEED" \
        --vcm-out "$VCM_OUT_MANIFEST" --wakeword-out "$WW_OUT_MANIFEST"
}

stage5_train() {
    need_gpus
    local g; g=$(first_gpu)
    make optionb-train OPTIONB_MANIFEST="$VCM_OUT_MANIFEST" OPTIONB_OUT_DIR="$VCM_RUN" \
        VCM_PRESET=optiond VCM_DEVICE=cuda:"$g" VCM_MAX_MINUTES="$VCM_MINUTES" VCM_SEED="$SEED" \
        > "$ROOT/logs/train_vcm.log" 2>&1 &
    local vcm_pid=$!
    local g2; g2=$(set -- $GPUS; echo "${2:-$1}")
    make wakeword-train WAKEWORD_TRAIN_MANIFEST="$WW_OUT_MANIFEST" WAKEWORD_TRAIN_OUT_DIR="$WW_RUN" \
        WAKEWORD_DEVICE=cuda:"$g2" WAKEWORD_MAX_MINUTES="$WAKEWORD_MINUTES" WAKEWORD_TRAIN_SEED="$SEED" \
        > "$ROOT/logs/train_wakeword.log" 2>&1
    wait "$vcm_pid"
}

stage6_eval() {
    need_gpus
    local g; g=$(first_gpu)
    make optionb-eval OPTIONB_MANIFEST="$VCM_OUT_MANIFEST" OPTIONB_OUT_DIR="$VCM_RUN" \
        VCM_DEVICE=cuda:"$g" > "$ROOT/logs/eval_vcm_new.log" 2>&1
    # Old checkpoints on the new manifest: same-test before/after, written to
    # separate out-dirs so the committed option-d-dataset-v2/out/wakeword reports
    # stay untouched.
    "$UV" run python -m me2_voicegen.vcm.evaluate --manifest "$VCM_OUT_MANIFEST" \
        --checkpoint "$VCM_OLD_RUN/checkpoints/checkpoint.pt" --out-dir "$VCM_RUN/baseline_old_ckpt" \
        --device cuda:"$g" --beam-width 50 --grammar optionb \
        > "$ROOT/logs/eval_vcm_old.log" 2>&1
    "$UV" run python -m me2_voicegen.wakeword.train --manifest "$WW_OUT_MANIFEST" \
        --out-dir "$WW_RUN/baseline_old_ckpt" --eval-only --checkpoint "$WW_OLD_RUN/checkpoints/checkpoint.pt" \
        --device cuda:"$g" \
        > "$ROOT/logs/eval_wakeword_old.log" 2>&1
    make wakeword-export WAKEWORD_TRAIN_MANIFEST="$WW_OUT_MANIFEST" WAKEWORD_TRAIN_OUT_DIR="$WW_RUN" \
        > "$ROOT/logs/export_wakeword.log" 2>&1
    make wakeword-bench WAKEWORD_TRAIN_MANIFEST="$WW_OUT_MANIFEST" WAKEWORD_TRAIN_OUT_DIR="$WW_RUN" \
        > "$ROOT/logs/bench_wakeword.log" 2>&1
    make vcmx-export VCMX_TRAINED_OUT_DIR="$VCM_RUN" VCMX_SERVE_MANIFEST="$VCM_OUT_MANIFEST" \
        > "$ROOT/logs/export_vcm.log" 2>&1
}

run_stage 0 refs     stage0_refs
run_stage 1 plan     stage1_plan
run_stage 2 generate stage2_generate
run_stage 3 qa       stage3_qa
if [ -n "$PILOT" ]; then
    run_stage 3b pilot_report stage3b_pilot_report
    log "pilot complete, stopping (PILOT=$PILOT) -- review $ROOT/pilot_report.md before running the full pipeline"
    exit 0
fi
run_stage 4 collate  stage4_collate
run_stage 5 train    stage5_train
run_stage 6 eval     stage6_eval
log "all requested stages finished: $STAGES"
