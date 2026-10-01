UV ?= $(shell command -v uv 2>/dev/null || echo /opt/uv/uv)
export UV_PYTHON := /usr/bin/python3.10
UV_CACHE_DIR ?= /tmp/uv-cache-$(USER)
export UV_CACHE_DIR

VENDOR_DIR := vendor/CosyVoice
COSYVOICE_SHA := 074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc

BACKEND ?= cosyvoice2
TEXT ?=
MANIFEST ?= personas.json
RAW_DIR ?= $(HOME)/cosy-voice-data/Raw_Audio
REFS_DIR ?= $(HOME)/cosy-voice-data/References
RESYNTH_PROB ?= 0.3
SEED ?=

VCM_MANIFEST ?= out/conversions/v2/test_set/manifest.csv
VCM_OUT_DIR ?= out/vcm
VCM_PRESET ?= default
VCM_DEVICE ?= cpu
VCM_MAX_MINUTES ?= 30
VCM_SEED ?= 0

VCM_CHECKPOINT ?= $(VCM_OUT_DIR)/checkpoints/checkpoint.pt
VCM_BENCH_CALIBRATION_SAMPLES ?= 32
VCM_BENCH_N_ITERS ?= 100

VCM_EVAL_BEAM_WIDTH ?= 50
VCM_SLOT_EVAL_MANIFEST ?= $(VCM_OUT_DIR)/metadata/slot_eval/manifest.csv
VCM_GRAMMAR ?= spec,toy
VCM_EVAL_REQUIRED_COMMAND_MARGIN ?=
# Empty = evaluate.py default (mean_frame). "per_char" = dense-d2-loose-impl.
VCM_EVAL_SCORE_MODE ?=

OPTIONB_MANIFEST ?= out/conversions/v2/optionb/manifest.csv
OPTIONB_OUT_DIR ?= out/vcm/optionb

# Live-upstream Option B refresh + VCM Dataset B merge (feature
# optionb-v3-vcmx). `optionb-refresh` re-fetches Option B from upstream
# into OPTIONB_V3_OUT_ROOT, distinct from OPTIONB_MANIFEST's own pinned
# out/conversions/v2/optionb/ baseline above (never overwritten by this).
# `vcmx-build` merges that refreshed Option B with VCM Dataset B (balanced)
# and the fold-rebuilt ESC-50 silence pool into one train/val/test pool
# (`vcm.vcmx_merge`, .scratch/optionb-v3-vcmx/tickets/00-RECAP.md Phase 2).
# No new train/eval targets: point optionb-train/optionb-eval at
# VCMX_TREATMENT or VCMX_CONTROL via OPTIONB_MANIFEST=/OPTIONB_OUT_DIR=.
OPTIONB_V3_OUT_ROOT ?= out/conversions/v2/optionb-v3
VCMX_B_ROOT ?= raw_datasets/VCM_BALANCED
VCMX_B_METADATA_ROOT ?= raw_datasets/VCM_BALANCED_METADATA
VCMX_A_METADATA_ROOT ?= raw_datasets/VCM_MASTER_METADATA
VCMX_ESC50_MANIFEST ?= out/conversions/v2/background_noise/manifest.csv
VCMX_TREATMENT ?= out/conversions/v2/optionb-v3-vcmx
VCMX_CONTROL ?= out/conversions/v2/optionb-v3-vcmx-control
VCMX_SEED ?= 0

# Serving the trained treatment checkpoint (out/vcm/option-d-dataset-v2 --
# optiond preset, trained on VCMX_TREATMENT; see
# out/vcm/option-d-dataset-v2-SUMMARY.md for why this is the one to ship).
# `vcmx-export` produces its INT8 ONNX export (not generated during
# train/eval); `vcmx-serve` runs it through vcm.streaming with that export
# plus this run's own calibrated operating threshold (-0.1, its
# eval_report.md) and a full-grammar incomplete-prefix margin (=4.0) chosen
# by `vcm.incomplete_calibration select --split val`, regression-budget
# check scored against real-recording (source_dataset=="vcm_balanced")
# target rows only -- out/conversions/v2/optionb-v3-vcmx/
# manifest.realtarget.csv, the same VCMX_TREATMENT manifest with its ~90%
# voice-cloned Option B target-command rows stripped out (babble/silence
# rows, already mostly real, are unchanged) -- and confirmed once on the
# disjoint `holdout --split test` split, same filter. Reports at
# out/vcm/option-d-dataset-v2/metadata/incomplete_calibration_beam50_realtarget/
# {val_report,test_report}.md. Held-out (real commands only): FAR
# 11.06%->6.73%, target exact accuracy 88.37%->86.82% (1.55pp), gate-caused
# FRR 1.75%. Supersedes margin=6.0, which the *pooled* (90% synthetic)
# calibration picked -- checked against real commands alone, margin=6.0's
# val accuracy drop is 1.54pp, already over the 1pp regression budget, so it
# was not actually safe for real speech despite passing the pooled check.
# margin=6.0 does reach a slightly lower FAR (6.46% vs 6.73% held-out) but
# not by enough to justify the extra real-command risk. Both margins were
# measured at beam-width 50 -- VCMX_SERVE_BEAM_WIDTH must stay 50 (not
# vcm.streaming's own default of 25) or the calibration doesn't transfer.
# Remaining caveat: the incomplete-prefix probes themselves are still cut
# from Option B clips only (source_dataset=="optionb") -- Dataset B's real
# recordings don't cover enough of the designated-prefix vocabulary to
# generate a real-only probe set from (see incomplete_probes.py's
# per-prefix speaker counts), so the FAR side of this number, unlike the
# accuracy side, is still a synthetic-audio measurement.
VCMX_TRAINED_OUT_DIR ?= out/vcm/option-d-dataset-v2
VCMX_SERVE_MANIFEST ?= $(VCMX_TREATMENT)/manifest.csv
VCMX_SERVE_VARIANT ?= int8
VCMX_SERVE_THRESHOLD ?= -0.1
VCMX_SERVE_MARGIN ?= 4.0
# "mean_frame" (default) or "per_char" (dense-d2-loose-impl; needs its own
# VCMX_SERVE_THRESHOLD ~ -1.2 and VCMX_SERVE_MARGIN= (empty) -- not the
# production setting until its validation gates pass, see
# .scratch/dense-d2-loose-impl/PLAN.md).
VCMX_SERVE_SCORE_MODE ?= mean_frame
VCMX_SERVE_BEAM_WIDTH ?= 50
VCMX_SERVE_GATE ?= none
VCMX_SERVE_SOURCE ?= mic
VCMX_SERVE_ARGS ?=

STREAM_MODEL ?= optionc
STREAM_BACKEND ?= onnx
STREAM_SOURCE ?= mic
STREAM_ARGS ?=
# Feature `ui-site`: app-pipeline's audio source (live mic by default;
# override with a wav path for a deterministic file-replay run).
APP_VENV_BIN ?= .venv/bin
APP_PIPELINE_SOURCE ?= mic
# Run dir `app-pipeline` serves. Default is the production optiond run, which
# is also the original hardcoded value, so `make -n app-pipeline` output is
# byte-identical to before the variable existed (quartznet-promotion ticket 01).
APP_PIPELINE_MODEL ?= out/vcm/option-d-fil50-ambient-rir-135m
# Defaults are the original hardcoded app-pipeline values (mean_frame scoring).
APP_PIPELINE_THRESHOLD ?= -0.075
APP_PIPELINE_MARGIN ?= 4.0
APP_PIPELINE_SCORE_MODE ?= mean_frame
# Opt-in trial threshold for `make app-pipeline-perchar` (per-character scale, ~-1; see
# docs/DENSE-SCORING-DECISION.md for the safe range).
APP_PERCHAR_THRESHOLD ?= -1.204
# `app-pipeline-quartznet` operating point. PLACEHOLDERS until
# quartznet-promotion ticket 02 fills them from the recalibration (see
# docs/QUARTZNET-OPERATING-POINT.md once it lands); until then they carry the
# seed-0 clean-val choice (-0.1) and optiond's margin (4.0).
APP_QUARTZNET_THRESHOLD ?= -0.1
APP_QUARTZNET_MARGIN ?= 4.0

.PHONY: sync vendor download-model generate generate-personas generate-conversions test clean vcm-train vcm-bench vcm-eval optionb-train optionb-eval optionb-refresh vcmx-build vcmx-export vcmx-serve vcmx-serve-wakeword stream app-pipeline-perchar app-pipeline-quartznet quartznet-eval-report wakeword-fetch-positives wakeword-fetch-positives-dry-run wakeword-convert-positives wakeword-convert-positives-dry-run wakeword-generate-silence wakeword-generate-silence-dry-run wakeword-generate-adversaries wakeword-generate-adversaries-dry-run wakeword-mix-noise wakeword-mix-noise-dry-run wakeword-consolidate-manifest wakeword-build-unknown-external wakeword-build-unknown-external-dry-run wakeword-build-dataset wakeword-derive-spans wakeword-train wakeword-export wakeword-bench soak-tally app app-pipeline

WAKEWORD_CLONE_DIR ?= $(if $(TMPDIR),$(TMPDIR),/tmp)/me2_wakeword_clone
WAKEWORD_POSITIVES_OUT_DIR ?= out/conversions/v2/wakeword/positives_real
WAKEWORD_CONVERTED_OUT_DIR ?= out/conversions/v2/wakeword/positives_converted
WAKEWORD_REFS_DIR ?= $(REFS_DIR)
WAKEWORD_K ?= 8
WAKEWORD_SEED ?= 0
WAKEWORD_MAX_CONVERSIONS ?=
WAKEWORD_SILENCE_OUT_DIR ?= out/conversions/v2/wakeword/silence_synthetic
WAKEWORD_SILENCE_SEED ?= 0
WAKEWORD_SILENCE_TOTAL_COUNT ?= 1806
WAKEWORD_ADVERSARIES_REFS_DIR ?= $(REFS_DIR)
WAKEWORD_ADVERSARIES_OUT_DIR ?= out/conversions/v2/wakeword/adversaries
WAKEWORD_ROOT ?= out/conversions/v2/wakeword
WAKEWORD_NOISE_ROOT ?= out/conversions/v2/background_noise
WAKEWORD_NOISE_PROB ?= 0.35
WAKEWORD_NOISE_SEED ?= 0
WAKEWORD_COMMON_VOICE_ROOT ?= out/conversions/v2/common_voice_negative
WAKEWORD_UNKNOWN_EXTERNAL_OUT_DIR ?= out/conversions/v2/wakeword/common_voice_negative_sample
WAKEWORD_UNKNOWN_TARGET_COUNT ?= 5172
WAKEWORD_UNKNOWN_SEED ?= 0
WAKEWORD_BUILD_SEED ?= 0

WAKEWORD_TRAIN_MANIFEST ?= out/conversions/v2/wakeword/manifest.csv
WAKEWORD_TRAIN_OUT_DIR ?= out/wakeword
WAKEWORD_PRESET ?= default
WAKEWORD_DEVICE ?= cpu
WAKEWORD_MAX_MINUTES ?= 30
WAKEWORD_TRAIN_SEED ?= 0
WAKEWORD_NOISE_ROOT_TRAIN ?= out/conversions/v2/background_noise
WAKEWORD_CHECKPOINT ?= $(WAKEWORD_TRAIN_OUT_DIR)/checkpoints/checkpoint.pt
WAKEWORD_BENCH_CALIBRATION_SAMPLES ?= 32
WAKEWORD_BENCH_N_ITERS ?= 100

sync:
	$(UV) sync

vendor:
	@if [ -d "$(VENDOR_DIR)/.git" ] && [ "$$(cd $(VENDOR_DIR) && git rev-parse HEAD)" = "$(COSYVOICE_SHA)" ]; then \
		echo "vendor/CosyVoice already at pinned SHA $(COSYVOICE_SHA), skipping clone"; \
	else \
		rm -rf $(VENDOR_DIR); \
		mkdir -p vendor; \
		git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git $(VENDOR_DIR); \
		cd $(VENDOR_DIR) && git checkout $(COSYVOICE_SHA) && git submodule update --init --recursive; \
	fi

download-model:
	$(UV) run python -m me2_voicegen.generation.download_model

generate:
	$(UV) run python -m me2_voicegen.generation.generate_sample --backend $(BACKEND) --text "$(TEXT)"

generate-personas:
	$(UV) run python -m me2_voicegen.generation.generate_personas --manifest "$(MANIFEST)" --backend $(BACKEND) --text "$(TEXT)"

generate-conversions:
	$(UV) run python -m me2_voicegen.generation.generate_conversions --raw-dir "$(RAW_DIR)" --refs-dir "$(REFS_DIR)" --backend $(BACKEND) --resynth-prob $(RESYNTH_PROB) $(if $(SEED),--seed $(SEED))

vcm-train:
	$(UV) run python -m me2_voicegen.vcm.train --manifest "$(VCM_MANIFEST)" --out-dir "$(VCM_OUT_DIR)" --preset $(VCM_PRESET) --device $(VCM_DEVICE) --max-minutes $(VCM_MAX_MINUTES) --seed $(VCM_SEED)

vcm-bench:
	$(UV) run python -m me2_voicegen.vcm.benchmark --checkpoint "$(VCM_CHECKPOINT)" --out-dir "$(VCM_OUT_DIR)" --manifest "$(VCM_MANIFEST)" --calibration-samples $(VCM_BENCH_CALIBRATION_SAMPLES) --n-iters $(VCM_BENCH_N_ITERS)

vcm-eval:
	$(UV) run python -m me2_voicegen.vcm.evaluate --manifest "$(VCM_MANIFEST)" --checkpoint "$(VCM_CHECKPOINT)" --out-dir "$(VCM_OUT_DIR)" --device $(VCM_DEVICE) --beam-width $(VCM_EVAL_BEAM_WIDTH) --slot-eval-manifest "$(VCM_SLOT_EVAL_MANIFEST)" --grammar $(VCM_GRAMMAR) $(if $(VCM_EVAL_REQUIRED_COMMAND_MARGIN),--required-command-margin $(VCM_EVAL_REQUIRED_COMMAND_MARGIN),) $(if $(VCM_EVAL_SCORE_MODE),--score-mode $(VCM_EVAL_SCORE_MODE),)

# Option B convenience targets: same vcm-train/vcm-eval machinery, pointed
# at out/conversions/v2/optionb/manifest.csv (D6c) and OPTIONB_GRAMMAR
# (D11, .scratch/optionb-dataset/tickets/04-pipeline-wiring.md), so the
# documented invocation is one command instead of a long VCM_MANIFEST=...
# override. `optionb-eval` intentionally has no --slot-eval-manifest
# default override -- Task 07's slot-eval-set is VCM-specific and does not
# exist for Option B; evaluate.py skips it gracefully when absent.
optionb-train:
	$(UV) run python -m me2_voicegen.vcm.train --manifest "$(OPTIONB_MANIFEST)" --out-dir "$(OPTIONB_OUT_DIR)" --preset $(VCM_PRESET) --device $(VCM_DEVICE) --max-minutes $(VCM_MAX_MINUTES) --seed $(VCM_SEED)

optionb-eval:
	$(UV) run python -m me2_voicegen.vcm.evaluate --manifest "$(OPTIONB_MANIFEST)" --checkpoint "$(OPTIONB_OUT_DIR)/checkpoints/checkpoint.pt" --out-dir "$(OPTIONB_OUT_DIR)" --device $(VCM_DEVICE) --beam-width $(VCM_EVAL_BEAM_WIDTH) --slot-eval-manifest "$(OPTIONB_OUT_DIR)/metadata/slot_eval/manifest.csv" --grammar optionb $(if $(VCM_EVAL_REQUIRED_COMMAND_MARGIN),--required-command-margin $(VCM_EVAL_REQUIRED_COMMAND_MARGIN),) $(if $(VCM_EVAL_SCORE_MODE),--score-mode $(VCM_EVAL_SCORE_MODE),)

# Re-fetch Option B from the live upstream repo (docs/OPTIONB-GRAMMAR-CONTRACT.md
# §6) into OPTIONB_V3_OUT_ROOT -- out/conversions/v2/optionb/ (OPTIONB_MANIFEST's
# own dir) is never touched by this.
optionb-refresh:
	$(UV) run python -m me2_voicegen.vcm.optionb.fetch_dataset --force-clone --out-root "$(OPTIONB_V3_OUT_ROOT)"

# Merge VCM Dataset B (balanced) + the fold-rebuilt ESC-50 silence pool
# into OPTIONB_V3_OUT_ROOT, producing VCMX_TREATMENT and VCMX_CONTROL.
vcmx-build:
	$(UV) run python -m me2_voicegen.vcm.vcmx_merge build --optionb-manifest "$(OPTIONB_V3_OUT_ROOT)/manifest.csv" --b-root "$(VCMX_B_ROOT)" --b-metadata-root "$(VCMX_B_METADATA_ROOT)" --a-metadata-root "$(VCMX_A_METADATA_ROOT)" --esc50-manifest "$(VCMX_ESC50_MANIFEST)" --out-treatment "$(VCMX_TREATMENT)" --out-control "$(VCMX_CONTROL)" --seed $(VCMX_SEED)

# fp32+INT8 ONNX export for VCMX_TRAINED_OUT_DIR's checkpoint -- vcm.benchmark
# does the export as a side effect (same pattern as vcm-bench above), writing
# to $(VCMX_TRAINED_OUT_DIR)/export/. Run once before vcmx-serve's first use
# with --backend onnx (the export isn't produced by optionb-train/-eval).
vcmx-export:
	$(UV) run python -m me2_voicegen.vcm.benchmark --checkpoint "$(VCMX_TRAINED_OUT_DIR)/checkpoints/checkpoint.pt" --out-dir "$(VCMX_TRAINED_OUT_DIR)" --manifest "$(VCMX_SERVE_MANIFEST)" --calibration-samples $(VCM_BENCH_CALIBRATION_SAMPLES) --n-iters $(VCM_BENCH_N_ITERS)

# Serve VCMX_TRAINED_OUT_DIR's INT8 export through vcm.streaming with its
# calibrated threshold/margin/beam-width baked in as defaults (see the
# VCMX_SERVE_* block above for why each value is what it is). Requires
# vcmx-export to have run at least once. VCMX_SERVE_SOURCE defaults to the
# live microphone; override with a wav path for deterministic file-replay.
# VCMX_SERVE_GATE defaults to "none" -- this is the non-gated recipe (the
# mic is always listening, no wakeword/spacebar in front of it). See
# vcmx-serve-wakeword below for the gated counterpart.
vcmx-serve:
	$(UV) run python -m me2_voicegen.vcm.streaming --model "$(VCMX_TRAINED_OUT_DIR)" --backend onnx --onnx-variant $(VCMX_SERVE_VARIANT) --grammar optionb --threshold $(VCMX_SERVE_THRESHOLD) $(if $(VCMX_SERVE_MARGIN),--required-command-margin $(VCMX_SERVE_MARGIN),) --score-mode $(VCMX_SERVE_SCORE_MODE) --beam-width $(VCMX_SERVE_BEAM_WIDTH) --gate $(VCMX_SERVE_GATE) --source "$(VCMX_SERVE_SOURCE)" $(VCMX_SERVE_ARGS)

# Same VCM model/margin/threshold as vcmx-serve, but fronted by the trained
# wakeword DS-CNN (out/wakeword, feature wakeword-gate) in place of an
# always-on mic -- mirrors stream-wakeword's pattern (--policy single_period
# --gate wakeword --gate-period 3, ticket 04 of feature optionb-streaming).
# --wakeword-backend onnx uses out/wakeword/export/wakeword_model.*.onnx
# (make wakeword-export regenerates it); pass
# VCMX_SERVE_ARGS="--wakeword-backend torch ..." to use the checkpoint
# directly instead. Wakeword's own open threshold stays at
# gate.DEFAULT_WAKEWORD_THRESHOLD (0.9) unless overridden with
# --wakeword-threshold.
vcmx-serve-wakeword:
	$(MAKE) vcmx-serve VCMX_SERVE_GATE=wakeword VCMX_SERVE_ARGS="--policy single_period --gate-period 3 --wakeword-backend onnx --log-periods $(VCMX_SERVE_ARGS)"

# Streaming spoken-command runner (ticket 04 of feature optionb-streaming).
# STREAM_SOURCE defaults to the live microphone; override with a wav path
# (STREAM_SOURCE=path/to/clip.wav) for deterministic file-replay runs.
# STREAM_ARGS carries any extra flags (e.g. STREAM_ARGS="--listen-for 10
# --log-all-windows") straight through to the CLI.
stream:
	$(UV) run python -m me2_voicegen.vcm.streaming --model $(STREAM_MODEL) --backend $(STREAM_BACKEND) --source "$(STREAM_SOURCE)" $(STREAM_ARGS)

stream-single-period:
	$(MAKE) stream STREAM_ARGS="--policy single_period --gate spacebar --gate-period 3 --threshold -0.132222 --log-periods"

# Feature `wakeword-gate` (feature-engineering/wakeword-gate/SPEC.md): the
# trained wakeword DS-CNN in place of the spacebar press. Requires
# `make wakeword-export` to have been run first for --wakeword-backend onnx
# (the default here); pass STREAM_ARGS="--wakeword-backend torch ..." to use
# the checkpoint directly instead.
stream-wakeword:
	$(MAKE) stream STREAM_ARGS="--policy single_period --gate wakeword --gate-period 3 --threshold -0.132222 --wakeword-backend onnx --log-periods"

# Feature `ui-site` (feature-engineering/ui-site/SPEC.md): the FastAPI
# dashboard, a *separate service* from the voice pipeline (its own process
# / terminal window). The model side reaches it one-way over its HTTP API:
# terminal 1 runs a streaming recipe piped into `python -m app.forward`,
# terminal 2 runs this. See the README "UI site" section.
app:
	$(APP_VENV_BIN)/uvicorn app.main:app --host 127.0.0.1 --port 8000

# Feature `ui-site`: the live model-serving pipeline that feeds the
# dashboard (terminal 1 of the two-window setup; terminal 2 runs `make
# app`). The Option D (fil50) VCM int8 export fronted by the trained
# `out/wakeword-sesame` DS-CNN (one 3 s listening period per wakeword
# detection), piped one-way into the forwarder. Period digest/banner go
# to stderr, so stdout stays clean JSONL for the pipe. Override
# APP_PIPELINE_SOURCE with a wav path for a file-replay run.
app-pipeline:
	$(APP_VENV_BIN)/python -m me2_voicegen.vcm.streaming --model $(APP_PIPELINE_MODEL) --backend onnx --onnx-variant int8 --grammar optionb --threshold $(APP_PIPELINE_THRESHOLD) $(if $(APP_PIPELINE_MARGIN),--required-command-margin $(APP_PIPELINE_MARGIN),) --score-mode $(APP_PIPELINE_SCORE_MODE) --beam-width 50 --gate wakeword --policy single_period --gate-period 3 --wakeword-model out/wakeword-sesame --wakeword-backend onnx --log-periods --source "$(APP_PIPELINE_SOURCE)" | $(APP_VENV_BIN)/python -m app.forward

# OPT-IN TRIAL (dense-d2-loose-impl): the same pipeline scored per character
# instead of per frame, no incomplete-prefix margin gate. NOT the production
# setting -- see docs/DENSE-SCORING-DECISION.md for the evidence, caveats, and
# what to report back. Rollback = use `make app-pipeline` (unchanged defaults).
app-pipeline-perchar:
	$(MAKE) app-pipeline APP_PIPELINE_SCORE_MODE=per_char APP_PIPELINE_THRESHOLD=$(APP_PERCHAR_THRESHOLD) APP_PIPELINE_MARGIN=

# quartznet-promotion ticket 01: the same pipeline serving the stride-2
# QuartzNet-5x3-tiny INT8 export. APP_QUARTZNET_THRESHOLD/APP_QUARTZNET_MARGIN
# are placeholders until ticket 02's recalibration lands (see
# docs/QUARTZNET-OPERATING-POINT.md); the default `make app-pipeline` above is
# unchanged. Run `make quartznet-eval-report` first so the run dir's
# metadata/eval_report.json exists (see that target).
app-pipeline-quartznet:
	$(MAKE) app-pipeline APP_PIPELINE_MODEL=out/vcm/quartznet5x3-s2-fil50-ambient-rir-135m APP_PIPELINE_THRESHOLD=$(APP_QUARTZNET_THRESHOLD) APP_PIPELINE_MARGIN=$(APP_QUARTZNET_MARGIN)

# quartznet-promotion ticket 01: materialize
# out/vcm/quartznet5x3-s2-fil50-ambient-rir-135m/metadata/eval_report.json --
# the layout the streaming CLI's resolve_threshold() reads (see
# docs/STREAMING-CONTRACT.md section 2). The noisy-gate copy already under
# <run_dir>/noisy_eval/ is a subdir and is NOT what gets read; without this
# artifact a QuartzNet serve with no explicit --threshold would fall back to
# FALLBACK_THRESHOLD (0.0, accepts almost nothing). ~45-60 min on a GPU; use
# a free non-6 GPU (VCM_DEVICE=cuda:N).
quartznet-eval-report:
	$(UV) run python -m me2_voicegen.vcm.evaluate --manifest out/conversions/v2/optionb-v3-vcmx-fil50/manifest.csv --checkpoint out/vcm/quartznet5x3-s2-fil50-ambient-rir-135m/checkpoints/checkpoint.pt --out-dir out/vcm/quartznet5x3-s2-fil50-ambient-rir-135m --device $(VCM_DEVICE) --beam-width 50 --grammar optionb --noisy-eval-seed 0

# Wakeword ("computer"-only) dataset-build targets. See
# docs/WAKEWORD-DATASET-CONTRACT.md for the manifest schema/output layout
# `fetch_positives.py` (ticket 01) produces; later tickets (02/03/04/05)
# add their own subset-fetch/convert/assemble targets alongside these.
#
# `?=` above does not repopulate a variable that is defined-but-empty (e.g.
# `make wakeword-fetch-positives WAKEWORD_POSITIVES_OUT_DIR=`), which would
# otherwise silently pass `--out-root ""` through to a tool that can rmtree
# that path (security-auditor finding R2-1) -- `fetch_positives.py` itself
# now refuses an empty/`.`-resolving --out-root/--clone-dir, but these two
# checks below fail the `make` invocation itself, before even launching
# python, as a second independent guard on the exact variables this
# Makefile exposes as overridable.
wakeword-fetch-positives:
	@test -n "$(strip $(WAKEWORD_CLONE_DIR))" || { echo "WAKEWORD_CLONE_DIR must not be empty" >&2; exit 1; }
	@test -n "$(strip $(WAKEWORD_POSITIVES_OUT_DIR))" || { echo "WAKEWORD_POSITIVES_OUT_DIR must not be empty" >&2; exit 1; }
	$(UV) run python -m me2_voicegen.wakeword.fetch_positives --clone-dir "$(WAKEWORD_CLONE_DIR)" --out-root "$(WAKEWORD_POSITIVES_OUT_DIR)"

wakeword-fetch-positives-dry-run:
	@test -n "$(strip $(WAKEWORD_CLONE_DIR))" || { echo "WAKEWORD_CLONE_DIR must not be empty" >&2; exit 1; }
	@test -n "$(strip $(WAKEWORD_POSITIVES_OUT_DIR))" || { echo "WAKEWORD_POSITIVES_OUT_DIR must not be empty" >&2; exit 1; }
	$(UV) run python -m me2_voicegen.wakeword.fetch_positives --clone-dir "$(WAKEWORD_CLONE_DIR)" --out-root "$(WAKEWORD_POSITIVES_OUT_DIR)" --dry-run

# Ticket 03: K=8 stratified voice-conversion of ticket 01's real positives.
# --seed is required by convert_positives.py itself (fails loudly if
# unset); WAKEWORD_SEED's own default here just keeps `make` runnable
# out of the box, same posture as VCM_SEED above. WAKEWORD_MAX_CONVERSIONS
# is empty by default (no cap, full run); set it to pilot/chunk real work,
# e.g. `make wakeword-convert-positives WAKEWORD_MAX_CONVERSIONS=20`.
wakeword-convert-positives:
	@test -n "$(strip $(WAKEWORD_CONVERTED_OUT_DIR))" || { echo "WAKEWORD_CONVERTED_OUT_DIR must not be empty" >&2; exit 1; }
	$(UV) run python -m me2_voicegen.wakeword.convert_positives --positives-manifest "$(WAKEWORD_POSITIVES_OUT_DIR)/manifest.csv" --refs-dir "$(WAKEWORD_REFS_DIR)" --out-root "$(WAKEWORD_CONVERTED_OUT_DIR)" --k $(WAKEWORD_K) --seed $(WAKEWORD_SEED) $(if $(WAKEWORD_MAX_CONVERSIONS),--max-conversions $(WAKEWORD_MAX_CONVERSIONS))

wakeword-convert-positives-dry-run:
	@test -n "$(strip $(WAKEWORD_CONVERTED_OUT_DIR))" || { echo "WAKEWORD_CONVERTED_OUT_DIR must not be empty" >&2; exit 1; }
	$(UV) run python -m me2_voicegen.wakeword.convert_positives --positives-manifest "$(WAKEWORD_POSITIVES_OUT_DIR)/manifest.csv" --refs-dir "$(WAKEWORD_REFS_DIR)" --out-root "$(WAKEWORD_CONVERTED_OUT_DIR)" --k $(WAKEWORD_K) --seed $(WAKEWORD_SEED) --dry-run

# `generate_silence.py` (ticket 05) -- synthetic Gaussian/pink/brown
# colored-noise `_silence_` chunks. Fully independent of the
# wakeword-fetch-positives targets above (no shared clone dir, no network).
wakeword-generate-silence:
	$(UV) run python -m me2_voicegen.wakeword.generate_silence --out-root "$(WAKEWORD_SILENCE_OUT_DIR)" --seed $(WAKEWORD_SILENCE_SEED) --total-count $(WAKEWORD_SILENCE_TOTAL_COUNT)

wakeword-generate-silence-dry-run:
	$(UV) run python -m me2_voicegen.wakeword.generate_silence --out-root "$(WAKEWORD_SILENCE_OUT_DIR)" --seed $(WAKEWORD_SILENCE_SEED) --total-count $(WAKEWORD_SILENCE_TOTAL_COUNT) --dry-run

# `generate_adversaries.py` (ticket 02) -- phonetic near-miss TTS renders of
# "computer" for the `_unknown_` class, across WAKEWORD_ADVERSARIES_REFS_DIR
# (defaults to the same reference-voice pool generate-conversions already
# uses, REFS_DIR). Independent of the fetch-positives/generate-silence
# targets above (no shared clone dir, no network -- local TTS only).
wakeword-generate-adversaries:
	$(UV) run python -m me2_voicegen.wakeword.generate_adversaries --refs-dir "$(WAKEWORD_ADVERSARIES_REFS_DIR)" --out-root "$(WAKEWORD_ADVERSARIES_OUT_DIR)" --backend $(BACKEND)

wakeword-generate-adversaries-dry-run:
	$(UV) run python -m me2_voicegen.wakeword.generate_adversaries --refs-dir "$(WAKEWORD_ADVERSARIES_REFS_DIR)" --out-root "$(WAKEWORD_ADVERSARIES_OUT_DIR)" --backend $(BACKEND) --dry-run

# `mix_background_noise.py` -- probabilistically mixes background_noise
# (ESC-50, CC-BY-NC-SA-4.0) into adversaries/positives_converted, additively,
# writing new `*_noisy` sibling subsets (originals untouched). This is a
# deliberate, documented licensing-encumbrance decision -- see
# docs/WAKEWORD-DATASET-CONTRACT.md section 7 before running it.
wakeword-mix-noise:
	$(UV) run python -m me2_voicegen.wakeword.mix_background_noise --wakeword-root "$(WAKEWORD_ROOT)" --noise-root "$(WAKEWORD_NOISE_ROOT)" --noise-prob $(WAKEWORD_NOISE_PROB) --seed $(WAKEWORD_NOISE_SEED)

wakeword-mix-noise-dry-run:
	$(UV) run python -m me2_voicegen.wakeword.mix_background_noise --wakeword-root "$(WAKEWORD_ROOT)" --noise-root "$(WAKEWORD_NOISE_ROOT)" --noise-prob $(WAKEWORD_NOISE_PROB) --seed $(WAKEWORD_NOISE_SEED) --dry-run

# `consolidate_manifest.py` -- flat merge of every subset generated so far
# into out/conversions/v2/wakeword/generated_manifest.csv. NOT ticket 04's
# final assembled dataset (no external corpora, no split assignment).
wakeword-consolidate-manifest:
	$(UV) run python -m me2_voicegen.wakeword.consolidate_manifest --wakeword-root "$(WAKEWORD_ROOT)"

# `build_unknown_external.py` -- scrubs common_voice_negative (CC0) for a
# "computer" token, seeded-samples the remainder to WAKEWORD_UNKNOWN_TARGET_COUNT,
# and writes common_voice_negative_sample/ (broadens `_unknown_` beyond
# adversaries' phonetic near-misses; no license encumbrance).
wakeword-build-unknown-external:
	$(UV) run python -m me2_voicegen.wakeword.build_unknown_external --common-voice-root "$(WAKEWORD_COMMON_VOICE_ROOT)" --out-root "$(WAKEWORD_UNKNOWN_EXTERNAL_OUT_DIR)" --target-count $(WAKEWORD_UNKNOWN_TARGET_COUNT) --seed $(WAKEWORD_UNKNOWN_SEED)

wakeword-build-unknown-external-dry-run:
	$(UV) run python -m me2_voicegen.wakeword.build_unknown_external --common-voice-root "$(WAKEWORD_COMMON_VOICE_ROOT)" --out-root "$(WAKEWORD_UNKNOWN_EXTERNAL_OUT_DIR)" --target-count $(WAKEWORD_UNKNOWN_TARGET_COUNT) --seed $(WAKEWORD_UNKNOWN_SEED) --dry-run

# `build_dataset.py` -- ticket 04's real final assembly: merges
# positives_real/positives_converted/positives_converted_noisy (_wakeword_),
# adversaries/adversaries_noisy/common_voice_negative_sample (_unknown_),
# silence_synthetic (_silence_), then a group-disjoint 70/20/10 split. Run
# wakeword-build-unknown-external and wakeword-generate-silence (sized to
# the intended class composition) before this.
wakeword-build-dataset:
	$(UV) run python -m me2_voicegen.wakeword.build_dataset --wakeword-root "$(WAKEWORD_ROOT)" --seed $(WAKEWORD_BUILD_SEED)

# Feature `wakeword-dscnn` (feature-engineering/wakeword-dscnn/SPEC.md):
# DS-CNN keyword-spotting classifier training/export/benchmark, on top of
# the dataset-build targets above. `wakeword-derive-spans` must run once
# after wakeword-build-dataset (or after any re-run of it) and before
# wakeword-train -- it writes speech_start_s/speech_end_s into the
# _wakeword_ chain + common_voice_negative_sample's own subset manifests;
# re-run wakeword-build-dataset afterward to carry those columns into the
# final assembled manifest.csv. adversaries/adversaries_noisy are
# deliberately untouched (SPEC.md round 3).
wakeword-derive-spans:
	$(UV) run python -m me2_voicegen.wakeword.derive_speech_spans --wakeword-root "$(WAKEWORD_ROOT)"

wakeword-train:
	$(UV) run python -m me2_voicegen.wakeword.train --manifest "$(WAKEWORD_TRAIN_MANIFEST)" --out-dir "$(WAKEWORD_TRAIN_OUT_DIR)" --preset $(WAKEWORD_PRESET) --device $(WAKEWORD_DEVICE) --max-minutes $(WAKEWORD_MAX_MINUTES) --seed $(WAKEWORD_TRAIN_SEED) --noise-root "$(WAKEWORD_NOISE_ROOT_TRAIN)"

# export_onnx.py/benchmark.py are vcm/'s own modules, extended with
# --model-family wakeword rather than duplicated (SPEC.md's explicit
# deferral decision) -- these targets are the wakeword-pointed invocation
# of the same CLIs vcm-bench/vcm.export_onnx already use.
wakeword-export:
	$(UV) run python -m me2_voicegen.vcm.export_onnx --model-family wakeword --checkpoint "$(WAKEWORD_CHECKPOINT)" --out-dir "$(WAKEWORD_TRAIN_OUT_DIR)" --manifest "$(WAKEWORD_TRAIN_MANIFEST)" --calibration-samples $(WAKEWORD_BENCH_CALIBRATION_SAMPLES)

wakeword-bench:
	$(UV) run python -m me2_voicegen.vcm.benchmark --model-family wakeword --checkpoint "$(WAKEWORD_CHECKPOINT)" --out-dir "$(WAKEWORD_TRAIN_OUT_DIR)" --manifest "$(WAKEWORD_TRAIN_MANIFEST)" --calibration-samples $(WAKEWORD_BENCH_CALIBRATION_SAMPLES) --n-iters $(WAKEWORD_BENCH_N_ITERS)

# Soak-test three-way tally (feature-engineering/soak-test-tally/SPEC.md):
# scores one podcast's Phase 1 soak-test JSONL (phase1_podcastN.jsonl,
# repo root -- produced by a live `vcm.streaming --log-decode-diagnostics`
# run, not by this Makefile) against that podcast's own transcript
# (soak-test/podcast_N/podcast_N.md), bucketing every trigger as
# ACOUSTIC_GHOST / MEDIA_TRUE_POSITIVE / PHONETIC_NEAR_MISS. Report is
# written inside the podcast's own directory, alongside its audio/
# transcript, rather than under .scratch/ -- SOAK_TALLY_PODCAST selects
# which one (default 1); override per-podcast, e.g.
# `make soak-tally SOAK_TALLY_PODCAST=2`. `.scratch/soak-test-tally/
# three_way_scorer.py` is a spike script (SPEC.md's own Non-goals), not a
# `me2_voicegen` module -- invoked directly, not via `-m`.
SOAK_TALLY_PODCAST ?= 1
SOAK_TALLY_JSONL ?= phase1_podcast$(SOAK_TALLY_PODCAST).jsonl
SOAK_TALLY_TRANSCRIPT ?= soak-test/podcast_$(SOAK_TALLY_PODCAST)/podcast_$(SOAK_TALLY_PODCAST).md
SOAK_TALLY_TOLERANCE_S ?= 2.0
SOAK_TALLY_OUT ?= soak-test/podcast_$(SOAK_TALLY_PODCAST)/phase1_three_way_report.csv

soak-tally:
	$(UV) run python .scratch/soak-test-tally/three_way_scorer.py "$(SOAK_TALLY_JSONL)" "$(SOAK_TALLY_TRANSCRIPT)" --tolerance-s $(SOAK_TALLY_TOLERANCE_S) --out "$(SOAK_TALLY_OUT)"

test:
	$(UV) run pytest

clean:
	rm -rf .venv vendor models out .pytest_cache
