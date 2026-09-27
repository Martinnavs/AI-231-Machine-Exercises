# wakeword-sesame — reproduction & hosting commands

Raw `uv run` commands for interacting with this feature's specific
artifacts (`out/wakeword-sesame/`), same spirit as
`ACCENT-BALANCE-FIL50-COMMANDS.md` in the main tree: every command below
is an existing Makefile `wakeword-*`/`stream*` target's own invocation
with this run's paths substituted for its defaults, so no new
wakeword-sesame-specific Makefile targets were added. See
`docs/WAKEWORD-DATASET-CONTRACT.md` §1 and `docs/MLOPS-PROJECTS.md`
§Iteration 3 for the feature's narrative and full results; this file is
commands only.

This is a **parallel, comparison-only** DS-CNN targeting the phrase
"sesame" instead of "computer" — it is not the production wakeword.
Nothing here writes to `out/wakeword/` or `out/wakeword-fil50/`.

Replace `cuda:N` with a free GPU index (or `cpu`).

## Paths this feature produced

| | value |
|---|---|
| dataset manifest | `out/conversions/v2/wakeword-sesame/manifest.csv` |
| run dir | `out/wakeword-sesame` |
| checkpoint | `out/wakeword-sesame/checkpoints/checkpoint.pt` (epoch 16) |
| gate threshold | `gate.DEFAULT_WAKEWORD_THRESHOLD` (0.9, unchanged — not separately calibrated for sesame) |

## Re-evaluate

`--eval-only` scores an existing checkpoint against a manifest without
retraining:

```bash
uv run python -m me2_voicegen.wakeword.train \
  --manifest out/conversions/v2/wakeword-sesame/manifest.csv \
  --out-dir out/wakeword-sesame --eval-only \
  --checkpoint out/wakeword-sesame/checkpoints/checkpoint.pt \
  --device cuda:N
```

Overwrites `metadata/eval_report.{json,md}` under `--out-dir` — copy the
existing report aside first if you want to keep it alongside a rerun.

## Export + benchmark

Already done for this run (`out/wakeword-sesame/export/`); re-run only if
the checkpoint changes.

```bash
uv run python -m me2_voicegen.vcm.export_onnx --model-family wakeword \
  --checkpoint out/wakeword-sesame/checkpoints/checkpoint.pt \
  --out-dir out/wakeword-sesame \
  --manifest out/conversions/v2/wakeword-sesame/manifest.csv \
  --calibration-samples 32

uv run python -m me2_voicegen.vcm.benchmark --model-family wakeword \
  --checkpoint out/wakeword-sesame/checkpoints/checkpoint.pt \
  --out-dir out/wakeword-sesame \
  --manifest out/conversions/v2/wakeword-sesame/manifest.csv \
  --calibration-samples 32 --n-iters 100
```

## Host / serve

There is no standalone "wakeword only" server in this codebase — the
wakeword gate always fronts the `vcm.streaming` runner (it decides *when*
to open a decode window; a VCM model still does the decoding once one
opens). To host the sesame wakeword model, point `--wakeword-model` at
`out/wakeword-sesame` and pick any VCM model for `--model` (the wakeword
gate and the VCM decode model are independent choices):

```bash
uv run python -m me2_voicegen.vcm.streaming \
  --model optionc --backend onnx \
  --policy single_period --gate wakeword --gate-period 3 \
  --wakeword-model out/wakeword-sesame --wakeword-backend onnx \
  --log-periods \
  --source mic
```

`--wakeword-onnx-variant {fp32,int8}` selects the export (default `fp32`).
`--wakeword-threshold <float>` overrides the gate-open softmax threshold
(default 0.9, unchanged from "computer" — not recalibrated for sesame,
per the ticket's non-goals). `--source` takes a wav path in place of `mic`
for deterministic file-replay. Swap `--model optionc`/`--backend onnx` for
whichever VCM model you actually want driving command decoding — the
wakeword gate doesn't care which one fronts it.

To sanity-check the gate alone against a single clip (does it *open*, not
what the VCM then decodes), point `--source` at a short wav of someone
saying "sesame" and watch stderr for the gate-open log line from
`--log-periods`; the JSONL decode result on stdout can be ignored for a
gate-only check.

## Compare against the pre-fil50 "computer" baseline (as actually run)

The commands that produced the before/after wakeword numbers in this
feature's Execution Log (`.scratch/wakeword-sesame/tickets/00-RECAP.md`):

```bash
# sesame checkpoint's own eval (already on disk, shown for reference)
uv run python -m me2_voicegen.wakeword.train \
  --manifest out/conversions/v2/wakeword-sesame/manifest.csv \
  --out-dir out/wakeword-sesame --eval-only \
  --checkpoint out/wakeword-sesame/checkpoints/checkpoint.pt \
  --device cuda:N
```

There is no meaningful "computer" checkpoint comparison for sesame (the
positive class targets a different spoken word entirely, so cross-checkpoint
same-test scoring isn't informative the way it is for same-word iterations
like fil50) — the relevant comparison instead lives in the QA pass-rate
progression across the two positive-generation approaches this feature
tried (8.8% resynthesis → 31.0% voice-conversion), recorded in the
ticket's Execution Log, not as a checkpoint-vs-checkpoint eval.
