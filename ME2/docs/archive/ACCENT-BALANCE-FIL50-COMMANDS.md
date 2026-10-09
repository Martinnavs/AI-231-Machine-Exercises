# accent-balance-fil50 — reproduction & serving commands

Raw `uv run` commands for interacting with this experiment's specific
artifacts (`out/vcm/option-d-fil50/`, `out/wakeword-fil50/`). Kept here
instead of new Makefile recipes — every command below is already exactly
what `Makefile`'s existing `vcm-*`/`optionb-*`/`vcmx-*`/`wakeword-*` targets
run, just with this experiment's paths substituted in place of their
defaults, so adding fil50-specific targets would only duplicate the
Makefile's own machinery under new names. See `MLOPS-PROJECTS.md`'s
Iteration 1 for the experiment's narrative and full results; this file is
commands only.

Replace `cuda:N` with a free GPU index (or `cpu`) — none of the values
below assume a particular device is idle.

## Paths this experiment produced

| | VCM | Wakeword |
|---|---|---|
| manifest | `out/conversions/v2/optionb-v3-vcmx-fil50/manifest.csv` | `out/conversions/v2/wakeword/manifest.fil50.csv` |
| run dir | `out/vcm/option-d-fil50` | `out/wakeword-fil50` |
| checkpoint | `out/vcm/option-d-fil50/checkpoints/checkpoint.pt` | `out/wakeword-fil50/checkpoints/checkpoint.pt` |
| calibrated threshold | `-0.075` (this checkpoint's own val-sweep, not the `-0.1` baseline default) | gate `0.9` (unchanged from baseline) |

## Re-evaluate

VCM:

```bash
uv run python -m me2_voicegen.vcm.evaluate \
  --manifest out/conversions/v2/optionb-v3-vcmx-fil50/manifest.csv \
  --checkpoint out/vcm/option-d-fil50/checkpoints/checkpoint.pt \
  --out-dir out/vcm/option-d-fil50 \
  --device cuda:N --beam-width 50 --grammar optionb
```

Wakeword (`--eval-only` scores an existing checkpoint without retraining):

```bash
uv run python -m me2_voicegen.wakeword.train \
  --manifest out/conversions/v2/wakeword/manifest.fil50.csv \
  --out-dir out/wakeword-fil50 --eval-only \
  --checkpoint out/wakeword-fil50/checkpoints/checkpoint.pt \
  --device cuda:N
```

Either writes `metadata/eval_report.{json,md}` under its `--out-dir`, overwriting the
existing report — copy it aside first if you want to keep the original alongside a rerun.

## Export + benchmark

Already done for this run (`out/vcm/option-d-fil50/export/`,
`out/wakeword-fil50/export/`); re-run only if the checkpoint changes.

VCM (`vcm.benchmark` does the ONNX export as a side effect):

```bash
uv run python -m me2_voicegen.vcm.benchmark \
  --checkpoint out/vcm/option-d-fil50/checkpoints/checkpoint.pt \
  --out-dir out/vcm/option-d-fil50 \
  --manifest out/conversions/v2/optionb-v3-vcmx-fil50/manifest.csv \
  --calibration-samples 32 --n-iters 100
```

Wakeword (export, then benchmark — separate steps for wakeword only):

```bash
uv run python -m me2_voicegen.vcm.export_onnx --model-family wakeword \
  --checkpoint out/wakeword-fil50/checkpoints/checkpoint.pt \
  --out-dir out/wakeword-fil50 \
  --manifest out/conversions/v2/wakeword/manifest.fil50.csv \
  --calibration-samples 32

uv run python -m me2_voicegen.vcm.benchmark --model-family wakeword \
  --checkpoint out/wakeword-fil50/checkpoints/checkpoint.pt \
  --out-dir out/wakeword-fil50 \
  --manifest out/conversions/v2/wakeword/manifest.fil50.csv \
  --calibration-samples 32 --n-iters 100
```

## Serve

Requires the export step above to have run at least once (it has, for this
run). Ungated (mic always listening):

```bash
uv run python -m me2_voicegen.vcm.streaming \
  --model out/vcm/option-d-fil50 --backend onnx --onnx-variant int8 \
  --grammar optionb --threshold -0.075 --required-command-margin 4.0 \
  --beam-width 50 --gate none --source mic
```

Wakeword-gated (fronts the VCM model with this run's own wakeword
checkpoint via `--wakeword-model`, not the pre-fil50 default):

```bash
uv run python -m me2_voicegen.vcm.streaming \
  --model out/vcm/option-d-fil50 --backend onnx --onnx-variant int8 \
  --grammar optionb --threshold -0.075 --required-command-margin 4.0 \
  --beam-width 50 --gate wakeword --policy single_period --gate-period 3 \
  --wakeword-model out/wakeword-fil50 --wakeword-backend onnx --log-periods \
  --source mic
```

`--source` takes a wav path in place of `mic` for deterministic file-replay.

## Before/after comparison (as actually run for this experiment)

The old (pre-fil50) checkpoints scored against fil50's *new* manifest, for
a clean same-data comparison — this is what produced the before/after
numbers in `MLOPS-PROJECTS.md`:

```bash
uv run python -m me2_voicegen.vcm.evaluate \
  --manifest out/conversions/v2/optionb-v3-vcmx-fil50/manifest.csv \
  --checkpoint out/vcm/option-d-dataset-v2/checkpoints/checkpoint.pt \
  --out-dir out/vcm/option-d-fil50/baseline_old_ckpt \
  --device cuda:N --beam-width 50 --grammar optionb

uv run python -m me2_voicegen.wakeword.train \
  --manifest out/conversions/v2/wakeword/manifest.fil50.csv \
  --out-dir out/wakeword-fil50/baseline_old_ckpt --eval-only \
  --checkpoint out/wakeword/checkpoints/checkpoint.pt \
  --device cuda:N
```
