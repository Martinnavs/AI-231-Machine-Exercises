# Cascade Soak Test: Wakeword Gate -> VCM (production checkpoints)

## Why this exists

Both `out/wakeword-sesame-ambient-rir-45m` and `out/vcm/option-d-fil50-ambient-rir-135m`
were promoted based on each stage's own **isolated** noisy/reverb-gate false-accept rate:
VCM's noisy babble false-accept rate is 0.039 (up from the shipped checkpoint's 0.004),
and wakeword's `_unknown_`-class miss rate under noise is 0.040. Naively multiplying the
two gives an estimated end-to-end false-action rate of roughly 0.16% per gated attempt --
but that assumes the two stages' failures are independent and it was never actually
measured: the two numbers come from different eval manifests, never run through the real
cascade on the same audio. This test measures the **real** compound false-action rate by
running the actual `me2_voicegen.vcm.streaming` CLI -- the same `ListeningGate` /
`wakeword_gate.py` code path a deployed device runs, not a hand-reconstructed call graph
-- against long, continuous, real background-conversation audio that contains zero
genuine "sesame" utterances. Any `ACCEPT` in this run is, by construction, a false
action.

## Soak audio

`raw_datasets/ambient-noise/` has the 4 real recordings the ambient-noise-overlay
feature's babble corpus was built from (real YouTube background-conversation/ambience,
CC-licensed, none containing the sesame wakeword):

- `COFFEE SHOP AMBIENCE ｜ People Talking ｜ FREE To Use.wav` (~117 MB -- shortest, start here)
- `1 Hour Filipino Café⧸Coffee Shop Noise Ambience for Studying, Focus and Homework.wav` (~691 MB, ~1 hr)
- `ANXIETY RELIEF ANTI-STRESS RELAXING AND CALMING MORNING SOUNDS OF SUBURBAN PHILIPPINES (1+ HOURS).wav` (~766 MB, ~1 hr+)
- `Boost productivity instantly with this 1 Hour Classroom Noises - Classroom Ambience - Study With Me.wav` (~703 MB, ~1 hr)

`WavFileSource` auto-downmixes to mono and resamples to 16 kHz on load, so these can be
passed directly with no preprocessing.

## Run

Flag choices, stated plainly:
- `--threshold -0.1` is `option-d-fil50-ambient-rir-135m`'s own clean-val-chosen operating
  threshold (from its `eval_report.md`) -- the same threshold whose noisy babble FAR
  (0.039) motivated this test.
- `--required-command-margin` is **omitted on purpose**: it's an separate, additional
  incomplete-prefix rejection gate that was tuned for a different checkpoint
  (`option-d-fil50`) in prior docs; carrying that value over here would confound this
  test's one job (measuring the threshold-based accept/reject gate's real false-action
  rate) with an untuned second gate. Leave it disabled (the default) for this test.
- `--policy single_period --gate-period 3` is the only policy the wakeword gate
  currently supports (`single_period` requires exactly `--gate-period 3`, enforced by the
  CLI itself).
- `--wakeword-onnx-variant` is left at its default, matching the existing
  `ACCENT-BALANCE-FIL50-COMMANDS.md` precedent for this same flag combination.

Short soak (start here):

```bash
uv run python -m me2_voicegen.vcm.streaming \
  --model out/vcm/option-d-fil50-ambient-rir-135m --backend onnx --onnx-variant int8 \
  --grammar optionb --threshold -0.1 \
  --beam-width 50 --gate wakeword --policy single_period --gate-period 3 \
  --wakeword-model out/wakeword-sesame-ambient-rir-45m --wakeword-backend onnx \
  --log-periods --log-all-windows \
  --source "raw_datasets/ambient-noise/COFFEE SHOP AMBIENCE ｜ People Talking ｜ FREE To Use.wav" \
  >soak_coffee_short.jsonl 2>soak_coffee_short.log
```

Full soak (run once the short one looks clean -- one file at a time, or all three):

```bash
uv run python -m me2_voicegen.vcm.streaming \
  --model out/vcm/option-d-fil50-ambient-rir-135m --backend onnx --onnx-variant int8 \
  --grammar optionb --threshold -0.1 \
  --beam-width 50 --gate wakeword --policy single_period --gate-period 3 \
  --wakeword-model out/wakeword-sesame-ambient-rir-45m --wakeword-backend onnx \
  --log-periods --log-all-windows \
  --source "raw_datasets/ambient-noise/1 Hour Filipino Café⧸Coffee Shop Noise Ambience for Studying, Focus and Homework.wav" \
  >soak_cafe.jsonl 2>soak_cafe.log

uv run python -m me2_voicegen.vcm.streaming \
  --model out/vcm/option-d-fil50-ambient-rir-135m --backend onnx --onnx-variant int8 \
  --grammar optionb --threshold -0.1 \
  --beam-width 50 --gate wakeword --policy single_period --gate-period 3 \
  --wakeword-model out/wakeword-sesame-ambient-rir-45m --wakeword-backend onnx \
  --log-periods --log-all-windows \
  --source "raw_datasets/ambient-noise/ANXIETY RELIEF ANTI-STRESS RELAXING AND CALMING MORNING SOUNDS OF SUBURBAN PHILIPPINES (1+ HOURS).wav" \
  >soak_anxiety.jsonl 2>soak_anxiety.log

uv run python -m me2_voicegen.vcm.streaming \
  --model out/vcm/option-d-fil50-ambient-rir-135m --backend onnx --onnx-variant int8 \
  --grammar optionb --threshold -0.1 \
  --beam-width 50 --gate wakeword --policy single_period --gate-period 3 \
  --wakeword-model out/wakeword-sesame-ambient-rir-45m --wakeword-backend onnx \
  --log-periods --log-all-windows \
  --source "raw_datasets/ambient-noise/Boost productivity instantly with this 1 Hour Classroom Noises - Classroom Ambience - Study With Me.wav" \
  >soak_classroom.jsonl 2>soak_classroom.log
```

`--source <wav-path>` file-replay is **not** real-time-paced -- it runs as fast as
inference allows, not at 1x playback speed. The wakeword gate scores every window
cheaply (~0.1-0.3ms per the shipped benchmark); VCM's beam decode only runs inside an
opened 3-second listening period, so a clean run with few/no false wakeword triggers
should finish well under real-time.

## Reading the results

Every `ACCEPT` here is a genuine defect (this audio has no real "sesame" utterances):

```bash
grep -c "period: ACCEPT" soak_*.log    # false-action count per run
grep "period: ACCEPT" soak_*.log       # the actual reason/confidence for each one
```

Total audio exposure per run, to turn a count into a rate:

```bash
uv run python3 -c "
import torchaudio, sys
info = torchaudio.info(sys.argv[1])
print(info.num_frames / info.sample_rate / 3600, 'hours')
" "raw_datasets/ambient-noise/<file>.wav"
```

False-action rate = `ACCEPT` count / hours of audio. Compare this measured rate against
the ~0.16%-per-gated-attempt estimate from multiplying the two stages' isolated
noisy-gate FARs (see "Why this exists" above) -- if the measured rate comes out much
higher, the two stages' failures are correlated rather than independent, and that
independence assumption should be dropped from future promotion write-ups rather than
reused.
