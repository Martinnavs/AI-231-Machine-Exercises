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

## QuartzNet-5x3-tiny stride-2 (quartznet-promotion ticket 03)

Same 4 ambient files (3.296 h total), same CLI and INT8 ONNX path, but with
`--model out/vcm/quartznet5x3-s2-fil50-ambient-rir-135m` at the ticket-02
operating point (mean_frame, threshold -0.1, margin 20.0). One deliberate
difference from the optiond commands above: `--required-command-margin 20.0`
is **included**, because it is the calibrated gate for this checkpoint
(docs/archive/QUARTZNET-OPERATING-POINT.md section 2); the optiond runs above omitted
it on purpose. The optiond P3b baseline (303) was measured at margin 4.0, so
the margin settings differ between the two columns -- stated, not hidden.

Runner: `scripts/quartznet_soak_gates.py` (stages `p3` / `p3b` / `p4`,
`summary`), per-file JSONL/stderr under
`out/vcm/quartznet5x3-s2-fil50-ambient-rir-135m/soak/{p3,p3b,p4}/`, summaries
in `out/vcm/quartznet5x3-s2-fil50-ambient-rir-135m/soak/summary.json`. The
P4 clips are the same 821 composed test clips the optiond dense pilot used
(`out/vcm/option-d-fil50-ambient-rir-135m/dense_pilot/cascade_stream/clips/`).

```bash
# P3 cascade (wakeword gate), P3b VCM-only, P4 recall replay -- one stage at a time:
PYTHONPATH=$PWD/src uv run python scripts/quartznet_soak_gates.py p3
PYTHONPATH=$PWD/src uv run python scripts/quartznet_soak_gates.py p3b
PYTHONPATH=$PWD/src uv run python scripts/quartznet_soak_gates.py p4
PYTHONPATH=$PWD/src uv run python scripts/quartznet_soak_gates.py summary
```

| stage (3.296 h unless noted) | optiond baseline | QuartzNet |
|---|---|---|
| P3 cascade wakeword periods opened | 0 (vacuous) | 0 (vacuous) |
| P3 cascade `period: ACCEPT` | 0 | 0 |
| P3b VCM-only `trigger` count | 303 (B: -0.1, margin 4.0) | **101** (3 / 48 / 43 / 7 per file; intents CALL, NEXT, PAUSE, STOP, TIME) |
| P4 correct-on-first-trigger / 821 | 720 (87.7%) | **749 (91.2%)** |
| P4 wrong-intent triggers | 111 | **49** |
| P4 per-focus PAUSE / STOP / TIME | 87/100, 129/134, 86/107 | 99/100, 129/134, **77/107** |
| `dropped` windows (2.5 s / 0.25 s) | 0 | 0 (P3, P3b, all 821 P4 clips) |

Promotion bar (set in the ticket before any number existed): P3b triggers <=
303, zero compound false actions in the cascade, P4 recall within 1 pp of
720/821, `dropped == 0`. **Result: all four hold** (101 <= 303; 0 accepts;
749/821 = 91.2% >= 86.7%; dropped 0). One disclosed shortfall outside the bar:
streaming recall on TIME clips is 77/107 (72.0%) vs optiond's 86/107
(80.4%), -8.4 pp per-focus, while overall recall is +3.5 pp -- reported per
the ticket's "do not tune to pass" rule, not masked by the aggregate.

Small-n caveat, stated plainly as for the optiond runs: the P3 cascade result
is 0 on a small exposure (the wakeword stage never opened a period on this
audio for either model, and the 8-hour hardware soak's baseline was n=10
wakeword false triggers -- an n=10 baseline gives only a ~25-30% CI upper
bound, and n=0 gives none at all). P3 is therefore a no-regression check,
not a measurement of the compound false-action rate; P3b is the stage with
real power.
