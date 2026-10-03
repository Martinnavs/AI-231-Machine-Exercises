# Soak test: wake word + gap + command, with reverb and ambient noise

A deployment-shaped test of the current model (the hybrid, see [`CURRENT-MODEL.md`](CURRENT-MODEL.md)): the real streaming pipeline (wake word gate ->
endpointed policy -> grammar CTC with classifier fallback) is fed one long recording made of the ai231 **holdout** commands, each preceded by a wake word,
and scored on accuracy and latency. The same audio is meant to be replayed on a Raspberry Pi.

## The audio

```
[ wake word ] [ gap 0.0-1.0 s, steps of 0.1 s ] [ command ] [ 0.5 s reverb tail ]   <- one unit
        ... 1.0-8.0 s of ambient noise (steps of 0.1 s) ...   <next unit>   ...
```

- **Units:** all 202 holdout rows: 186 commands and 16 out-of-scope clips (Common Voice sentences that must NOT trigger). Order is seeded and shuffled.
- **Wake words:** clean "sesame" positives from the `test` split of the wake-word manifest (121 clips; the noise-mixed copies are excluded).
- **Room and noise:** each session is convolved with a room impulse response from the seeded pool of `vcm.noisy_eval` (200 synthetic rooms), then mixed with an
  ambient clip from the dataset's own `background_noise` rows (other splits only; never ESC-50) at 10-25 dB SNR measured against the reverberated speech,
  not the silent parts. The ambient noise between units continues the previous unit's own noise clip at the level measured in its noise-only tail.
- **Truth:** `continuous.json` has, per unit, the label, slot, where the wake word ends, the gap, where the command's speech starts and ends (CTC forced alignment of the
  clean clip, times relative to the unit start), the RIR index, the noise clip and the SNR. 32.6 minutes, 16 kHz mono, 63 MB, `sha256` in `soak/holdout-wake-gap-v1/SHA256SUMS`.
- **In git, by decision:** `soak/holdout-wake-gap-v1/continuous.wav` is force-added (`*.wav` is git-ignored) so a Pi checkout has the audio. The repo is public and the clips come from datasets with several
  non-commercial terms, plus wake-word voices of unconfirmed provenance (see `REPRODUCE-HYBRID.md`): treat the recording as research and education use only, and do not reuse it commercially.
  The 63 MB file is also in `out/soak/holdout-wake-gap-v1/` and `AI-222-Machine-Exercises/archive/soak/holdout-wake-gap-v1.zip` on the cluster. It rebuilds from the seeds:

```bash
M=out/conversions/v2     # ai231-v2 (import_ai231 output) and wakeword-sesame manifests
uv run python scripts/build_soak_audio.py --out-dir out/soak/holdout-wake-gap-v1 --split holdout --seed 0 \
    --vcm-manifest $M/ai231-v2/manifest.csv --wakeword-manifest $M/wakeword-sesame/manifest.csv      # 202 per-session wavs + sessions.json
uv run python scripts/build_soak_continuous.py --sessions out/soak/holdout-wake-gap-v1                 # continuous.wav + continuous.json
# tuning soak from validation: --split val --wake-split val --sample 240 --seed 1 --out-dir out/soak/val-wake-gap-v1
```

(A rebuild is expected to give identical audio because every choice is seeded; it was not compared bit for bit.)

## Running it

```bash
# the current streaming settings, scored; results in <sessions>/results/<name>.{md,json} and raw JSONL in <sessions>/raw/<name>/
make soak-run SOAK_DIR=out/soak/holdout-wake-gap-v1 SOAK_NAME=my-run                                       # CPU, INT8 ONNX (what the Pi runs)
make soak-run SOAK_DIR=out/soak/holdout-wake-gap-v1 SOAK_NAME=a100 SOAK_ARGS="--backend torch --device cuda:0 --gpu 5"   # PyTorch on a GPU (never GPU 6)
# options: --no-cls (CTC only), --stride-s 0.125, --gate-period 3, --wakeword-threshold, --cls-slot-threshold, --workers N (per-session mode), --limit N
```

`scripts/soak_run.py` streams `continuous.wav` once (or one process per session without `--continuous`) with `--log-all-windows --log-timing` and scores by time range.
**On a Raspberry Pi:** copy the repo (or `scripts/soak_run.py` and the model files), put `continuous.wav` and `continuous.json` in a folder, and run the same command with
`--backend onnx --threads 1`. The timings in the result use the same fields, so the two machines are directly comparable; `raw/<name>/continuous.jsonl` is every streaming record.

**Scoring.** A command is correct if the first trigger in its time range (unit start to the next unit's start) has the right intent and slot; missed if there is no trigger
(`missed_no_period_opened` counts those where the wake word never opened a period); a *wrong action* is a first trigger with the wrong intent or slot; an out-of-scope unit with any
trigger is a false accept; `triggers_in_ambient_gaps` counts triggers more than 1.5 s after a unit ended. Latency is the first trigger's time minus the end of speech (algorithmic: the
replay waits for each decode) and the wall-clock `gate_ms` (wake-word scoring) and `decode_ms` (encoder + beam search + policy) of every decoded window. The tuning objective is
`correct - 2 * (wrong actions + out-of-scope false accepts + gap triggers)`.

## Results (holdout, 186 commands + 16 out-of-scope; one seed; stride 0.25 s, wake word polled every 0.05 s unless stated)

Tuned settings (below): wake-word threshold 0.8, classifier slot gate 0.6. Full tables: `soak/holdout-wake-gap-v1/results/`.

| run | correct first trigger | wrong actions | missed (no period) | out-of-scope false accepts | latency after speech end p50 / p95 | decode per window mean / p95 |
|---|---|---|---|---|---|---|
| A100 hybrid, tuned | 152 (81.7%) | 9 | 25 (5) | 4 / 16 | 0.38 / 0.66 s | 63.8 / 101.8 ms |
| CPU INT8 hybrid, tuned | 151 (81.2%) | 9 | 26 (5) | 3 / 16 | 0.38 / 0.65 s | 70.2 / 126 ms |
| A100 CTC only, tuned wake word | 139 (74.7%) | 2 | 45 (5) | 3 / 16 | 0.37 / 0.48 s | 65.5 / 103 ms |
| CPU INT8 CTC only, tuned wake word | 137 (73.7%) | 3 | 46 (5) | 3 / 16 | 0.37 / 0.48 s | 70.4 / 111 ms |
| CPU INT8 hybrid, tuned, stride 0.125 s | 154 (82.8%) | 10 | 22 (5) | 3 / 16 | 0.30 / 0.55 s | 69.0 / 116 ms |
| A100 hybrid, before tuning (3 s, thr 0.9, no slot gate) | 148 (79.6%) | 12 | 26 (8) | 4 / 16 | 0.38 / 0.66 s | 67.6 / 107 ms |
| CPU INT8 hybrid, before tuning | 147 (79.0%) | 12 | 27 (8) | 3 / 16 | 0.38 / 0.66 s | 74.4 / 123 ms |
| A100 hybrid, before tuning, stride 0.05 s | 150 (80.7%) | n/a | 20 (8) | 4 / 16 | 0.26 / 0.54 s | 64.6 / 105 ms |
| CPU INT8 hybrid, before tuning, stride 0.05 s | 149 (80.1%) | n/a | 20 (8) | 3 / 16 | 0.26 / 0.54 s | 72.6 / 132 ms |

- **Triggers in the ambient gaps: 0 in every run.** The out-of-scope false accepts are speech after a wake word, which the gate cannot filter.
- **A100 = CPU.** The A100 is not faster per window because the grammar beam search runs on the CPU; the encoder is 8-15 ms of a ~70 ms decode. The CPU INT8 run is the fair stand-in for the Pi.
- **Real time.** p95 (gate + decode) over the stride: 0.45-0.53 at 0.25 s; 0.96 at 0.125 s (just keeps up on a server core; a Pi core is slower); 2.1-2.7 at 0.05 s (cannot keep up live: the
  replay is in lockstep so its accuracy and latency are exact, but a live microphone would queue and drop windows). Stride mostly changes latency: 0.38 s -> 0.30 s -> 0.26 s.
- **Hybrid vs CTC only.** The hybrid answers 12-13 more commands (the classifier recovers CTC rejections) but makes about 7 more wrong actions (mostly a wrong slot value), so by the objective
  above the two are about even (A100: 126 hybrid vs 129 CTC only). If a wrong action costs more than a missed command, prefer CTC only or raise `--cls-slot-threshold`.

## Why the failures happen (A100 hybrid before tuning, 186 commands)

| outcome | count | cause and what was done |
|---|---|---|
| correct | 148 | 12 of them fired slightly before the aligned end of speech |
| wrong slot or intent, fired early | 9 | 8 are the classifier fallback answering the right intent with the wrong slot (a 1-minute timer instead of 10 seconds): **slot-confidence gate added** |
| wrong, late | 3 | |
| wake word never opened a period | 8 | wake word at threshold 0.9 under reverb and noise: **threshold 0.8** recovers 3; the rest are the wake-word model (retraining) |
| period timed out before the command ended | 6 | the 3 s period is too short for commands that end more than ~4 s after the wake word; a longer period was tried (below) |
| rejected in an open period | 12 | CTC rejects and the classifier is below threshold: accent coverage in the training data, not tunable |
| out-of-scope false accept | 4 of 16 | speech after a wake word |

## Tuning on a validation soak (so the holdout stays a test)

A second soak was built from 240 randomly sampled **validation** rows (221 commands, 19 non-commands) with wake words from the wake-word `val` split (`soak/val-wake-gap-v1/`).
Rule fixed before running: maximise the objective; among settings within 1 point of the best take the one nearest the old default.
CPU INT8 hybrid, one setting changed at a time (period s / wake-word threshold / classifier slot threshold):

| setting | correct | wrong actions | out-of-scope false accepts | missed (no period) | objective |
|---|---|---|---|---|---|
| base: 3 / 0.9 / off | 169 | 7 | 4 | 45 (18) | 147 |
| period 4 | 175 | 9 | 4 | 37 (18) | 149 |
| period 5 | 175 | 10 | 5 | 36 (18) | 145 |
| period 6 | 175 | 10 | 6 | 36 (18) | 143 |
| wake 0.8 | 173 | 7 | 4 | 41 (11) | 151 |
| wake 0.7 | 173 | 7 | 4 | 41 (8) | 151 |
| wake 0.6 | 174 | 7 | 4 | 40 (7) | 152 |
| slot 0.6 | 169 | 6 | 4 | 46 (18) | 149 |
| slot 0.8 | 168 | 6 | 4 | 47 (18) | 148 |
| slot 0.95 | 164 | 5 | 4 | 52 (18) | 146 |
| **wake 0.8 + slot 0.6 (chosen)** | 173 | 6 | 4 | 42 (11) | **153** |
| period 4 + wake 0.8 + slot 0.6 | 176 | 9 | 5 | 36 (12) | 148 |

A longer period recovers timeouts but lets in wrong actions and false accepts, so the period stays 3 s. The chosen settings were then run **once** on the holdout soak (tables above):
+4 correct, wrong actions 12 -> 9 and no period opened 8 -> 5 for the hybrid.

## Caveats

- **Seen noise and rooms.** 131 of the 202 sessions mix an ambient clip that is in the ai231 train split (the models' training noise pool); the room impulse responses come from the same seed-0 pool as training. The speech (holdout clips and test wake words) is unseen. The unseen-noise analog that already exists is the 784-session wake word + command replay on the ai231 test split (room tone from the test split's own noise clips, no reverb): hybrid 90.1%, CTC only 88.9% (`AI231-FIL50.md`, "Hybrid in streaming"). A rebuild with val/test noise only is not done.
- **False wakes from ordinary speech are not measured.** The gaps between units contain ambient noise only, so a lower wake-word threshold (0.8) could add false wakes from conversation that this soak cannot see.
- One seed, one soak build, 186 commands from one real Filipino speaker plus 100 synthetic voices; the out-of-scope set is 16 clips.
- Some wake-word clips in the older `sessions-ai231-test` replay (`out/vcm/hybrid-eval/`) carry ESC-50 noise; this soak uses only clean positives.
- No Raspberry Pi has been run yet. Server numbers are one thread per process on a shared 256-core node (other users' load can inflate timings a little).
- `scripts/soak_run.py` and the streaming flags `--log-timing`, `--wakeword-poll-s`, `--wakeword-threshold`, `--cls-model`, `--cls-slot-threshold`, `--cls-hold-ms`, `--cls-min-speech-ms` are documented in
  [`AI231-FIL50.md`](AI231-FIL50.md) ("Hybrid in streaming") and `python -m me2_voicegen.vcm.streaming --help`.
