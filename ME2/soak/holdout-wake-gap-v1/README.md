# Holdout soak v1: wake word + gap + command

One 32.6-minute recording (`continuous.wav`, 16 kHz mono) of all 202 ai231 holdout rows (186 commands, 16 out-of-scope clips that must not trigger),
each preceded by a wake word with a random 0.0-1.0 s gap (0.1 s steps), mixed with a room impulse response and dataset ambient noise, with 1-8 s of ambient
noise between units. Truth for every unit is in `continuous.json` (times relative to the unit start; `offset_s` is where the unit starts in the file).
Method, results and tuning: [`docs/SOAK-TEST.md`](../../docs/SOAK-TEST.md).

**The wav is not committed** (public repo; clip and wake-word terms unconfirmed). Get it from the cluster (`out/soak/holdout-wake-gap-v1/continuous.wav`, or
`AI-222-Machine-Exercises/archive/soak/holdout-wake-gap-v1.zip`), or rebuild it from the seeds (commands in `docs/SOAK-TEST.md`). Check it with `sha256sum -c SHA256SUMS`.

## Run it (CPU INT8 ONNX, the same on a Raspberry Pi)

```bash
cp <path>/continuous.wav soak/holdout-wake-gap-v1/                 # next to continuous.json
make soak-run SOAK_DIR=soak/holdout-wake-gap-v1 SOAK_NAME=rpi4 SOAK_ARGS="--backend onnx --threads 1"
cat soak/holdout-wake-gap-v1/results/rpi4.md
```

Compare with `results/tuned-cpu-onnx-int8-hybrid.md` (server CPU) and `results/tuned-a100-hybrid.md` (A100). On a Pi, watch `decode per window` and the
real-time factor: at the 0.25 s stride the server's p95 is about 0.5; a value above 1 means the Pi cannot keep up live.
Add `--stride-s 0.125` for the faster stride and `--no-cls` for CTC only.

`results/`: `*.md` summaries of every run (before tuning, tuned, strides) and `*.json` (per-unit triggers) for the tuned runs.
