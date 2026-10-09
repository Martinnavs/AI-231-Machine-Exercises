# holdout-nowake-v1: the holdout soak with no wake word

Same 202 holdout units, rooms, noise clips and SNRs as `../holdout-wake-gap-v1` (seed 0), but no "sesame" and no gap, so run it with `--gate always`
(every 0.25 s window is decoded). 27.7 min, 16 kHz mono; about 15 min is noise only. Research and education use only, same terms as `../holdout-wake-gap-v1`.

```bash
M=out/conversions/v2
uv run python scripts/build_soak_audio.py --out-dir out/soak/holdout-nowake-v1 --split holdout --seed 0 --no-wake \
    --vcm-manifest $M/ai231-v2/manifest.csv --wakeword-manifest $M/wakeword-sesame/manifest.csv
uv run python scripts/build_soak_continuous.py --sessions out/soak/holdout-nowake-v1
uv run python scripts/soak_run.py --sessions out/soak/holdout-nowake-v1 --continuous --gate always --name nowake-ctc-only --no-cls --backend onnx --threads 1
uv run python scripts/soak_run.py --sessions out/soak/holdout-nowake-v1 --continuous --gate always --name nowake-hybrid --cls-slot-threshold 0.6 --backend onnx --threads 1
```

Results: `results/nowake-ctc-only.{md,json}`, `results/nowake-hybrid.{md,json}` (summary in the top-level README). Not rebuilt bit for bit.
