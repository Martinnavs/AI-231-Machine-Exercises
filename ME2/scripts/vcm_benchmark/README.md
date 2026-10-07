# VCM benchmark replay and analysis scripts

Used for `docs/VCM-BENCHMARK-RESULTS.md`. They run from the root of a clone of <https://github.com/airimonda/vcm-benchmark> made at `<repo>/.vcm-benchmark/`
(copy these files into that clone; they import its `benchmark.py` / `vcmbench` and find this repo's `ME2/` one level up). The clone, its `runs/` and the recordings are not committed.

| File | What it does |
|---|---|
| `replay_bench.py` | `build` / `stream` / `score`: the benchmark's own trial building and scoring, with the trials streamed through `me2_voicegen.vcm.streaming` as one wav (file replay) |
| `calib.py` | per-mel-band gain map between the clean audio and a recording; applying it |
| `agc.py`, `agc_score.py` | causal AGC with a noise gate, and the fixed-gain / AGC scoring of the laptop recording |
| `phone_align*.py`, `phone_score*.py` | per-trial alignment of a recording to the clean timeline, then raw and mel-mapped scoring (phone and laptop runs) |

The acoustic runs need the played file (`vcm_full.wav`, built from `runs/full/stream.wav` plus three beeps) and an `arecord` capture; see the results document for the procedure and its limits.
