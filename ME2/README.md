# ME2: spoken-command voice assistant

**BLUF.** A wake word ("sesame") followed by one of 19 spoken commands, decoded on-device by a 14.3 MB INT8 model trained on public data only.
The answer is ready about 0.5 s after the end of speech on a Raspberry Pi 4 using one core. Out-of-scope rejection is the weak spot.
Branch `optionb-ctc-attention-fast-beam`; not promoted (`make app-pipeline*` still runs the earlier model).

## Headline results

| Measure (INT8 ONNX, one seed) | Result |
| --- | ---: |
| ai231 test, command + slot accuracy (3,823 clips) | **98.8%** |
| ... human voices only | 91.8% (214/233) |
| VCM benchmark, clean replay on a Pi 4: intent / command (218 trials) | **91.6% / 89.6%** |
| Pi 4 latency after end of speech, median / p95 | **0.48 / 0.92 s** |
| Pi 4 real-time factor, p95 (1 = live limit) | 0.96 |

Weak spots: out-of-scope false accepts (3.9% on test, 25% on the benchmark), false wakes with no wake word, and real voices.
Through a real speaker and microphone it is lower (77% / 75%, partial run). Full tables and caveats: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md), [`docs/VCM-BENCHMARK-RESULTS.md`](docs/VCM-BENCHMARK-RESULTS.md), [`docs/SOAK-TEST.md`](docs/SOAK-TEST.md); every model side by side: [`docs/BASELINES.md`](docs/BASELINES.md).

## Architecture

- **Wide CTC** (4.2 M params): grammar-constrained prefix beam search over the 19-intent, 93-wording grammar; if it accepts, that is the answer.
- **XL heads** (10 M params): intent and slot classifier, used only when the CTC rejects and confidence is above 0.8787.
- **Streaming:** wake word -> listen until the command ends -> CTC, else classifier -> reject. Detail: [`docs/CURRENT-MODEL.md`](docs/CURRENT-MODEL.md).

## Quickstart

```bash
module load uv && make sync                          # once
make hybrid-decode HYBRID_WAV="clip1.wav clip2.wav"  # JSON per clip, which model answered
make hybrid-stream                                   # live microphone: wake word, then the command
make reproduce                                       # downloads ~4.8 GB, rebuilds the data, re-checks every headline number: ~9 min on a server CPU
```

`make reproduce` writes `out/reproduce/REPORT.md` (PASS/FAIL per check). Retraining too: `make reproduce REPRO_TRAIN=1 REPRO_GPU=N` (~2.5 h on one A100).
Raspberry Pi: `uv sync` does not work there, see [`docs/RASPBERRY-PI.md`](docs/RASPBERRY-PI.md). All commands: [`docs/USAGE.md`](docs/USAGE.md).

## Data and licence

- **Data:** [`airimonda/ai231-me2-voice-commands`](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands), DOI [10.57967/hf/10723](https://doi.org/10.57967/hf/10723): Ailene Nunez 2026, "ai231-me2-voice-commands (Revision e828320)", Hugging Face. The gap-fill and numeral-wording clips come from [`martinnavs/ai231-fil-supplemental-data`](https://huggingface.co/datasets/martinnavs/ai231-fil-supplemental-data) (revision 230c8b8); they are not in the DOI dataset.
- **Licence:** code is MIT ([`LICENSE`](LICENSE)). The model weights are also bound by the dataset's research-and-education terms ([`MODEL-CARD.md`](out/vcm/hybrid-ctcwide-clsxl/MODEL-CARD.md)).

## Where everything else is

| Need | Read |
| --- | --- |
| Commands (laptop / Pi), serving, Python API, tests, training | [`docs/USAGE.md`](docs/USAGE.md) |
| Current model, data, recipe, code map, open items | [`docs/CURRENT-MODEL.md`](docs/CURRENT-MODEL.md) |
| Reproducing it, what is not public | [`docs/REPRODUCE-HYBRID.md`](docs/REPRODUCE-HYBRID.md) |
| Every model tried, side by side | [`docs/BASELINES.md`](docs/BASELINES.md) |
| Every benchmark table | [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) |
| UI dashboard, music, headless Pi demo on a phone hotspot | [`docs/UI-DEMO.md`](docs/UI-DEMO.md) |
| Soak test (full tables), Raspberry Pi | [`docs/SOAK-TEST.md`](docs/SOAK-TEST.md), [`docs/RASPBERRY-PI.md`](docs/RASPBERRY-PI.md) |
| Final pre-production test (VCM benchmark on a Pi 4) | [`docs/VCM-BENCHMARK-RESULTS.md`](docs/VCM-BENCHMARK-RESULTS.md), [`scripts/vcm_benchmark/`](scripts/vcm_benchmark/README.md) |
| Experiment log, streaming commands | [`docs/AI231-FIL50.md`](docs/AI231-FIL50.md), [`docs/CTC-ATTENTION.md`](docs/CTC-ATTENTION.md) |
| Released files, model card | `out/vcm/hybrid-ctcwide-clsxl/` ([`MODEL-CARD.md`](out/vcm/hybrid-ctcwide-clsxl/MODEL-CARD.md)), `out/wakeword-sesame-ambient-rir-45m/`, `soak/holdout-wake-gap-v1/` |
| All docs; the earlier ~1M-parameter model and the TTS data pipeline | [`docs/README.md`](docs/README.md), [`docs/archive/`](docs/archive/README-EARLIER-MODEL.md) |
