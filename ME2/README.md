# ME2: spoken-command voice assistant

**BLUF.** A wake word ("sesame") followed by one of 19 spoken commands, decoded on-device by a 14.3 MB INT8 model trained on public data only.
The answer is ready about 0.5 s after the end of speech on a Raspberry Pi 4 using one core. Out-of-scope rejection is the weak spot.
This is the canonical ME2 on `master`. The UI demo serves it with `make app-pipeline` (`app-pipeline-ctcwide` is the same target).

## Model: hybrid CTC + attention classifier

```
Mic 16 kHz -> log-mel 40 x 250 (2.5 s window, 0.25 s stride)
  -> wake word "sesame" (DS-CNN, 50 KB INT8, threshold 0.8) -> listen 3 s
  -> wide QuartzNet 5x3: grammar-constrained CTC beam search (beam 50, 19 intents, 93 wordings)
       else XL QuartzNet 5x3: attention-pooled intent head + 6 slot heads (confidence >= 0.8787)
       else reject
  -> smart-home UI dashboard (intent + slot; 7 panels, music plays real audio, the rest simulated)
```

| Part | Params | INT8 ONNX |
| --- | ---: | ---: |
| CTC (wide QuartzNet 5x3) | 4.18 M | 4.16 MB |
| Attention heads (XL QuartzNet 5x3) | 9.99 M | 10.10 MB |
| Total (plus the 50 KB wake word) | 14.17 M | 14.3 MB |

Runtime: ONNX Runtime 1.18, one thread, numba beam search. Detail: [`docs/CURRENT-MODEL.md`](docs/CURRENT-MODEL.md).

## Results (INT8 ONNX, one seed)

| Measure | Result |
| --- | ---: |
| ai231 test, command + slot accuracy (3,823 speaker-disjoint clips) | **98.8%** (3,778) |
| ... human voices only | 91.8% (214/233) |
| Pi 4 soak (186 holdout commands, reverb + noise): wake word / command on first trigger | 97.3% / **81.2%** |
| Pi 4 soak false accepts | 3 of 16 out-of-scope, 0 in the noise gaps |
| Pi 4 latency after end of speech, median / p95 | **0.48 / 0.92 s** |
| Pi 4 real-time factor, p95 (1 = live limit) | 0.96 |
| VCM benchmark, clean replay on a Pi 4: intent / command (218 trials) | 91.6% / 89.6% |

Weak spots: out-of-scope false accepts (3.9% on test, 25% on the benchmark), false wakes with no wake word, and real voices.
Through a real speaker and microphone it is lower (77% / 75%, partial run). Full tables and caveats: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md), [`docs/SOAK-TEST.md`](docs/SOAK-TEST.md), [`docs/VCM-BENCHMARK-RESULTS.md`](docs/VCM-BENCHMARK-RESULTS.md); every model tried, side by side: [`docs/BASELINES.md`](docs/BASELINES.md).

## Dataset and training

| | |
| --- | --- |
| Source | [`airimonda/ai231-me2-voice-commands`](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands), incl. the Filipino supplemental clips, DOI [10.57967/hf/10723](https://doi.org/10.57967/hf/10723) |
| Size | 16.5 h, 28,858 clips (train 18,300) |
| Speakers | 613 train groups, 104 unseen in test, 17 Filipino persona voices |
| Labels | 19 intents, 6 slots, 93 wordings |
| Hardware | UPD CoE HPC, 1x A100 per run: wide 94 min, XL 43 min |
| Objective | CTC 1.0 + intent CE 0.3 + slot CE 0.1 |
| Optimiser | AdamW lr 1e-4, wd 0.01, OneCycle, batch 16, AMP |
| Best epoch / val loss | wide 38 of 58 / 0.391; XL 23 of 43 / 0.336 |
| Augmentation | room reverb 0.7, dataset noise 0.5, babble 0.15, time-stretch 0.25; seed 0, one run per network |

## Quickstart

```bash
module load uv && make sync                          # once
make hybrid-decode HYBRID_WAV="clip1.wav clip2.wav"  # JSON per clip, which model answered
make hybrid-stream                                   # live microphone: wake word, then the command
make reproduce                                       # downloads ~4.8 GB, rebuilds the data, re-checks every headline number: ~9 min on a server CPU
```

`make reproduce` writes `out/reproduce/REPORT.md` (PASS/FAIL per check; 12/12 PASS on 2026-10-10). Retraining too: `make reproduce REPRO_TRAIN=1 REPRO_GPU=N` (~2.5 h on one A100).
Raspberry Pi: `uv sync` does not work there, see [`docs/RASPBERRY-PI.md`](docs/RASPBERRY-PI.md). All commands: [`docs/USAGE.md`](docs/USAGE.md).

## Data and licence

- **Data:** Ailene Nunez 2026, "ai231-me2-voice-commands (Revision e828320)", Hugging Face, doi:[10.57967/hf/10723](https://doi.org/10.57967/hf/10723). The gap-fill and numeral-wording clips come from [`martinnavs/ai231-fil-supplemental-data`](https://huggingface.co/datasets/martinnavs/ai231-fil-supplemental-data) (revision 230c8b8); they are not in the DOI dataset.
- **Licence:** code is MIT ([`LICENSE`](LICENSE)). The model weights are in this repo (`out/vcm/hybrid-ctcwide-clsxl/`) and are also bound by the dataset's research-and-education terms ([`MODEL-CARD.md`](out/vcm/hybrid-ctcwide-clsxl/MODEL-CARD.md)).

## Reviewer checklist

| Item | Evidence |
| --- | --- |
| Public repo, one-command reproduction | `make reproduce`: 12/12 checks PASS, ~9 min on CPU ([`docs/REPRODUCE-HYBRID.md`](docs/REPRODUCE-HYBRID.md)) |
| Dataset licensed and citable | DOI 10.57967/hf/10723 |
| Training logs and final checkpoints committed | `train.log`, `metadata/loss_history.json`, `.pt` and ONNX under `out/vcm/hybrid-ctcwide-clsxl/{ctc-wide,cls-xl}/` |
| Pi 4 latency reproduced by the posted script | `scripts/soak_run.py`; results in [`docs/SOAK-TEST.md`](docs/SOAK-TEST.md) |
| Held-out test set with unseen speakers | ai231 test is speaker-disjoint; the real-accent holdout is one speaker (186 clips) |
| Baseline of comparable size | [`docs/BASELINES.md`](docs/BASELINES.md): ~1 M, ~4 M, ~10 M, ~14 M models |

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
