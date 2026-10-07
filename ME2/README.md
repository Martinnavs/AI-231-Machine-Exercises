# ME2: spoken-command voice assistant

**BLUF.** A wake word ("sesame") followed by one of 19 spoken commands, decoded on-device by a 14.3 MB INT8 model trained on public data only.
On the speaker-disjoint test split it gets **98.8% command + slot accuracy** (human voices 91.8%). On the benchmark holdout (202 clips) it reaches **91.6% intent / 89.6% command accuracy**,
with the answer ready **about 0.5 s after the end of speech** on a **Raspberry Pi 4 using one core**. Through a real speaker and microphone it is lower (77% / 75% on a partial, uncalibrated run), and out-of-scope rejection is its weak spot (see Results).
Branch `optionb-ctc-attention-fast-beam`; not promoted (`make app-pipeline*` still runs the earlier model). Start at [`docs/CURRENT-MODEL.md`](docs/CURRENT-MODEL.md).

## Approach and architecture

Two QuartzNet-5x3 CTC encoders (40-mel features, 10 ms hop) read the same 2.5 s window; a DS-CNN wake word opens a 3 s listening period first.

- **Wide CTC** (4.2 M params, 4.2 MB INT8): grammar-constrained CTC prefix beam search (exact numba kernel, width 50) over the 19-intent, 93-wording Option B grammar. If it accepts, that is the answer, slots included.
- **XL heads** (10 M params, 10.1 MB INT8): attention-pooled intent head and six slot heads. Used only when the CTC rejects, and only above confidence 0.8787 (slotted intents also need slot confidence 0.6).
- **Streaming:** wake word (threshold 0.8, polled every 0.05 s) -> endpointed policy (answer when the command ends) -> CTC, else classifier -> reject. One thread, stride 0.25 s.

Detail: [`docs/CURRENT-MODEL.md`](docs/CURRENT-MODEL.md), [`docs/CTC-ATTENTION.md`](docs/CTC-ATTENTION.md), [`docs/OPTIONB-GRAMMAR-CONTRACT.md`](docs/OPTIONB-GRAMMAR-CONTRACT.md), [`docs/STREAMING-CONTRACT.md`](docs/STREAMING-CONTRACT.md), [`docs/BEAM-SEARCH.md`](docs/BEAM-SEARCH.md).

## Data, splits and evaluation

- **Train:** 28,858-row manifest rebuilt from public datasets ([`airimonda/ai231-me2-voice-commands`](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands), [`martinnavs/ai231-fil-supplemental-data`](https://huggingface.co/datasets/martinnavs/ai231-fil-supplemental-data)), with room reverb, noise and time-stretch augmentation.
- **Test:** the ai231 speaker-disjoint **test** split (3,823 in-scope clips, 76 out-of-scope), scored on whole clips.
- **Holdout:** the ai231 **holdout** (202 clips: 186 commands over 93 wordings + 16 out-of-scope; one real speaker plus synthetic voices). Used two ways: the soak (wake word + gap + command with reverb and noise, 32.6 min), and the VCM benchmark.
- **VCM benchmark** ([`airimonda/vcm-benchmark`](https://github.com/airimonda/vcm-benchmark)): the class benchmark's trials and scoring on a Pi 4, run as the final test before productionizing.

## Results

Whole clip, ai231 **test** split, INT8 ONNX, one seed:

| | Hybrid | Wide CTC alone |
| --- | ---: | ---: |
| Command + slot accuracy | 98.8% | 95.0% |
| ... human voices only | 91.8% (214/233) | 84.5% |
| Out-of-scope false accept | 3.9% (3/76) | 0.0% |
| In-scope false reject | 1.0% | 4.9% |

**VCM benchmark, final test (Pi 4, one thread)**, details and limits in [`docs/VCM-BENCHMARK-RESULTS.md`](docs/VCM-BENCHMARK-RESULTS.md):

| | Clean file replay (all 218 trials) | Laptop speaker 25 cm -> USB mic (raw; 108 of 218 trials, partial) |
| --- | ---: | ---: |
| Intent accuracy (19) | **91.6%** [87-95] | 76.9% |
| Command accuracy (93) | **89.6%** [85-93] | 75.0% |
| Slot exact match | 96.0% | 94.7% |
| Out-of-scope false accept | 25.0% (4/16) | 11.1% (1/9) |
| False wake (no wake word) | 12.5% (2/16) | 0.0% (0/4) |
| Latency after end of speech, p50 / p95 | 0.10 / 0.47 s | 0.11 / 0.56 s |
| Inference per decision, mean / p95 | 150 / 298 ms | 153 / 295 ms |

Read it with care: the clean replay is the model's honest best case; the acoustic run failed the benchmark's own mic check (SNR 4 dB, mic gain at maximum), used one laptop and one mic, and covers the first 28 minutes only.
A plain gain or an AGC does not recover the loss; a per-band EQ map recovers about half, so the loss is the speaker and mic frequency response, not level or distance.
Weak spots that a better room will not fix: out-of-scope false accepts, false wakes with no wake word, and real voices (85% intent accuracy against 97% for synthetic). Latency is algorithmic and was not measured live.

**Soak on a Pi 4** (186 holdout commands + 16 out-of-scope clips behind a wake word, reverb and noise, 32.6 min; beam 50, wake word 0.8, slot gate 0.6, INT8 ONNX, one thread, stride 0.25 s):

| | Pi 4, fast beam search | Pi 4, original beam search |
| --- | ---: | ---: |
| Estimated live latency after end of speech, median / p95 | **0.48 / 0.92 s** | 0.99 / 1.49 s (lower bound: it falls behind) |
| Compute per 0.25 s window, mean / p95 / max | 100 / 239 / 305 ms | 481 / 802 / 1,654 ms |
| Real-time factor, p95 (1 = live limit) | 0.96 | 3.21 |
| Correct first trigger (right intent and slot) | 81.2% (151/186) | 81.2% (151/186) |
| Wrong action / missed | 9 (4.8%) / 26 (14.0%) | same |
| Out-of-scope triggered / ambient-noise triggers | 3 of 16 / 0 | same |

The fast Pi keeps up on average and falls briefly behind on the slowest windows. The plain-language breakdown (misses, wrong actions, pure CTC without the classifier, no wake word at all, noise and out-of-scope),
the latency definitions and the caveats are in [`docs/SOAK-TEST.md`](docs/SOAK-TEST.md); every table and the seen-vs-unseen data are in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

## Serve

```bash
module load uv && make sync                                   # once (Raspberry Pi: docs/RASPBERRY-PI.md, `uv sync` fails on aarch64)
make hybrid-decode HYBRID_WAV="clip1.wav clip2.wav"           # decode files: JSON per clip, which model answered
make hybrid-stream                                            # live microphone: wake word, then the command
make hybrid-stream HYBRID_SOURCE=path/to/recording.wav        # replay a file
```

On a Pi: `uv venv --python 3.10 .venv-pi && uv pip install --python .venv-pi/bin/python -r requirements-pi.txt`, then the streaming command with
`--mic-command "arecord -D plughw:3,0 -f S16_LE -r 16000 -c 1 -t raw -"` ([`docs/RASPBERRY-PI.md`](docs/RASPBERRY-PI.md)). Python:

```python
from me2_voicegen.vcm.hybrid import HybridDecoder, load_hybrid_part
from me2_voicegen.vcm.optionb.grammar import OPTIONB_GRAMMAR
from me2_voicegen.common.features import LogMelFeatureExtractor
H = "out/vcm/hybrid-ctcwide-clsxl"
decoder = HybridDecoder(load_hybrid_part(f"{H}/ctc-wide/export/vcm_model.int8.onnx", "ctc"),
                        load_hybrid_part(f"{H}/cls-xl/export/vcm_heads.int8.onnx", "cls"),
                        LogMelFeatureExtractor(), OPTIONB_GRAMMAR, cls_threshold=0.8787)
decision, trace = decoder.decode(wav)   # wav: mono 16 kHz float tensor in [-1, 1]; decision.intent is None = rejected
```

## Test

```bash
make hybrid-test                                              # hybrid decoder, streaming policy, wake-word gate, soak scorer
make soak-run SOAK_DIR=soak/holdout-wake-gap-v1 SOAK_NAME=run SOAK_ARGS="--backend onnx --threads 1"   # the soak (the 63 MB recording is in the repo)
```

On a Pi `make soak-run` fails (`uv run` tries to install the x86 CUDA torch): call (from `ME2/`, with `export PYTHONPATH=src`, in the lean Pi environment from [`docs/RASPBERRY-PI.md`](docs/RASPBERRY-PI.md)) `.venv-pi/bin/python scripts/soak_run.py --sessions soak/holdout-wake-gap-v1 --continuous --name rpi4 --wakeword-threshold 0.8 --cls-slot-threshold 0.6 --backend onnx --threads 1` (needs numba, in `requirements-pi.txt`; tests on a Pi: `requirements-pi-test.txt`, [`docs/RASPBERRY-PI.md`](docs/RASPBERRY-PI.md); keep the Pi cool: `vcgencmd get_throttled` should read `0x0`; run one soak at a time, see the Pi doc).
Whole-clip scoring and replays: [`docs/CURRENT-MODEL.md`](docs/CURRENT-MODEL.md) ("Evaluating"). Method: [`docs/SOAK-TEST.md`](docs/SOAK-TEST.md).

## Train

```bash
uv run python -m me2_voicegen.vcm.train --device cuda:0 --manifest out/conversions/v2/ai231-fil50-supp/manifest.csv \
  --preset quartznet5x3-wide-heads \        # or quartznet5x3-xl-heads
  --seed 0 --p-rir 0.7 --p-timestretch 0.25 --p-noise 0.5 --p-babble 0.15 --noise-source dataset \
  --perturbation-plan table --dump-plan --skip-prenoised --noise-random-offset \
  --onecycle-epochs 60 --max-epochs 70 --patience 20 --max-minutes 135 --out-dir out/vcm/<run> --license-note "<terms of your data>"
uv run python -m me2_voicegen.vcm.export_onnx --checkpoint <ckpt> --out-dir <dir> --manifest <manifest>   # add --heads-only for the XL heads; INT8 included
```

About 94 min (wide) and 43 min (XL) on one A100. The training manifest rebuilds exactly from the public datasets (28,858 of 28,858 rows, checked); read [`docs/REPRODUCE-HYBRID.md`](docs/REPRODUCE-HYBRID.md) first.
Data: [`airimonda/ai231-me2-voice-commands`](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands) and [`martinnavs/ai231-fil-supplemental-data`](https://huggingface.co/datasets/martinnavs/ai231-fil-supplemental-data) (research and education only).

## Where everything else is

| Need | Read |
| --- | --- |
| Current model, data, recipe, code map, open items | [`docs/CURRENT-MODEL.md`](docs/CURRENT-MODEL.md) |
| Reproducing it, what is not public | [`docs/REPRODUCE-HYBRID.md`](docs/REPRODUCE-HYBRID.md) |
| Every benchmark table | [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) |
| Soak test (full tables), Raspberry Pi | [`docs/SOAK-TEST.md`](docs/SOAK-TEST.md), [`docs/RASPBERRY-PI.md`](docs/RASPBERRY-PI.md) |
| Final pre-production test (VCM benchmark on a Pi 4, laptop speaker + USB mic, AGC check) | [`docs/VCM-BENCHMARK-RESULTS.md`](docs/VCM-BENCHMARK-RESULTS.md), [`scripts/vcm_benchmark/`](scripts/vcm_benchmark/README.md) |
| Experiment log, streaming commands | [`docs/AI231-FIL50.md`](docs/AI231-FIL50.md), [`docs/CTC-ATTENTION.md`](docs/CTC-ATTENTION.md) |
| Released files, model card | `out/vcm/hybrid-ctcwide-clsxl/` ([`MODEL-CARD.md`](out/vcm/hybrid-ctcwide-clsxl/MODEL-CARD.md)), `out/wakeword-sesame-ambient-rir-45m/`, `soak/holdout-wake-gap-v1/` |
| All docs; the earlier ~1M-parameter model and the TTS data pipeline | [`docs/README.md`](docs/README.md), [`docs/archive/`](docs/archive/README-EARLIER-MODEL.md) |
