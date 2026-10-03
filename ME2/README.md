# ME2: spoken-command voice assistant

**BLUF.** A wake word ("sesame") followed by one of 19 spoken commands, decoded on-device by a 14.3 MB INT8 model trained on public data only.
On the speaker-disjoint test split it gets **98.8% command + slot accuracy** (human voices 91.8%). Behind a wake word with room reverb and noise it answers **81% of commands**
with 0.38 s median latency after the end of speech. On a **Raspberry Pi 4 (one core)** the answers are the same but each 0.25 s window takes 0.48 s, so it does not keep up live yet.
Branch `optionb-ctc-attention`; not promoted (`make app-pipeline*` still runs the earlier model). Start at [`docs/CURRENT-MODEL.md`](docs/CURRENT-MODEL.md).

## Architecture

Two QuartzNet-5x3 CTC encoders (40-mel features, 10 ms hop) read the same 2.5 s window; a DS-CNN wake word opens a 3 s listening period first.

- **Wide CTC** (4.2 M params, 4.2 MB INT8): grammar-constrained CTC prefix beam search over the 19-intent, 93-wording Option B grammar. If it accepts, that is the answer, slots included.
- **XL heads** (10 M params, 10.1 MB INT8): attention-pooled intent head and six slot heads. Used only when the CTC rejects, and only above confidence 0.8787 (slotted intents also need slot confidence 0.6).
- **Streaming:** wake word (threshold 0.8, polled every 0.05 s) -> endpointed policy (answer when the command ends) -> CTC, else classifier -> reject.

Detail: [`docs/CURRENT-MODEL.md`](docs/CURRENT-MODEL.md), [`docs/CTC-ATTENTION.md`](docs/CTC-ATTENTION.md), [`docs/OPTIONB-GRAMMAR-CONTRACT.md`](docs/OPTIONB-GRAMMAR-CONTRACT.md), [`docs/STREAMING-CONTRACT.md`](docs/STREAMING-CONTRACT.md).

## Performance

Whole clip, ai231 **test** split (3,823 in-scope clips, 76 out-of-scope), INT8 ONNX, one seed:

| | Hybrid | Wide CTC alone |
| --- | ---: | ---: |
| Command + slot accuracy | 98.8% | 95.0% |
| ... human voices only | 91.8% (214/233) | 84.5% |
| Out-of-scope false accept | 3.9% (3/76) | 0.0% |
| In-scope false reject | 1.0% | 4.9% |

Soak: 186 holdout commands + 16 out-of-scope clips behind a wake word, reverb and noise, 32.6 min, tuned settings, INT8 ONNX, one thread:

| | Raspberry Pi 4 | Server CPU |
| --- | ---: | ---: |
| Correct first trigger | 81.2% (151/186) | 81.2% (151/186) |
| Wrong actions / out-of-scope triggered / gap triggers | 9 / 3 of 16 / 0 | 9 / 3 of 16 / 0 |
| Latency after end of speech, median / p95 | 0.38 / 0.65 s | 0.38 / 0.65 s |
| Decode per window, mean / p95 | **443 / 764 ms** | 70 / 126 ms |
| Real-time factor, p95 (1 = live limit) | **3.21** | 0.53 |

Caveats: the holdout's human voices are one real Filipino speaker (50% whole-clip, so accent coverage is the weak spot); soak rooms and most noise clips overlap training; false wakes from ordinary speech are not measured.
All tables, seen-vs-unseen data and the soak history: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md), [`docs/SOAK-TEST.md`](docs/SOAK-TEST.md).

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

On a Pi `make soak-run` fails (`uv run` tries to install the x86 CUDA torch): call `.venv/bin/python scripts/soak_run.py --sessions soak/holdout-wake-gap-v1 --continuous --name rpi4 --wakeword-threshold 0.8 --cls-slot-threshold 0.6 --backend onnx --threads 1`.
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

About 94 min (wide) and 43 min (XL) on one A100. **Part of the training manifest cannot be rebuilt from the public datasets yet**: read [`docs/REPRODUCE-HYBRID.md`](docs/REPRODUCE-HYBRID.md) first.
Data: [`airimonda/ai231-me2-voice-commands`](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands) and [`martinnavs/ai231-fil-supplemental-data`](https://huggingface.co/datasets/martinnavs/ai231-fil-supplemental-data) (research and education only).

## Where everything else is

| Need | Read |
| --- | --- |
| Current model, data, recipe, code map, open items | [`docs/CURRENT-MODEL.md`](docs/CURRENT-MODEL.md) |
| Reproducing it, what is not public | [`docs/REPRODUCE-HYBRID.md`](docs/REPRODUCE-HYBRID.md) |
| Every benchmark table | [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) |
| Soak test, Raspberry Pi | [`docs/SOAK-TEST.md`](docs/SOAK-TEST.md), [`docs/RASPBERRY-PI.md`](docs/RASPBERRY-PI.md) |
| Experiment log, streaming commands | [`docs/AI231-FIL50.md`](docs/AI231-FIL50.md), [`docs/CTC-ATTENTION.md`](docs/CTC-ATTENTION.md) |
| Released files, model card | `out/vcm/hybrid-ctcwide-clsxl/` ([`MODEL-CARD.md`](out/vcm/hybrid-ctcwide-clsxl/MODEL-CARD.md)), `out/wakeword-sesame-ambient-rir-45m/`, `soak/holdout-wake-gap-v1/` |
| All docs; the earlier ~1M-parameter model and the TTS data pipeline | [`docs/README.md`](docs/README.md), [`docs/archive/`](docs/archive/README-EARLIER-MODEL.md) |
