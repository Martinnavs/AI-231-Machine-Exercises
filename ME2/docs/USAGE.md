# Usage: commands, serving, tests, training

Everything the README leaves out. Run from `ME2/`. The one-command reproduction (`make reproduce`) is in [`REPRODUCE-HYBRID.md`](REPRODUCE-HYBRID.md) section 4; the numbers are in [`BASELINES.md`](BASELINES.md) and [`BENCHMARKS.md`](BENCHMARKS.md).

## Commands at a glance

Run from `ME2/`. On a laptop or server use `make` (after `module load uv && make sync`). On a Raspberry Pi `uv sync` and `make` do not work (torch CUDA pin, Python 3.10 pin), so use the lean environment and call Python directly.

| Task | Laptop / server | Raspberry Pi 4 |
| --- | --- | --- |
| One-time setup | `module load uv && make sync` | `uv venv --python 3.10 .venv-pi && uv pip install --python .venv-pi/bin/python -r requirements-pi.txt -r requirements-pi-test.txt`; then `export PYTHONPATH=src` |
| Streaming, live microphone | `make hybrid-stream` | the command below (needs `alsa-utils`; `arecord -l` shows the card number) |
| Streaming, replay a file | `make hybrid-stream HYBRID_SOURCE=clip.wav` | the same command with `--source clip.wav` and no `--mic-command` |
| UI dashboard, live pipeline into it ([`UI-DEMO.md`](UI-DEMO.md)) | `make app-lan`, then `make app-pipeline-ctcwide` in a second terminal | the same |
| Holdout soak test | `make soak-run SOAK_DIR=soak/holdout-wake-gap-v1 SOAK_NAME=run SOAK_ARGS="--backend onnx --threads 1"` | `.venv-pi/bin/python scripts/soak_run.py --sessions soak/holdout-wake-gap-v1 --continuous --name rpi4 --wakeword-threshold 0.8 --cls-slot-threshold 0.6 --backend onnx --threads 1` |
| Tests | `make hybrid-test` | `.venv-pi/bin/python -m pytest -q tests/test_vcm_hybrid.py tests/test_vcm_streaming_endpointed.py tests/test_vcm_streaming_wakeword_gate.py tests/test_vcm_streaming_config.py tests/test_vcm_quartznet_heads.py tests/test_vcm_perturbation_plan.py tests/test_soak_run.py` |

The Pi streaming command (the same settings as `make hybrid-stream`, plus `--ort-threads 1 --log-timing`; capture is 16 kHz, signed 16-bit, mono):

```bash
.venv-pi/bin/python -m me2_voicegen.vcm.streaming \
  --model out/vcm/hybrid-ctcwide-clsxl/ctc-wide --backend onnx --onnx-variant int8 --ort-threads 1 \
  --grammar optionb --threshold=-0.1 --required-command-margin 4.0 --beam-width 50 \
  --gate wakeword --policy endpointed --gate-period 3 --hold-ms 200 --stable-strides 1 \
  --wakeword-model out/wakeword-sesame-ambient-rir-45m --wakeword-backend onnx --wakeword-threshold 0.8 --wakeword-poll-s 0.05 \
  --cls-model out/vcm/hybrid-ctcwide-clsxl/cls-xl/export/vcm_heads.int8.onnx --cls-threshold 0.8787 --cls-slot-threshold 0.6 \
  --log-periods --log-timing \
  --source mic --mic-command "arecord -D plughw:3,0 -f S16_LE -r 16000 -c 1 -t raw -"
```

The soak test takes about 8 minutes on a Pi 4 (7 min 30 s with the fast beam search; add `--limit 20` for a 1-minute check) and writes `soak/holdout-wake-gap-v1/results/<name>.md` only when it finishes.
Run one soak at a time and keep the Pi cool (`vcgencmd get_throttled` should read `0x0`). More, including the mic and the Pi gotchas: [`RASPBERRY-PI.md`](RASPBERRY-PI.md).

## Serve

```bash
module load uv && make sync                                   # once (Raspberry Pi: RASPBERRY-PI.md, `uv sync` fails on aarch64)
make hybrid-decode HYBRID_WAV="clip1.wav clip2.wav"           # decode files: JSON per clip, which model answered
make hybrid-stream                                            # live microphone: wake word, then the command
make hybrid-stream HYBRID_SOURCE=path/to/recording.wav        # replay a file
```

On a Pi: `uv venv --python 3.10 .venv-pi && uv pip install --python .venv-pi/bin/python -r requirements-pi.txt`, then the streaming command with
`--mic-command "arecord -D plughw:3,0 -f S16_LE -r 16000 -c 1 -t raw -"` ([`RASPBERRY-PI.md`](RASPBERRY-PI.md)). Python:

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

On a Pi `make soak-run` fails (`uv run` tries to install the x86 CUDA torch): call (from `ME2/`, with `export PYTHONPATH=src`, in the lean Pi environment from [`RASPBERRY-PI.md`](RASPBERRY-PI.md)) `.venv-pi/bin/python scripts/soak_run.py --sessions soak/holdout-wake-gap-v1 --continuous --name rpi4 --wakeword-threshold 0.8 --cls-slot-threshold 0.6 --backend onnx --threads 1` (needs numba, in `requirements-pi.txt`; tests on a Pi: `requirements-pi-test.txt`, [`RASPBERRY-PI.md`](RASPBERRY-PI.md); keep the Pi cool: `vcgencmd get_throttled` should read `0x0`; run one soak at a time, see the Pi doc).
Whole-clip scoring and replays: [`CURRENT-MODEL.md`](CURRENT-MODEL.md) ("Evaluating"). Method: [`SOAK-TEST.md`](SOAK-TEST.md).

## Train

```bash
# --preset quartznet5x3-wide-heads, or quartznet5x3-xl-heads for the classifier
uv run python -m me2_voicegen.vcm.train --device cuda:0 --manifest out/conversions/v2/ai231-fil50-supp/manifest.csv \
  --preset quartznet5x3-wide-heads \
  --seed 0 --p-rir 0.7 --p-timestretch 0.25 --p-noise 0.5 --p-babble 0.15 --noise-source dataset \
  --perturbation-plan table --dump-plan --skip-prenoised --noise-random-offset \
  --onecycle-epochs 60 --max-epochs 70 --patience 20 --max-minutes 135 --out-dir out/vcm/<run> --license-note "<terms of your data>"
uv run python -m me2_voicegen.vcm.export_onnx --checkpoint <ckpt> --out-dir <dir> --manifest <manifest>   # add --heads-only for the XL heads; INT8 included
```

About 94 min (wide) and 43 min (XL) on one A100. The training manifest rebuilds exactly from the public datasets (28,858 of 28,858 rows, checked); read [`REPRODUCE-HYBRID.md`](REPRODUCE-HYBRID.md) first.
The same recipe in one command (about 2.5 h on one A100, never GPU 6 on the shared server): `make reproduce REPRO_TRAIN=1 REPRO_GPU=N`.
Data: [`airimonda/ai231-me2-voice-commands`](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands) (DOI 10.57967/hf/10723) and, for the gap-fill and numeral-wording clips, [`martinnavs/ai231-fil-supplemental-data`](https://huggingface.co/datasets/martinnavs/ai231-fil-supplemental-data) (research and education only).
