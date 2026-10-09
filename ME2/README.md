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

## Commands at a glance

Run from `ME2/`. On a laptop or server use `make` (after `module load uv && make sync`). On a Raspberry Pi `uv sync` and `make` do not work (torch CUDA pin, Python 3.10 pin), so use the lean environment and call Python directly.

| Task | Laptop / server | Raspberry Pi 4 |
| --- | --- | --- |
| One-time setup | `module load uv && make sync` | `uv venv --python 3.10 .venv-pi && uv pip install --python .venv-pi/bin/python -r requirements-pi.txt -r requirements-pi-test.txt`; then `export PYTHONPATH=src` |
| Streaming, live microphone | `make hybrid-stream` | the command below (needs `alsa-utils`; `arecord -l` shows the card number) |
| Streaming, replay a file | `make hybrid-stream HYBRID_SOURCE=clip.wav` | the same command with `--source clip.wav` and no `--mic-command` |
| UI dashboard (terminal 1), reachable on the LAN | `make app` (this machine only) or `make app-lan` (`0.0.0.0:8000`, no auth: trusted networks only) | `make app-lan` |
| Live pipeline into the UI (terminal 2) | `make app-pipeline-ctcwide` (the settings below plus `--emit-listening`, piped into `app.forward`; override the mic with `APP_MIC_COMMAND`, a wav with `APP_PIPELINE_SOURCE`) | the same; `make` works for these targets because they call `.venv/bin` directly (`APP_VENV_BIN=.venv-pi/bin` for the lean venv) |
| Holdout soak test | `make soak-run SOAK_DIR=soak/holdout-wake-gap-v1 SOAK_NAME=run SOAK_ARGS="--backend onnx --threads 1"` | `.venv-pi/bin/python scripts/soak_run.py --sessions soak/holdout-wake-gap-v1 --continuous --name rpi4 --wakeword-threshold 0.8 --cls-slot-threshold 0.6 --backend onnx --threads 1` |
| Tests | `make hybrid-test` | `.venv-pi/bin/python -m pytest -q tests/test_vcm_hybrid.py tests/test_vcm_streaming_endpointed.py tests/test_vcm_streaming_wakeword_gate.py tests/test_vcm_streaming_config.py tests/test_vcm_quartznet_heads.py tests/test_vcm_perturbation_plan.py tests/test_soak_run.py` |

`--emit-listening` makes the runner print a JSONL `{"event":"listening","state":"active"|"passive"}` record when the wake-word gate opens and closes, which the UI's listening indicator and music soft-pause follow ([`docs/STREAMING-CONTRACT.md`](docs/STREAMING-CONTRACT.md)). Without it stdout holds only trigger records.

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
Run one soak at a time and keep the Pi cool (`vcgencmd get_throttled` should read `0x0`). More, including the mic and the Pi gotchas: [`docs/RASPBERRY-PI.md`](docs/RASPBERRY-PI.md).

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

## UI music (for testing)

The UI's music panel plays real audio from `extras/music/` (git-ignored, so the audio never enters the repo). To try it, create the folder and add a few `.mp3`/`.wav`/`.flac`/`.ogg`/`.m4a` files:

```bash
mkdir -p extras/music && cp ~/Music/*.mp3 extras/music/    # then restart the UI: make app
```

File names become the track titles. With no files (or no `ffplay`) the panel stays silent and simulated. Playback starts at 30% volume; the wake word soft-pauses it and it resumes afterwards, except after a PAUSE or STOP command. Set `ME2_MUSIC_AUDIODEV` (an ALSA device, e.g. `plughw:CARD=CD002AUDIO,DEV=0`) to pick the speaker.

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

## Offline demo on a phone hotspot (Pi has no monitor)

The Pi runs headless: the UI and the live pipeline start as user services at boot, and the laptop shows the UI in a browser. `http://raspberrypi.local:8000` resolved from a laptop on the phone hotspot (checked); the UI needs no internet.

1. One-time, on the Pi: `./deploy/install-demo.sh` (enables `me2-ui` = `make app-lan` and `me2-pipeline` = `make app-pipeline-ctcwide`, plus linger so they start without a login). The mic is addressed by card name (`plughw:CARD=UACDemoV10,DEV=0`) because card numbers can change between boots; edit `deploy/systemd/me2-pipeline.service` for a different mic.
2. One-time: save the hotspot so the Pi auto-joins it:
   `nmcli device wifi connect "<ssid>" password "<pw>"`, then `nmcli connection modify "<ssid>" connection.autoconnect-priority 100`.
3. Demo: turn the hotspot on, power the Pi, join the laptop to the hotspot, open `http://raspberrypi.local:8000`. If the name does not resolve, use the Pi's IP (the phone's connected-devices list, or `nmap -sn <subnet>/24` from the laptop).
4. Debug: `journalctl --user -u me2-ui -u me2-pipeline -f`; restart with `systemctl --user restart me2-pipeline`. Run the services by hand only after `systemctl --user stop me2-ui me2-pipeline`, otherwise port 8000 and the mic are taken.

Not yet verified: a cold boot through the services, and the stable mic name capturing audio. The Pi has no battery clock: offline, its time after a cold boot is the last saved time, so alarm and timer demos that read the wall clock may be off. The UI has no authentication, so use a network you trust. Fallbacks if the hotspot is a problem (client isolation, `.local` not resolving): an Ethernet cable with a shared fixed address, or the Pi's own Wi-Fi access point.
