# Running the hybrid on a Raspberry Pi (aarch64)

## The torch error and its fix

`torch==2.3.1+cu121` fails on a Pi because `pyproject.toml` pins `torch` and `torchaudio` to the PyTorch CUDA 12.1 index (`[tool.uv.sources]`, locked in `uv.lock`). Those wheels
are x86_64 with CUDA; a Pi is aarch64 with no CUDA. `uv sync` also tries to install the whole CosyVoice/TTS stack (`openai-whisper`, `pyworld`, `modelscope`, ...), which the streaming runtime
does not need and which may not build on ARM. So do not `uv sync` on the Pi: use the lean environment below.

**Disk:** about 3 GB free for a cold start (measured: the clone is 0.4 GB with its history, the `.venv-pi` environment 0.7 GB, uv's download cache and the Python 3.10 uv fetches add more, and a soak writes a few MB). On a 16 GB SD card with other projects on it, check `df -h /` first; an install that runs out of space fails with `No space left on device` while extracting wheels.

**Requirements:** 64-bit Raspberry Pi OS (a 32-bit OS has no torch wheels), Python 3.10 (the project pins `>=3.10,<3.11`; Bookworm ships 3.11, so let uv fetch 3.10), and for the microphone `alsa-utils`.

```bash
sudo apt install -y alsa-utils libsndfile1 libgomp1          # arecord, soundfile, OpenMP
curl -LsSf https://astral.sh/uv/install.sh | sh              # or: pip install uv
git clone <repo> && cd <repo>/ME2 && git checkout optionb-ctc-attention
uv venv --python 3.10 .venv-pi
uv pip install --python .venv-pi/bin/python -r requirements-pi.txt     # plain CPU torch/torchaudio from PyPI (aarch64), no +cu121
export PYTHONPATH=src
.venv-pi/bin/python -c "import torch, torchaudio, onnxruntime; print(torch.__version__, torchaudio.__version__, onnxruntime.__version__)"   # expect 2.3.1 (no +cu121)
```

`requirements-pi.txt` is the complete dependency list of the streaming path: `torch` and `torchaudio` (log-mel features), `numpy`, `onnxruntime` (both models run as INT8 ONNX), `soundfile`
(wav loading), and `numba` (JIT-compiles the grammar beam search; see below). The list was checked on x86_64 in an empty Python 3.10 environment holding only these five packages: the hybrid streaming command below fired the right intent on a soak clip.
**It has not been run on a Pi** (no Pi was available), so the first run there is the real check.

**Beam search backend.** The decoder uses numba when it imports and falls back to the same search in plain Python (about 5x slower than numba's ~100x gain over the original, identical output). The start-up line
`beam search: backend=numba warm-up=<ms> ms` on stderr says which one is running; the JIT compile takes about 4 s on the server and is **not measured on the Pi**. `ME2_BEAM_BACKEND=auto|numba|python|reference`
forces one (`reference` is the original loop, for A/B timing). Before trusting a Pi number, run `pytest tests/test_vcm_decoder_fast_beam.py` there: it checks the fast paths are bit-identical to the original on aarch64.

## Run the hybrid with the microphone

`arecord` captures from the Pi's ALSA device (`arecord -l` lists them; the card number is the first number in `plughw:<card>,<device>`). The pipeline needs **16 kHz, signed 16-bit, mono, raw PCM** on stdout:

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

Or `make hybrid-stream HYBRID_MIC_COMMAND="arecord -D plughw:3,0 -f S16_LE -r 16000 -c 1 -t raw -"` (needs `uv` and the full environment; on the Pi prefer the command above).
The default capture command is `arecord -f S16_LE -r 16000 -c 1 -t raw -` (default device); `--mic-command` replaces it. **The sample rate is 16000.** An `-r 160000` (a stray zero) would capture
at 160 kHz and the model would hear audio ten times too slow.

## Run the soak recording on the Pi

```bash
export PYTHONPATH=src
.venv-pi/bin/python scripts/soak_run.py --sessions soak/holdout-wake-gap-v1 --continuous --name rpi4 --backend onnx --threads 1 \
  --wakeword-threshold 0.8 --cls-slot-threshold 0.6
cat soak/holdout-wake-gap-v1/results/rpi4.md        # compare with results/tuned-cpu-onnx-int8-hybrid.md (server CPU)
```

Look at "decode per window" and the real-time factor. On the server one decode takes about 70 ms against the 250 ms stride (p95 factor 0.5); a factor above 1 means the Pi cannot keep up live,
and `--stride-s 0.5` or a CTC-only run (`--no-cls`) would be the next things to try. Method and server results: [`SOAK-TEST.md`](SOAK-TEST.md).

## Run the tests on a Pi

`make hybrid-test` needs the full `uv` environment, which does not install on a Pi. Use the lean one plus pytest:

```bash
uv pip install --python .venv-pi/bin/python -r requirements-pi-test.txt
export PYTHONPATH=src
.venv-pi/bin/python -m pytest -q tests/test_vcm_hybrid.py tests/test_vcm_streaming_endpointed.py tests/test_vcm_streaming_wakeword_gate.py \
  tests/test_vcm_streaming_config.py tests/test_vcm_quartznet_heads.py tests/test_vcm_perturbation_plan.py tests/test_soak_run.py     # the make target's files
.venv-pi/bin/python -m pytest -q tests/test_vcm_decoder_fast_beam.py                                                                 # fast paths are bit-identical to the original (about 4 minutes on a Pi 4)
```

Checked on a cold clone of the branch: the first set passes (162 tests, 4 deselected; about 1.5 minutes) and the second passes (130 tests). `requirements-pi-test.txt` includes `onnx`, which one export test needs. Tests that need the archived model runs skip when the files are missing.

## What a fresh clone does and does not contain

Tracked and enough for everything on this page: the hybrid model (`out/vcm/hybrid-ctcwide-clsxl/`), the wake word (`out/wakeword-sesame-ambient-rir-45m/`), and the 63 MB soak recording with its truth file (`soak/holdout-wake-gap-v1/`, check with `sha256sum -c SHA256SUMS`).
**Not in a clone:** the earlier models. `make vcmx-serve`, `make stream-wakeword` and the older tests default to them (`make app-pipeline` serves the hybrid); see [`ARCHIVED-CHECKPOINTS.md`](ARCHIVED-CHECKPOINTS.md) to restore them or point the variable at the hybrid.
**`make` and Python 3.10:** the Makefile pins `UV_PYTHON=/usr/bin/python3.10`. Raspberry Pi OS does not ship 3.10 (this Pi has 3.13), so use the `uv venv --python 3.10` route above, which downloads it.

## Soak-run gotchas on a Pi

- **One run at a time.** A second run halves the cores and corrupts the timings of both. Check first: `pgrep -fc '[v]cm.streaming --model'` must print `0`. A "wait, then run" loop plus a direct start once launched two copies.
- **Do not `pkill -f` a pattern that appears in your own command line**: it kills the calling shell (exit 144). Use the bracket trick (`pkill -f '[v]cm.streaming'`) or kill by pid.
- **No progress indicator.** `scripts/soak_run.py --continuous` writes `results/<name>.{md,json}` only when it finishes (about 7.5 minutes with the fast beam search, about 21 minutes with the original). The raw JSONL is also written only at the end; watch `top` to see it is alive.
- **Keep the Pi cool and powered**: `vcgencmd get_throttled` should read `0x0` before and after; a throttled run is slower and under-volted.
- **A quick check**: `--limit N` streams only the audio of the first N units (N=20 takes about 1 minute instead of 8, with the same counts for those units).
- **Options**: `--gate {wakeword,always}` (`always` runs with no wake word), `--beam-width N` (default 50), `--threads 1` (more threads slow the tiny wake-word network).
- `soak/holdout-wake-gap-v1/results/tuned-cpu-onnx-int8-hybrid.json` is committed but is not valid UTF-8 JSON (`file` reports `data`); read the `.md` beside it.

## Running the VCM benchmark's replay on the Pi

The results in [`VCM-BENCHMARK-RESULTS.md`](VCM-BENCHMARK-RESULTS.md) come from `scripts/vcm_benchmark/`. It needs its own small environment next to a clone of <https://github.com/airimonda/vcm-benchmark> at `<repo>/.vcm-benchmark/`:

```bash
git clone https://github.com/airimonda/vcm-benchmark .vcm-benchmark && cd .vcm-benchmark
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt       # numpy, soundfile, pandas, pyarrow, onnx, pytest
cp ../ME2/scripts/vcm_benchmark/*.py .
```

`sounddevice` cannot be imported on a Pi without PortAudio (`sudo apt install libportaudio2`), so `python benchmark.py` fails at start-up and one of the benchmark's tests (`test_sim_run`) fails; the replay scripts never import it. They stream through the
ME2 environment's Python (`ME2/.venv/bin/python`), so build that first (or edit `replay_bench.py` to use `.venv-pi`). The microphone capture for the acoustic runs is `arecord -D plughw:<card>,0 -f S16_LE -r 16000 -c 1 file.wav`.

## If torch still will not install

- `pip download torch==2.3.1` on the Pi should pick a `manylinux2014_aarch64` wheel; if it picks nothing, the OS or Python is the wrong architecture or version (`uname -m` must print `aarch64`; `python3.10 -V`).
- A different torch version also works for this path (no CUDA is used): any CPU build whose `torchaudio` pair matches; keep `numpy==1.26.4` and `onnxruntime==1.18.0`.
- Running without torch would need the log-mel front end rewritten in numpy; not done.
