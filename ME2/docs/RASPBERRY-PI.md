# Running the hybrid on a Raspberry Pi (aarch64)

## The torch error and its fix

`torch==2.3.1+cu121` fails on a Pi because `pyproject.toml` pins `torch` and `torchaudio` to the PyTorch CUDA 12.1 index (`[tool.uv.sources]`, locked in `uv.lock`). Those wheels
are x86_64 with CUDA; a Pi is aarch64 with no CUDA. `uv sync` also tries to install the whole CosyVoice/TTS stack (`openai-whisper`, `pyworld`, `modelscope`, ...), which the streaming runtime
does not need and which may not build on ARM. So do not `uv sync` on the Pi: use the lean environment below.

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
(wav loading). The list was checked on x86_64 in an empty Python 3.10 environment holding only these five packages: the hybrid streaming command below fired the right intent on a soak clip.
**It has not been run on a Pi** (no Pi was available), so the first run there is the real check.

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

## If torch still will not install

- `pip download torch==2.3.1` on the Pi should pick a `manylinux2014_aarch64` wheel; if it picks nothing, the OS or Python is the wrong architecture or version (`uname -m` must print `aarch64`; `python3.10 -V`).
- A different torch version also works for this path (no CUDA is used): any CPU build whose `torchaudio` pair matches; keep `numpy==1.26.4` and `onnxruntime==1.18.0`.
- Running without torch would need the log-mel front end rewritten in numpy; not done.
