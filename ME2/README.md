# ME2 — Spoken-command voice assistant (grammar-constrained VCM + wakeword DS-CNN)

A Raspberry-Pi-targeted, zero-cloud spoken-command system: a DS-CNN wake-word detector
("computer") gates a ~1M-parameter grammar-constrained CTC acoustic model (the "VCM") that
decodes speech into one of a fixed set of intents/slots. The training data for both models is
synthetic-plus-real audio produced by a CosyVoice2 zero-shot TTS / voice-conversion pipeline that
this repo also owns end to end.

This repo started (see "Setup: the four `make` steps, in order" below) as a Phase-1 spike proving
CosyVoice2 zero-shot voice-cloning synthesis runs end-to-end on this HPC node (3x idle
A100-SXM4-40GB, no root/sudo), and grew from there into the full pipeline described here —
training data generation, the VCM model and grammar/decode, the wakeword DS-CNN, and a live
streaming runtime. That history is still visible in the section ordering below (synthesis setup
first, everything built on top of it after); the "VCM toy" section onward is the current state of
the project, not a later add-on to a still-scoped-down spike.

> **New here?** Read the process documentation first: an end-to-end, human-readable
> walkthrough of the whole project — raw audio → TTS/voice-conversion → dataset build →
> CTC training → grammar decode → wakeword DS-CNN → ONNX export → live streaming — with
> the command for each stage, the modelling core (architecture, training recipe,
> grammar-to-trie compile, decode/rejection gating), and the real, honestly-framed
> results. It's committed in this repo as five paired documents:
> `docs/PROCESS-OVERVIEW.md` (pipeline hub), `docs/PROCESS-DATA-GENERATION.md`,
> `docs/PROCESS-VCM-MODEL.md`, `docs/PROCESS-WAKEWORD.md`, and
> `docs/PROCESS-STREAMING-SERVING.md`. Start at `PROCESS-OVERVIEW.md`.

## Project status (2026-09-26, branch `optionb-grammar-v2`)

| Area | Status | Key number |
|---|---|---|
| VCM (Option B grammar, `optiond` preset) | trained, calibrated | test exact-intent accuracy 97.04% pre-gate / 86.82% real-audio-scored post-gate (incomplete-prefix margin=4.0) |
| Wakeword DS-CNN | trained, wired into `ListeningGate` | val F1 0.997 (`_wakeword_`); fixed threshold 0.9 (no FAR/FRR calibration pipeline yet) |
| Streaming runtime | shipped | fp32 ONNX is the serving default (INT8 needs its own threshold re-tuning pass) |
| VCMX (combined VCM+wakeword export/serve) | shipped, tech-lead reviewed, approved | treatment 25,231 rows / control 21,055 rows, speaker-disjoint |
| `accent-balance-fil50` (50/50 Filipino/non-Filipino rebalance) | **complete, all criteria met** | wakeword Filipino-accent recall gap 13.0pts → 0.4pts; VCM exact accuracy 0.956 → 0.979 |

Everything in this table is landed and working end to end. `accent-balance-fil50`'s full run
(references-only path, per the decision in `docs/PROCESS-DATA-GENERATION.md`'s "Current status"
section) is complete; results in `docs/MLOPS-PROJECTS.md`'s Iteration 1. Promoting the new
checkpoints to production (`make vcmx-serve` / `make vcmx-serve-wakeword`) is a separate,
not-yet-made decision.

## Scope

**In scope for the original Phase-1 spike** (this foundation is still exactly how synthesis
happens today — nothing below replaces it, later work only builds on top of it):
- Vendoring upstream CosyVoice at a pinned commit and getting it importable under `uv`.
- Downloading the CosyVoice2-0.5B checkpoint.
- A swappable synthesis interface (`Synthesizer` ABC + factory) with one real backend
  (`cosyvoice2`) wired through a thin CLI.
- Proving a single zero-shot synthesis call produces a real, non-silent, intelligible wav on
  the A100, with a fast unit-test suite plus one real end-to-end GPU test.

**Explicitly NOT in scope (later work):**
- Batch/looped generation over a **phrase list** — i.e. multiple *texts* in one run. Still not
  supported: `generate_personas.py` (below) synthesizes exactly one text per run, never a list of
  texts.
- RIR (reverb) augmentation of generated clips at dataset-build time — still train-time-only, via
  `common/augment.py`.
- A second backend implementation (the interface is designed to make one easy to add later —
  see `docs/adding-a-tts-backend.md` — but none is implemented here).

**Added after the initial spike, and since grown into the current pipeline:**
- Batching one text over multiple *personas* (multiple reference voices/manifests) in a single
  run, sharing one backend construction — see "Batch generation over personas" below. This is a
  different axis from the phrase-list batching above (personas vary, not text) and does not
  contradict it.
- The "computer"-only wake-word/KWS dataset-build pipeline (`src/me2_voicegen/wakeword/`) — real
  + voice-converted + noise-augmented positives, phonetic adversaries, scrubbed general negative
  speech, and synthetic silence, assembled into a group-disjoint 70/20/10 split. See
  `docs/WAKEWORD-DATASET-CONTRACT.md` (schema/licensing) and
  `.scratch/wakeword-computer-dataset/HANDOFF.md` (build narrative + current state). Note: this
  dataset is CC-BY-NC-SA-4.0-encumbered (non-commercial, share-alike) once its noise-augmented
  rows are included — see the contract doc's licensing section before using or redistributing it.
  A DS-CNN has since been trained on this dataset and wired into the live streaming runtime as a
  `ListeningGate` — see "Wakeword DS-CNN" below, this is no longer a follow-up.
- **A real DS-CNN wake-word model** (`src/me2_voicegen/wakeword/model.py`), trained, benchmarked,
  and wired into the streaming runtime's `ListeningGate` seam — see "Wakeword DS-CNN" below.
- **Option B**: a second, real-dataset spoken-command grammar (19 intents) with its own trained
  VCM checkpoint, a combined VCM+wakeword export/serve pipeline ("VCMX"), and a calibrated
  incomplete-prefix rejection gate — see "Option B spoken-command grammar", "VCMX" and the
  "Streaming inference" sections below.
- **`accent-balance-fil50`** (complete): a 50/50 Filipino/non-Filipino speaker rebalancing pass
  for both the VCM and wakeword datasets — full run done, all pre-committed criteria met (see
  "Project status" above and `docs/PROCESS-DATA-GENERATION.md`).

See `docs/raw_requirements/sources.md`, `docs/raw_requirements/potential_model_approach.md`,
`docs/raw_requirements/voice_generation_approach.md`, and
`docs/raw_requirements/streaming_approach.md` for the original planning context (read-only — not
updated by this spike).

## Prerequisites

`uv` is **not** on `$PATH` by default on this node — you must load it first:

```bash
module load uv
```

(The project's `Makefile` itself doesn't depend on this — it resolves `/opt/uv/uv` directly if
`uv` isn't found on `$PATH` — but you need `uv` on your own shell's `$PATH` for anything you run
by hand outside `make`, e.g. `uv run pytest -m slow` below.)

No Docker, no root/sudo required or used anywhere in this setup.

## Setup: the four `make` steps, in order

Run from `ME2/`:

```bash
cd ME2
make sync            # 1
make vendor           # 2
make download-model   # 3
make generate         # 4
```

1. **`make sync`** — `uv sync`. Creates `.venv` on Python 3.10 (pinned via `.python-version`
   and `UV_PYTHON`), installs the trimmed dependency set from `pyproject.toml` (see below),
   including `torch==2.3.1+cu121`/`torchaudio==2.3.1+cu121` from the explicit
   `pytorch-cu121` index. No manual `pip install` steps. Uses `/tmp/uv-cache-$USER` for uv's
   cache (not `$HOME`, which is on JuiceFS and slow for many-small-files operations like a
   package cache).

2. **`make vendor`** — clones upstream
   [`FunAudioLLM/CosyVoice`](https://github.com/FunAudioLLM/CosyVoice) (Apache-2.0) into
   `ME2/vendor/CosyVoice` at pinned SHA `074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc`, **recursively**
   (the `third_party/Matcha-TTS` submodule, MIT-licensed, pinned at
   `dd9105b34bf2be2230f4aa1e4769fb586a3c824e`, is mandatory — a non-recursive clone silently
   omits it and the failure only surfaces later, as `ModuleNotFoundError: matcha` at model
   construction time, not at clone time). Idempotent: re-running when already at the pinned SHA
   is a no-op ("vendor/CosyVoice already at pinned SHA ..., skipping clone"). Upstream ships no
   `setup.py`/`pyproject.toml` — it's not pip-installable, which is why this is a `git clone`
   into a `sys.path`-shimmed vendor directory rather than a dependency.

3. **`make download-model`** — `uv run python -m me2_voicegen.download_model`. Downloads the
   CosyVoice2-0.5B checkpoint (Apache-2.0) into `ME2/models/CosyVoice2-0.5B/`: **19 files,
   ~4.9GB total** (`llm.pt` 2.02GB, `CosyVoice-BlankEN/model.safetensors` 988MB,
   `speech_tokenizer_v2.onnx` 496MB, `flow.pt` 451MB, plus `cosyvoice2.yaml`, `hift.pt`,
   `campplus.onnx`, and the rest of the `CosyVoice-BlankEN/` subdirectory). After downloading it
   verifies the file manifest and fails loudly, naming exactly which entries are missing, if
   incomplete. Idempotent — a second run detects a complete manifest and skips the download
   entirely (no network call; confirmed at ~0.4s on a warm check).

   **Intended path:** default source is `modelscope.snapshot_download('iic/CosyVoice2-0.5B', ...)`;
   pass `--source huggingface` to use `huggingface_hub.snapshot_download('FunAudioLLM/CosyVoice2-0.5B', ...)`
   instead:

   ```bash
   uv run python -m me2_voicegen.download_model --source huggingface
   ```

   **What actually happened on this node:** modelscope.cn was unreliable from here — a real
   attempt via the default source timed out after 7.5 minutes (`ReadTimeoutError`) and a retry
   was still crawling at 28MB transferred when abandoned. A `--source huggingface` attempt did
   complete (fast, ~12s, thanks to `huggingface_hub`'s content-addressed dedup) but the model's
   verified state on disk was ultimately established via a **manual out-of-band transfer**: the
   user downloaded the same checkpoint independently and uploaded it via FileZilla/SFTP directly
   into `ME2/models/CosyVoice2-0.5B/`. `make download-model`/`download_model.py`'s manifest check
   and idempotency logic both work correctly regardless of how the files got there — it only
   inspects what's on disk. If you're re-running this from scratch on a similarly flaky network
   path, don't be surprised if you need to fall back to a manual transfer too; the automated path
   is the intended default, not a guaranteed one from this node.

   Note: the manifest does **not** include `spk2info.pt` despite it appearing as a
   `CosyVoiceFrontEnd.__init__` parameter — it isn't shipped by either upstream registry and is
   an optional runtime artifact (created only when a speaker embedding is cached via
   `add_zero_shot_spk`/`save_spkinfo`); requiring it would make the manifest permanently
   unsatisfiable.

4. **`make generate`** — `uv run python -m me2_voicegen.generate_sample --backend $(BACKEND) --text "$(TEXT)"`
   (`BACKEND ?= cosyvoice2`, `TEXT ?=` — an empty `TEXT` falls through to `generate_sample.py`'s
   own natural-sentence default). Runs one real zero-shot synthesis call against the vendored
   CosyVoice2 code and the downloaded weights, writes a `.wav` to `ME2/out/`, and prints:

   ```
   output path: ME2/out/sample_<timestamp>.wav
   duration: 4.160s
   sample rate: 24000
   wall-clock synthesis time: 6.745s
   RTF: 1.621
   peak GPU memory: 2779971584 bytes
   ```

   Verified on the real A100: the synthesized English text ("Hey computer, could you please turn
   on the lights in the living room?"), cloned from the vendored Chinese zero-shot prompt asset
   (`asset/zero_shot_prompt.wav` + its transcript), was independently checked non-silent
   (RMS ~0.058, non-zero frame count) and re-transcribed with Whisper to confirm intelligibility
   — near-exact match to the input text. `inference_zero_shot` (not the `inference_cross_lingual`
   fallback) handles the Chinese-prompt/English-text cross-lingual case correctly on its own.

   You'll see a harmless warning during this step:
   ```
   UserWarning: Specified provider 'CUDAExecutionProvider' is not in available provider names.
   ```
   This is expected — CosyVoice's own ONNX-based components (speech tokenizer, campplus) fall
   back to CPU because this project deliberately installs CPU-only `onnxruntime` rather than
   `onnxruntime-gpu` (see "Trimmed dependencies" below). It's a warning, not a failure; the
   overall pipeline still runs on the GPU (torch models load onto CUDA normally).

Override the backend or text at the command line, e.g.:

```bash
make generate TEXT="turn off the kitchen lights"
```

## The `--backend`/`--opt` CLI shape

`generate_sample.py` is a thin, backend-agnostic CLI — no backend-specific identifier (e.g.
`cosyvoice2`, `AutoModel`, `inference_zero_shot`, `cosyvoice`) appears anywhere in that file
except the `DEFAULT_BACKEND` constant's value. Dispatch goes entirely through
`synthesis.factory`. This is proven, not just asserted: `tests/test_generate_sample_cli.py` drives
a `FakeSynthesizer` with a deliberately unrelated constructor signature end-to-end through the
real CLI, and `test_smoke_fast_suite.py` asserts the real `cosyvoice` package never lands in
`sys.modules` from `--help` or `list_backends()` alone.

```bash
uv run python -m me2_voicegen.generate_sample \
  --backend cosyvoice2 \
  --text "some sentence" \
  --prompt-wav /path/to/reference.wav \
  --prompt-text "transcript of the reference clip" \
  --device auto \
  --opt fp16=true \
  --opt model_dir=/custom/model/dir
```

- `--backend` choices are built live from the factory's registry (`list_backends()`) — `--help`
  auto-updates as backends are added, no hardcoded choice list to keep in sync.
- Common flags (`--device` today) map to conventional constructor kwarg names and are only
  forwarded if the selected backend's `__init__` actually declares that parameter (checked via
  `inspect.signature`) — a backend that doesn't accept `device` simply doesn't receive it, no
  error.
- `--opt KEY=VALUE` (repeatable) is the escape hatch for anything backend-specific not covered by
  a common flag — passed through **unfiltered**, so an unknown key is a loud, immediate error
  rather than silently ignored. Values are coerced: `true`/`false` → `bool`, `none` → `None`,
  else `int`/`float`/`str`.
- Because the interface is genuinely swappable this way, a future backend (e.g. Piper, XTTS-v2)
  plugs in without touching this CLI file at all — see `docs/adding-a-tts-backend.md`.

## Batch generation over personas

`generate_personas.py` synthesizes **one text, once per persona**, from a JSON manifest of
personas — each persona is a reference (voice-cloning) wav plus its exact transcript — building
the backend exactly once and reusing it across the whole batch. This is the "batching over
personas" case described in Scope above; it does not add phrase-list (multiple-text) batching.

### Manifest schema

```json
{
  "personas": [
    {
      "name": "english_woman",
      "wav_path": "personas/english_woman.wav",
      "text": "Replace this with the exact transcript of personas/english_woman.wav."
    },
    {
      "name": "indian_man",
      "wav_path": "personas/indian_man.wav",
      "text": "Replace this with the exact transcript of personas/indian_man.wav."
    }
  ]
}
```

See `personas.example.json` (committed, placeholder paths/text only — no audio ships with this
repo) for the canonical shape. Fields, all required per entry:

- `name` — must match `^[A-Za-z0-9_-]{1,64}$`; unique within the manifest. This whitelist is what
  makes `name` safe to interpolate directly into the output filename (see below) — no path
  traversal or separator character can reach it.
- `wav_path` — the reference clip to clone the voice from. **Resolved relative to the manifest
  file's own directory, not the current working directory** — a manifest at
  `/some/where/personas.json` with `"wav_path": "personas/foo.wav"` always resolves to
  `/some/where/personas/foo.wav`, regardless of where you run the command from.
- `text` — the exact transcript of `wav_path`. Required (unlike `generate_sample.py`'s optional
  prompt text) because the manifest is the only source of a per-persona transcript; the backend
  needs it for every zero-shot call.

### Running it

```bash
make generate-personas MANIFEST=personas.json TEXT="turn off the kitchen lights"
```

or directly:

```bash
uv run python -m me2_voicegen.generate_personas \
  --manifest personas.json \
  --text "turn off the kitchen lights" \
  --backend cosyvoice2 \
  --out-dir out/ \
  --device auto \
  --opt fp16=true
```

`--manifest` is required; `--text` falls through to the same default carrier sentence as
`generate_sample.py` if omitted; `--backend`/`--out-dir`/`--device`/`--opt`/`-v` all mirror
`generate_sample.py`'s flags and semantics exactly (see "The `--backend`/`--opt` CLI shape" above).

Output files are named `sample_<persona-name>_<timestamp>.wav`, written to `ME2/out/` (or
`--out-dir`). All personas in one run share a **single** timestamp (captured once, after the
backend is constructed, before the per-persona loop starts) — this is what makes it possible to
tell which wavs belong to the same batch run.

Sample output (two-persona manifest):

```
[english_woman] output path: ME2/out/sample_english_woman_1789259028225.wav
[english_woman] duration: 4.160s
[english_woman] sample rate: 24000
[english_woman] wall-clock synthesis time: 6.745s
[english_woman] RTF: 1.621
[indian_man] output path: ME2/out/sample_indian_man_1789259028225.wav
[indian_man] duration: 4.032s
[indian_man] sample rate: 24000
[indian_man] wall-clock synthesis time: 6.201s
[indian_man] RTF: 1.538
--- summary ---
[english_woman] ok
[indian_man] ok
total wall-clock time: 12.946s
peak GPU memory: 2779971584 bytes
```

### Partial-failure and exit-code semantics

A single persona's synthesis failure does **not** abort the batch: the remaining personas still
render, and a per-persona `ok`/`failed` summary line is printed for every persona regardless of
outcome. The process exits `1` if *any* persona failed, `0` only if all succeeded — so a batch run
is safe to script (check the exit code) without having to parse the summary text.

### Fail-fast before model load

The manifest is fully parsed and validated — `name`/`wav_path`/`text` present, `name` unique and
matching the whitelist regex, and every `wav_path` resolving to an existing file — **before the
backend is constructed at all**. An invalid manifest (missing field, duplicate name, bad name
pattern, or a nonexistent reference wav) exits `1` immediately, naming the offending entry, with
**zero** synthesizer construction and no multi-GB model load — you don't wait through a weights
load only to find out entry 2 of 5 has a typo.

### Timbre is entirely a function of the reference clip

Distinct personas require genuinely distinct reference clips. CosyVoice2 derives the speaker
embedding/timbre from the reference wav (`wav_path`) itself, via zero-shot voice cloning — there is
no instruction-text or accent-text mechanism anywhere in this codebase (no `inference_instruct2`
wiring; see Non-Goals) that changes speaker identity on a single shared base clip. Pointing two
manifest entries at the same `wav_path` produces two identically-timbred outputs under different
names — that's expected mechanical behavior (used deliberately in this project's own tests and
manual verification runs, where only one real reference wav exists), not a bug. If you want two
persons to actually sound different, you need two actually-different reference recordings.

### Reference-clip consent/licensing, and keeping real manifests out of git

Cloning a voice from a reference clip has real consent/licensing implications — only use reference
clips you have the rights to use for this purpose. This repo does not commit any audio anywhere,
and `personas.example.json` is a placeholder manifest only (fake paths, placeholder transcript
text) — it exists purely to document the manifest shape above, not to be run as-is. If you build a
real manifest with real persona clips, keep both the manifest (conventionally `ME2/personas.json`,
the Makefile's default `MANIFEST` value) and the audio directory (conventionally `ME2/personas/`)
out of version control the same way `out/`, `vendor/`, and `models/` already are — treat them as
local, untracked artifacts, not repo content.

## Trimmed dependencies

`pyproject.toml`'s dependency list is **not** a copy of upstream CosyVoice's
`requirements.txt`. The following are deliberately excluded, because this spike only needs
`AutoModel(...)` + `inference_zero_shot(...)` on a single node with no training, no
TensorRT/DeepSpeed acceleration, and no web UI:

| Excluded | Why |
|---|---|
| `deepspeed==0.15.1` | training-only (`bin/train.py`, `utils/train_utils.py`) |
| `tensorrt-cu12`, `tensorrt-cu12-bindings`, `tensorrt-cu12-libs` | needs `load_trt=True`, which defaults to `False` and is never set here |
| `onnxruntime-gpu==1.18.0` | replaced with CPU `onnxruntime==1.18.0` — CosyVoice's ONNX components (speech tokenizer, campplus) fall back to CPU with a warning, not a failure, when handed `CUDAExecutionProvider` and it isn't available |
| `gradio`, `fastapi`, `fastapi-cli`, `uvicorn`, `grpcio`, `grpcio-tools` | webui/runtime-server only, unused by direct `AutoModel` inference |
| `tensorboard` | training helper |
| ~~`onnx`~~ | ~~training helper~~ — **re-added** for the unrelated `vcm-toy` feature's Task 06 (ONNX export/quantization/benchmark of the toy VCM CTC model), pinned `onnx==1.16.1`. This is a deliberate, documented reversal of the original CosyVoice-spike exclusion, not an oversight: `onnx` (the export/graph package) was never needed for CosyVoice's own `AutoModel`/`inference_zero_shot` inference path, but `vcm.export_onnx` (task 06) needs it to build/quantize an ONNX graph from the trained `vcm.model` checkpoint. See `docs/raw_requirements/` and `.scratch/vcm-toy/tickets/06-onnx-export-benchmark.md`. |
| `gdown`, `wget` | **re-added** during this spike — see below |

**`gdown` and `wget` were re-added.** They looked like training-only helpers from an import grep
of `cosyvoice/` alone, but the real import chain (`cosyvoice.flow.flow_matching` →
`matcha.models.components.flow_matching` → `matcha.utils` → `matcha.utils.utils`) reaches them
at *module import time* via Matcha-TTS, not CosyVoice's own code. Re-added at their exact
upstream-pinned versions (`gdown==5.1.0`, `wget==3.2`) per the escape-hatch rule below —
this is the escape hatch working as designed, not a violation of the trimming decision. A plain
`setuptools<81` runtime dependency was added for the same reason (`lightning.fabric` does
`pkg_resources.declare_namespace(...)` at import time, and `setuptools>=81` removed
`pkg_resources`) — distinct from the `[tool.uv.extra-build-dependencies]` entry that only
affects `openai-whisper`'s isolated build environment.

**Escape-hatch rule for any future `ModuleNotFoundError`:** add back that *one* missing module,
at its exact `vendor/CosyVoice/requirements.txt`-pinned version, and log why in the relevant
ticket/PR. Do not upgrade, un-pin, or bulk-restore the whole upstream requirements file. If
`deepspeed` or `tensorrt` ever turn out to be genuinely required for real inference (as opposed to
training or accelerated-but-optional paths), stop and escalate to a human rather than silently
restoring them — those are the two packages the trimming decision was actually built around.

## Running the tests

```bash
make test                    # fast suite: no GPU, no vendor clone, no weights required
uv run pytest -m slow         # the one real end-to-end test: real GPU, real weights
```

`make test` runs `uv run pytest` with `addopts = "-m 'not slow'"`, so the default run excludes
the slow tests automatically. As of the last full run: **78 passed, 2 deselected**. The
fast suite is guaranteed not to touch the vendor clone, model weights, or a GPU —
`test_smoke_fast_suite.py` asserts the real `cosyvoice` package never lands in `sys.modules` as a
side effect of importing either CLI (`generate_sample` or `generate_personas`) or calling
`list_backends()`/`get_backend_class()`, and that `--help` exits 0 without it either for either
CLI (this was originally verified manually by renaming `vendor/` away; that test makes the same
guarantee an automatic regression check).

`uv run pytest -m slow -q` runs the two `@pytest.mark.slow` tests: **2 passed in ~55s** on the
A100. `tests/test_end_to_end_slow.py` drives the real factory → real `AutoModel` →
`inference_zero_shot` → `save_wav` against the real downloaded weights and the real GPU, and
asserts the output is a valid, non-silent 24kHz wav. No mocks of CosyVoice internals, `AutoModel`,
or ONNX Runtime anywhere in this test — if the real thing can't run, that's meant to surface as a
real failure, not be papered over. It skips (doesn't fail) with a clear message if
`ME2/models/CosyVoice2-0.5B/` isn't present/complete. The second slow test,
`tests/test_personas_end_to_end_slow.py`, is the persona-batch end-to-end GPU test: it builds a
2-entry manifest in a temp dir from the vendored `zero_shot_prompt.wav` asset, constructs the real
backend exactly once, and asserts two distinctly-named, non-silent 24kHz wavs from that single
model load. It shares the same skip-when-weights-absent pattern as the first slow test (also
skipping if the vendored asset itself is missing). Because both entries in that manifest point at
the same vendored reference wav under two different persona names, the two outputs are expected to
share identical timbre in this specific test — that's the mechanical single-clip case described
above, not a check of distinct-voice cloning, which requires real, distinct, user-supplied clips.

## Cleaning up

`make clean` removes `.venv vendor models out .pytest_cache` — note this deletes the ~4.9GB
model download and the vendor clone along with cheap-to-regenerate build artifacts, so
re-running setup after `make clean` means repeating steps 2-3 above (`make vendor`,
`make download-model`) too, not just `make sync`.

## VCM toy — toy Voice Command Model (CTC acoustic model + grammar decoder)

**What this is.** A toy, first-pass validation of a much larger target spec — a
Raspberry-Pi-deployable, zero-cloud voice command system built from a tiny CTC acoustic
model plus a grammar-constrained decoder. This pass deliberately used only data already on
disk (`out/conversions/v2/test_set/`, built by the conversion/dataset-assembly pipeline
described earlier in this README) to validate the approach end-to-end *before* any decision
to scale up to real/larger data. Nothing here should be read as a claim that the approach
generalizes beyond this toy scale — see "Known gaps" below.

The full technical contract (alphabet, text normalization, transcript-resolution rules,
feature/batch conventions, decoder I/O shape, license provenance) lives in
[`docs/VCM-CONTRACT.md`](docs/VCM-CONTRACT.md) — read that instead of this section for exact
shapes/columns; this section covers what was built, how to run it, and what the real results
were.

### Architecture

- **Acoustic model** (`src/me2_voicegen/vcm/model.py`): a MatchboxNet-style 1D
  time-channel-separable CNN CTC model, outputting logits over a 29-token character
  alphabet (blank + `a`-`z` + space + apostrophe). Note: the original spec text says "32
  tokens," but its own enumeration only ever lists 29 — this is a documented discrepancy,
  not a bug; we did not pad to 32. A `default` preset (~254k params, sized for this toy
  corpus) was actually trained; a `--preset spec-scale` config (~2.1M params, inside the
  original spec's stated 1.2M-2.5M range) was instantiated and size-checked, but never
  trained this pass.
- **Decoder** (`src/me2_voicegen/vcm/grammar.py`, `decoder.py`): a hand-written Python
  grammar/trie plus CTC prefix beam search — not an FST library. `kaldifst` was present in
  the venv (transitively) but deliberately not used; this was a design decision, not a
  missing dependency, and is noted as a possible future scale-up path. Two grammars are
  exposed, never merged:
  - `SPEC_GRAMMAR` — the user's BNF spec, compiled verbatim.
  - `TOY_GRAMMAR` — `SPEC_GRAMMAR` plus a labeled `TOY_ALIASES` overlay covering the 12 of
    the 20 dataset intents whose bare canonical phrase doesn't parse under the verbatim
    spec. Only 2 of those 12 (`MESSAGE`, `SET_REMINDER`) have literally no BNF rule at all;
    the other 10 have a real `$CMD_*` rule that requires a slot or different phrasing than
    the dataset's bare phrase provides (e.g. `$CMD_SET_ALARM` requires an `am`/`pm` slot,
    but the dataset's `ALARM` phrase is just "set alarm").
  - One genuine spec gap, not a bug: `$CMD_TEMPERATURE`'s BNF (`(set | adjust) (the)?
    temperature to $NUMBER (degrees)?`) has no up/down direction at all, so a
    SPEC_GRAMMAR-parsed temperature phrase can't be resolved to `TEMP_UP` vs `TEMP_DOWN`.
    Only the toy alias bare words ("warmer"/"cooler") are unambiguous. Fixing this would
    require a change to the BNF spec itself, not the decoder.
- **Training** (`train.py`) uses on-the-fly RIR convolution (RT60 sampled in [0.1, 0.5]s),
  additive noise (SNR sampled in [5, 25]dB), and SpecAugment — all per the original
  requirements.

### Reproducing

```bash
cd ME2
make vcm-train    # train the acoustic model
make vcm-eval      # evaluate the trained checkpoint (SPEC + TOY grammar sections)
make vcm-bench     # ONNX export + INT8 quantization + CPU latency/size benchmark
```

**Important — `train.py`'s shipped CLI defaults (`--max-epochs 150 --patience 10`) do NOT
reproduce the checkpoint the results below are measured against.** The real run that
produced it used higher values, to make use of available wall-clock budget after an
earlier noise-pool I/O bottleneck was fixed. The literal reproducing command was:

```bash
uv run python -m me2_voicegen.vcm.train \
  --device cuda:2 --max-minutes 30 --seed 0 \
  --max-epochs 300 --patience 60 --out-dir out/vcm
```

(`--device` should be any free GPU index on your machine, not necessarily `cuda:2` —
running `make vcm-train` with its stock defaults will train *a* checkpoint, just not one
whose numbers match the section below without also overriding `--max-epochs`/`--patience`
as shown.)

### Real results (honestly framed — do not read these as generalization evidence)

**Training** (`out/vcm/checkpoint.pt`, `out/vcm/loss_history.json`): val CTC loss fell from
**105.34** (epoch 0) to **1.04** (best, epoch 165) over 225 epochs (~18 min wall-clock, well
inside the 30-minute cap). Greedy-decoding all 292 held-out `target_commands` val clips
against the best checkpoint: **290/292 (99.3%)** produced a non-empty decode, **194/292
(66.4%)** decoded to an exact character-for-character match of the canonical phrase. This
reads as toy-scale memorization of the 20 closed-set canonical phrases the dataset actually
contains, not demonstrated generalization — every `target_commands` row (train/val/test
alike) is a synthetic TTS render of one of those same 20 phrases.

**Evaluation** (`out/vcm/eval_report.md`, `.json`), grammar-constrained beam search + a
per-grammar operating threshold chosen by a val-split sweep (never on `test`):
- `TOY_GRAMMAR`: **91.1%** test-split exact-match (133/146 accepted, all 133 also
  intent-correct).
- `SPEC_GRAMMAR`: **30.8%** test-split exact-match (45/146). This is a grammar-coverage
  artifact, not a model failure: only 2 of the 20 intents (`MESSAGE`, `SET_REMINDER`) have
  literally no BNF rule; the other 10 of the 12 non-passing intents have a real `$CMD_*`
  rule that requires a slot/phrasing the dataset's bare canonical phrase doesn't supply.
- **0% false-accept** on `babble`/`silence` at each grammar's chosen operating threshold —
  but this is threshold-specific, not an unconditional property: the val-split sweep shows
  TOY_GRAMMAR's false-accept rate climbing to **42.7%** at looser thresholds (past roughly
  -0.4) and plateauing there.

**Benchmark** (`out/vcm/vcm_benchmark.md`, `.json`) — ONNX export + static INT8
quantization of the trained (toy-scale, ~254k-param) checkpoint:
- Model file size: fp32 ONNX **1,020,103 bytes**, INT8 ONNX **277,219 bytes** — well under
  the original spec's 5MB budget, but at toy model scale, not the spec's full 1.2M-2.5M
  param target (that `spec-scale` preset was instantiated and sized, never trained).
- **Every single benchmark number below was measured on this node's AMD EPYC 7742 CPU —
  this node has no Raspberry Pi hardware, and none of these numbers are RPi measurements.**
  Latency pinned to 1 intra-op/1 inter-op onnxruntime thread, window = 1.5s:

  | variant | size (MB) | p50 (ms) | p95 (ms) | process peak RSS (MB) | inference-only RSS delta (KB) |
  |---|---|---|---|---|---|
  | fp32 ONNX | 0.973 | 1.065 | 1.076 | 423.895 | 0 |
  | INT8 ONNX (static, val-calibrated) | 0.264 | 1.039 | 1.049 | 508.570 | 0 |

  "Process peak RSS" is the whole Python/torch/onnxruntime process's high-water mark (model
  + runtime already loaded), not a per-inference figure — it is nowhere near a
  Raspberry-Pi-comparable number and should not be compared against the spec's 25MB budget
  directly. "Inference-only RSS delta" (0 KB for both variants) reflects that inference adds
  no *additional* high-water mark beyond process/model load, not that inference itself uses
  zero memory.

### Known gaps — not validated, stated plainly

- **Slot extraction was never tested on real audio.** The acoustic model saw zero
  slot-word (numbers/artists/personas) audio during training. Slot extraction correctness
  is verified only deterministically, via synthetic CTC posteriors built directly from
  text (`vcm/decoder.py`'s test suite), never real speech. The real-audio slot-eval task
  (2 personas x 12 slot-bearing phrases via TTS) was cleanly dropped — no persona
  reference audio exists on this checkout — but this is a ready-to-run gap, not an
  abandoned one: the resample/manifest/non-silence-check logic is built and unit-tested; it
  only needs real reference wavs to actually run (`python -m
  me2_voicegen.vcm.slot_eval_set --manifest personas.json`).
- **No Raspberry Pi hardware exists on this node.** Every latency/memory number above is a
  same-architecture-class estimate on a very different (256-thread server) CPU, not a real
  on-device measurement.
- **`$CMD_TEMPERATURE` has no up/down direction in the BNF** — a genuine gap in the
  original spec text (see Architecture above), not a decoder bug. Resolving it (if it
  matters for a scale-up pass) needs a spec change.
- **This is training-audio-memorization-scale data**: ~45 minutes total audio, 20 fixed
  command phrases. None of the numbers above are evidence the approach generalizes to more
  or real data — that is exactly what a scale-up pass would need to test.

### License

Any checkpoint trained on this data — and its ONNX/INT8 exports and any derivative —
inherits **CC-BY-NC-SA-4.0** (NonCommercial + ShareAlike), via `background_noise/`'s ESC-50
license, the same restriction that already governs `test_set/` itself. This note is carried
in the checkpoint/loss-history metadata and in every generated report's header.

### Dependency note

`onnx==1.16.1` (Apache-2.0) was added for ONNX export/quantization (`vcm.export_onnx`) — see
the "Trimmed dependencies" table above, which records this as a deliberate, documented
reversal of that table's earlier CosyVoice-spike exclusion, not an oversight.

## Option B spoken-command grammar

A second, independent context-free grammar — spec plus implementation — for the AI231
`MEX2/OptionB` spoken-command dataset (19 intents: 13 fixed-phrase x 3 phrasings, 6 slotted x 3
templates x 3 slot values). It lives in `src/me2_voicegen/vcm/optionb/` and reuses the same
combinator/trie grammar machinery as `vcm` (factored out into `grammar_core.py` for that
purpose) but is otherwise unrelated to `vcm`. The full technical contract (rules, slot
vocabularies, normalization, the canonical-93-vs-accepted-129 arithmetic, and the vocabulary
census) lives in [`docs/OPTIONB-GRAMMAR-CONTRACT.md`](docs/OPTIONB-GRAMMAR-CONTRACT.md).

**This is not the same taxonomy as `vcm`'s 20-intent grammar (`docs/VCM-CONTRACT.md`), and the
two are never interchangeable.** Notably: Option B has no `DIM_UP`/`DIM_DOWN` or
`TEMP_UP`/`TEMP_DOWN` split — `BRIGHTNESS` and `TEMPERATURE` are each a single intent
parameterized by a slot value, not directional pairs; Option B's `COLOR` intent has no VCM
counterpart at all; and Option B's reminder-creation intent is named `CREATE_REMINDER` where
VCM's equivalent is named `SET_REMINDER`. Do not attempt to map one taxonomy onto the other or
reuse `vcm` decoder/model code against `OPTIONB_GRAMMAR`, or vice versa — see
`OPTIONB-GRAMMAR-CONTRACT.md` section 5 for the full divergence list.

**Training + evaluation on Option B** uses the same `vcm` model/training code as the toy VCM
above, just pointed at the `optiond` preset (~1.01M params, 5 TCS blocks, 128 channels — the
config actually shipped downstream, not the toy `default` preset) and the Option B manifest:

```bash
make optionb-train
make optionb-eval
```

Real numbers (`optiond`, epoch 71, against the Option B test split): 1,717/1,766 (97.2%) target
commands accepted, 1,715/1,766 (97.1%) exact-intent-correct, false-accept rate 1/56 (1.8%) on
babble and 0/22 (0.0%) on silence, 1,022/1,023 slot values correct on intent-correct slot-bearing
clips. Full architecture, training recipe, and the incomplete-prefix rejection-gate calibration
history (margin 5.0 → 6.0 → 4.0, with the real-vs-synthetic-audio pitfall that drove the final
recalibration) are in [`docs/PROCESS-VCM-MODEL.md`](docs/PROCESS-VCM-MODEL.md).

## Wakeword DS-CNN

A depthwise-separable CNN ("computer"-only keyword spotter, `src/me2_voicegen/wakeword/model.py`,
≈50K params) trained on the dataset-build pipeline described above, sized to gate the VCM the way
a real always-on device would: cheap enough to run continuously, so the expensive grammar-
constrained beam search only wakes up once the wake word is heard.

```bash
make wakeword-train
make wakeword-bench     # ONNX export + INT8 quantization + CPU latency/size benchmark
```

Real numbers (val split, checkpoint epoch=20): F1 0.997 (`_wakeword_`), 0.997 (`_unknown_`),
0.998 (`_silence_`). Benchmark (AMD EPYC 7742 — **not** RPi hardware, which doesn't exist on this
node): fp32 ONNX 0.119MB / 0.256ms p50 latency; INT8 ONNX ~48.4KB / 0.148ms p50 latency, inside
the MLPerf Tiny reference range for this model class. Because the dataset's noise augmentation
mixes in ESC-50 (CC-BY-NC-SA-4.0), **this checkpoint — and anything trained on this dataset — is
CC-BY-NC-SA-4.0-encumbered** (non-commercial, share-alike), same restriction as the VCM-toy
checkpoint above.

The trained checkpoint is wired into the streaming runtime as a `WakeWordGate`, satisfying the
same `ListeningGate` protocol the manual `SpacebarGate` does — see "Streaming inference" below.
Full dataset-build stage-by-stage detail, architecture, and benchmark numbers are in
[`docs/PROCESS-WAKEWORD.md`](docs/PROCESS-WAKEWORD.md).

## VCMX — combined VCM + wakeword export & serving

`vcmx_merge.py` merges Option B (refreshed to live upstream grammar) with a second balanced
dataset ("VCM Dataset B") into speaker-disjoint **treatment** (25,231 rows) vs. **control**
(21,055 rows) manifests with byte-identical val/test splits, so the two training arms are a fair
before/after comparison. This is the manifest `optionb-train`/`optionb-eval` above actually run
against in production, and it's the export/serve path that ships both the VCM and wakeword
checkpoints together for live use:

```bash
make optionb-refresh   # pull the latest upstream Option B grammar/dataset
make vcmx-build         # build the treatment/control manifests
make vcmx-export        # fp32 + INT8 ONNX export of the trained VCM
make vcmx-serve          # live streaming, always-listening (gate=none)
make vcmx-serve-wakeword # live streaming, gated by the trained wakeword DS-CNN
```

`vcmx-serve-wakeword` is the actual "wake word gates the VCM, not both always running" deployment
shape: the DS-CNN runs continuously and cheaply, and the grammar-constrained beam search — the
expensive part — only executes once a listening period is open. Tech-lead review of the VCMX
merge: approved, no blocking findings. Full merge numbers, the fp32-vs-INT8 serving-default
rationale, and known gaps are in
[`docs/PROCESS-STREAMING-SERVING.md`](docs/PROCESS-STREAMING-SERVING.md).

## Streaming inference

A live, continuous spoken-command runner — `me2_voicegen.vcm.streaming` — that consumes a
trained VCM checkpoint and the `OPTIONB_GRAMMAR` decoder (above) against a rolling window of
audio instead of one pre-cut clip at a time, emitting detected `(intent, slots)` events as
JSONL as they happen. The full technical contract (interfaces, pipeline order, JSONL schema,
config precedence, trust boundaries) lives in
[`docs/STREAMING-CONTRACT.md`](docs/STREAMING-CONTRACT.md); this section covers what it does,
how to run it, and the risks/limitations stated plainly.

### Running it

```bash
make stream                                    # live mic, optionc preset, ONNX fp32
make stream STREAM_SOURCE=path/to/clip.wav     # deterministic file replay instead
make stream STREAM_ARGS="--policy mode_period --gate spacebar"      # gated run: SPACE opens a 5.0 s listening period
make stream STREAM_ARGS="--policy mode_period --gate spacebar --gate-period 10"
make stream-single-period                         # one exact 3.0 s inference per SPACE press
make stream-wakeword                              # gated by the trained wakeword DS-CNN instead of SPACE
```

or directly:

```bash
uv run python -m me2_voicegen.vcm.streaming
uv run python -m me2_voicegen.vcm.streaming --source path/to/clip.wav
uv run python -m me2_voicegen.vcm.streaming --policy mode_period --gate spacebar --gate-period 10
```

With no arguments, it opens the live microphone, loads the `optionc` checkpoint preset (the
`MODEL_REGISTRY` entry that beats the `default` preset on every eval metric — see the
"VCM toy" section above), and runs inference through the ONNX fp32 backend. It runs
continuously, printing one JSONL object per detected, non-suppressed command on stdout and a
startup banner (checkpoint/run dir, preset, backend, grammar, window/stride/refractory/beam,
resolved threshold, listening gate, license) on stderr, until source EOF (file replay), `Ctrl-C`,
or `--listen-for <seconds>` expires — all three exit 0 with a summary line
(`windows`/`events`/`suppressed`/`dropped`).

### Gated runs: one consolidated event per listening period

The default run (`--policy threshold --gate none`) is the original per-window behavior,
unchanged. A **gated run** pairs the new `mode_period` policy with the new spacebar
listening gate:

```bash
make stream STREAM_ARGS="--policy mode_period --gate spacebar"
make stream STREAM_ARGS="--policy mode_period --gate spacebar --gate-period 10"
```

In a gated run, press **SPACE** on the terminal to open a bounded listening period
(default 5.0 s; `--gate-period <seconds>` to change it). Every window evaluated while the
period is open is collected, and at the period's end the runner emits **at most one**
consolidated event for the whole period: the **mode** (most frequent `(intent, slots)`
decode) across the period, accepted only if that decode class's *mean* confidence clears the
resolved threshold (ties break by count, then confidence sum, then first-seen order —
deterministic). Pressing SPACE again mid-period discards everything collected so far and
starts a fresh period from that press. While no period is open, **nothing is accepted,
no matter how confident a decode is** — a gate-closed window is rejected outright. The
startup banner tells you which mode you're in: `listening gate: none`, or
`listening gate: spacebar (press SPACE to open a 5.0 s listening period)`.

Add `--log-periods` to a gated run for a per-period digest on stderr: one line when a
period opens (or is restarted by a re-press), and one line at the period's end carrying
the consolidated result — including periods that produced *no* trigger, which are
otherwise invisible:

```
gate: open      t=12.30s
gate: closed    t=17.30s  period: REJECT  silence dominated the period (None 9/10)
```

The JSONL on stdout is unchanged (an accepted period still emits its `"event": "trigger"`
line); `--log-all-windows` remains the full per-window verbosity.

### Single full-period inference

`single_period` is the low-compute alternative to `mode_period`: it collects
exactly 3.0 seconds after SPACE, makes one model inference and grammar decode
over that exact interval, then uses the normal threshold and debounce.

```bash
make stream-single-period
# equivalent:
make stream STREAM_ARGS="--policy single_period --gate spacebar --gate-period 3 --threshold -0.132222 --log-periods"
```

The explicit `-0.132222` is the provisional zero-false-accept winner from the
local scorer spike (89 negative windows), not a production FAR/hour
calibration. `single_period` requires `--gate spacebar --gate-period 3`;
choose `mode_period` when you want the existing multi-window vote.

Gated runs are intended for the live microphone: file replay is lockstep (it runs faster
than real time), so keypress timing against the audio stream is unreliable.

Two flag combinations are hard errors (exit 1 with an actionable message, before any model
load or microphone open, never a traceback): `--policy mode_period` or `--policy single_period` without a gate, and
`--gate spacebar` with `--policy threshold` — the gate and the policy are only useful
together, and a silently ignored flag would be a trap. `--gate spacebar` also fails fast
(exit 1, actionable message: run in an interactive shell, pass `--gate none`, or replay a
file with `--source <wav>`) when stdin isn't a real terminal, e.g. piped or CI runs. Ctrl-C
still works mid-run (raw mode keeps signal handling), and the terminal is restored on every
exit path — clean, Ctrl-C, or error.

**How this mitigates the two failure modes observed on real hardware (above):**

- **Lingering duplicate triggers** — a phrase that decodes correctly across roughly ten
  consecutive windows as it slides through the 2.5 s window used to fire twice, once the
  1.5 s refractory expired. In a gated run one period produces at most one accept, so the
  near-identical decodes collapse into the single mode decision; the `Debouncer` still
  independently rate-limits consecutive *periods* as before.
- **Ambient false positives** — the spurious `TIME` triggers observed in the "Known risk"
  section below came from confident-but-wrong decodes with nobody speaking. In a gated run,
  gate-closed windows are rejected before any threshold arithmetic, so ambient noise cannot
  trigger at all; inside a period, noise must additionally be the *most frequent* decode
  class across the whole period *and* clear the threshold on its class's mean confidence,
  which intermittent noise cannot do.

The full contract (gate state machine, the policy's flush/tie-break rules, decision reason
formats) is in [`docs/STREAMING-CONTRACT.md`](docs/STREAMING-CONTRACT.md) sections 4 and 6.

### Mic-primary design and prerequisites

The live microphone is the primary, default input path; `--source <wav>` (deterministic file
replay, no drops, byte-for-byte identical events across runs) is the secondary path used for
testing and reproducible demos. Live capture works by spawning `arecord` (part of
`alsa-utils`, present on essentially every Linux distro) as a subprocess and reading raw PCM
from its stdout — it needs `arecord` on `PATH` and an actual capture device. If the mic can't
be opened, the run exits with a clear, actionable message (never a traceback) naming the two
fallbacks: `--source <wav>` to replay a file instead, or `--mic-command` to point at a working
capture command of your own. `--mic-command` also lets an operator swap in `parec`/`ffmpeg`/a
different ALSA device string without touching code.

### Backend: ONNX by default, torch as an explicit fallback

`--backend onnx|torch` (default `onnx`) selects the inference backend; `--onnx-variant
fp32|int8` (default `fp32`) selects the ONNX export variant. **fp32 is the default, not int8,
because the shipped `-0.1` operating threshold was tuned on torch-fp32 logits** (the same
threshold `vcm.evaluate`'s val-split sweep chose) — the int8 export would need its own
re-tuning pass against that quantized model's own confidence distribution before it could
safely reuse the same threshold. The two backends are verified numerically close (ONNX vs.
torch logits within 1e-2 max-abs-diff on the checked-in `optionc` artifacts) and produce the
same intent/slots on the same wav.

### Known risk: confidence is sensitive to window padding

Stated plainly, in the same spirit as the "Known gaps" section above: **a correctly-decoded
phrase can still be rejected purely because of how much silence padding surrounds it in the
fixed-length window**, even from a moderately (not maximally) confident model. Confidence is
`beam_total / T` — a mean per-frame log-probability over the *entire* fixed-length window — so
non-speech padding frames dilute it. Measured directly (holding the utterance constant, varying
only the model's blank-confidence margin during non-speech padding of a 2.5s window):

| blank-confidence margin | resulting confidence | accepted at -0.1? |
|---|---|---|
| 12.0 | -0.0002 | yes |
| 6.0 | -0.0436 | yes |
| 3.0 | -0.5670 | **no** |
| 1.5 | -1.2863 | **no** |

This is a real, live phenomenon, not just a synthetic measurement: a manual smoke test run
directly on this node's real microphone (ambient room noise, no one speaking) produced two
spurious `TIME` intent triggers at confidence -0.085 and -0.09 — both just inside the -0.1
threshold. Because every window is decoded once at the most permissive threshold and the
accept/reject decision is applied afterward (see the contract's fixed pipeline order), two
mitigations are built in rather than bolted on: `--threshold <value>` overrides the resolved
threshold outright, and `--log-all-windows` emits every evaluated window's real confidence (not
just accepted triggers) as a JSONL `"window"` record, so an operator can empirically re-tune
`--threshold` for a specific room/mic rather than trust the eval-set-chosen default blind.

### Security notes

- **`--model`/`--config` and `--mic-command` are operator-supplied trust boundaries, not
  sandboxed inputs.** The resolved model path is fed to either `torch.load` (pickle
  deserialization) or ONNX Runtime's model parser — neither is safe against an untrusted file.
  `--mic-command` is spawned as an argv list (`shell=False`, no shell injection surface) but
  still executes whatever command you name. Only point either at artifacts/commands you
  produced or otherwise trust.
- **`weights_only=True` is a mitigation, not a guarantee.** Torch checkpoint loading defaults
  to `weights_only=True`, but this project's pinned `torch==2.3.1` is within the affected range
  of **CVE-2025-32434**, a known bypass of `weights_only=True` (fixed in torch 2.6.0).
  Upgrading torch is a deliberately separate, larger, backlogged concern — a repo-wide pinned
  CUDA stack — not something this feature attempted.
- **A `--config <file>.json` is exactly as trusted as CLI flags.** It can set `model` or
  `mic_command` just as freely as the equivalent command-line flag can (field names and
  per-field types/choices are validated, but the *values* are not judged for safety). Never
  treat a `--config` file as inert data safe to copy from an untrusted source.

### Known limitations, stated plainly

- **No real Raspberry Pi hardware exists on this node** (same caveat as the VCM-toy section
  above), so no on-device latency numbers exist for this feature either. The measured latency
  numbers that validated the shipped real-time budget (`window_s=2.5`, `stride_s=0.25`,
  `beam_width=25`) are this-node estimates on an AMD EPYC CPU, not RPi numbers: beam search over
  a real 2.5s window decoded in ~83.6ms — **beam search is ~97% of per-window compute; the ONNX
  forward pass itself is only ~2.8ms** and is not what makes this real-time.
- **The live-mic path was verified end-to-end on this specific node**: a manual smoke test
  correctly opened the microphone, ran real ONNX inference against live audio, and exited
  cleanly on `--listen-for` expiry with a correct summary line. Mic/device availability will
  vary on other machines — hence `MicrophoneUnavailableError`'s actionable fallback message
  naming `--source <wav>` and `--mic-command` rather than a bare traceback.

### License

Same as the VCM-toy checkpoint above: any checkpoint used here inherits **CC-BY-NC-SA-4.0**
(NonCommercial + ShareAlike) via `background_noise/`'s ESC-50 license. The streaming CLI's
startup banner echoes this license string on every run.
