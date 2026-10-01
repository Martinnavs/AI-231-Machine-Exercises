#!/usr/bin/env bash
# One-shot environment init. Idempotent -- safe to re-run.
#
#   Kaggle : use the image's torch if it can run the augmenter (else install the
#            pinned one), add the few missing runtime deps, find/extract the data
#            dataset and link it into the repo layout, then smoke-check everything.
#   Local  : `uv sync` (the repo's normal env), then the same data/GPU checks.
#
# Env: DATA_DIR  dir that already contains out/conversions/v2 + raw_datasets
#                (default: auto-search /kaggle/input, else extract me2-data.tar)
#      DATA_CACHE  where a me2-data.tar is extracted (default /kaggle/temp or /tmp)
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=${PY:-python}
ON_KAGGLE=0; [ -d /kaggle/input ] && ON_KAGGLE=1
TRAIN_MANIFEST=${TRAIN_MANIFEST:-out/conversions/v2/optionb-v3-vcmx-fil50-ambient/manifest.csv}
EVAL_MANIFEST=${EVAL_MANIFEST:-out/conversions/v2/optionb-v3-vcmx-fil50/manifest.csv}
say() { printf '\n==> %s\n' "$*"; }

# ---------------------------------------------------------------- python deps
if [ "$ON_KAGGLE" = 1 ]; then
  say "python deps (Kaggle)"
  # The augmenter needs torchaudio.prototype.functional.simulate_rir_ism, which newer
  # torchaudio releases dropped; fall back to the repo's pinned torch 2.3.1 if so.
  if ! $PY -c "from torchaudio.prototype.functional import simulate_rir_ism" 2>/dev/null; then
    echo "image torch/torchaudio can't import simulate_rir_ism -> installing pinned torch 2.3.1 (cu121)"
    pip install -q "numpy==1.26.4" "torch==2.3.1" "torchaudio==2.3.1" --index-url https://download.pytorch.org/whl/cu121 \
      --extra-index-url https://pypi.org/simple
  fi
  need=()
  $PY -c "import onnx" 2>/dev/null || need+=("onnx==1.16.1")
  $PY -c "import onnxruntime" 2>/dev/null || need+=("onnxruntime==1.18.0")
  $PY -c "import soundfile" 2>/dev/null || need+=("soundfile")
  $PY -c "import librosa" 2>/dev/null || need+=("librosa")
  if [ ${#need[@]} -gt 0 ]; then pip install -q "${need[@]}"; fi
  # No `pip install -e .`: the full dependency list is CosyVoice/TTS-heavy and unused here.
  # kaggle.mk exports PYTHONPATH=src instead.
else
  say "python deps (local: uv sync)"
  ${UV:-uv} sync
fi

# ------------------------------------------------------------------- the data
say "data"
if [ ! -f "$TRAIN_MANIFEST" ]; then
  root=""
  if [ -n "${DATA_DIR:-}" ]; then
    root=$DATA_DIR
  elif [ "$ON_KAGGLE" = 1 ]; then
    # (a) already-extracted dataset layout
    hit=$(find /kaggle/input -maxdepth 6 -path "*/$TRAIN_MANIFEST" 2>/dev/null | head -1 || true)
    if [ -n "$hit" ]; then
      root=${hit%/$TRAIN_MANIFEST}
    else
      # (b) the tar produced by `make kaggle-pack-data`
      tarball=$(find /kaggle/input -maxdepth 4 -name me2-data.tar 2>/dev/null | head -1 || true)
      [ -n "$tarball" ] || { echo "no data found under /kaggle/input: attach the me2-data dataset (see docs/KAGGLE.md)" >&2; exit 1; }
      cache=${DATA_CACHE:-$([ -d /kaggle/temp ] && echo /kaggle/temp || echo /tmp)}/me2-data
      if [ ! -f "$cache/.extracted" ]; then
        echo "extracting $tarball -> $cache"
        mkdir -p "$cache" && tar -xf "$tarball" -C "$cache" && touch "$cache/.extracted"
      fi
      root=$cache
    fi
  else
    echo "$TRAIN_MANIFEST not found. Locally the data lives in out/ already; set DATA_DIR=<dir> to link another copy." >&2
    exit 1
  fi
  [ -f "$root/$TRAIN_MANIFEST" ] || { echo "DATA_DIR=$root has no $TRAIN_MANIFEST" >&2; exit 1; }
  # Link only what's read-only input. out/ itself stays a real, writable dir (run outputs go in
  # out/vcm); raw_datasets is linked whole. Manifest paths are relative, so the layout just works.
  mkdir -p out
  [ -e out/conversions ] || ln -s "$root/out/conversions" out/conversions
  [ -e raw_datasets ] || ln -s "$root/raw_datasets" raw_datasets
  echo "linked data from $root"
else
  echo "data already present at $TRAIN_MANIFEST"
fi

# ------------------------------------------------------------------ the check
say "smoke check"
PYTHONPATH=src $PY - "$TRAIN_MANIFEST" "$EVAL_MANIFEST" <<'PYEOF'
import csv, os, subprocess, sys
import torch, torchaudio
from torchaudio.prototype.functional import simulate_rir_ism  # noqa: F401
import me2_voicegen.vcm.train, me2_voicegen.vcm.evaluate, me2_voicegen.vcm.benchmark  # noqa: F401
print(f"torch {torch.__version__}  torchaudio {torchaudio.__version__}  cuda={torch.cuda.is_available()}",
      f"({torch.cuda.get_device_name(0)} x{torch.cuda.device_count()})" if torch.cuda.is_available() else "-> CPU ONLY, training will be very slow")
for m in sys.argv[1:]:
    base = os.path.dirname(m)
    rows = list(csv.DictReader(open(m, newline="", encoding="utf-8")))
    # check a spread of rows, not just the first: a broken symlink usually breaks whole subtrees
    sample = rows[:: max(1, len(rows) // 200)]
    bad = [r["path"] for r in sample if not os.path.exists(os.path.join(base, r["path"]))]
    print(f"{m}: {len(rows)} rows, {len(sample)} spot-checked, {len(bad)} missing")
    if bad:
        sys.exit(f"missing audio, e.g. {bad[:3]}")
print("nproc", os.cpu_count())
PYEOF
say "environment ready -> run: make experiment"
