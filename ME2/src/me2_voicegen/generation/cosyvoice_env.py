"""Path resolution and vendored CosyVoice sys.path shim.

All paths are anchored off this package's own location so that Makefile
recipes and callers work correctly regardless of the invoking cwd.
"""

from __future__ import annotations

import sys
from pathlib import Path

# src/me2_voicegen/generation/cosyvoice_env.py -> parents[3] is the ME2 project root.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

VENDOR_ROOT = PROJECT_ROOT / "vendor"
COSYVOICE_DIR = VENDOR_ROOT / "CosyVoice"
MATCHA_TTS_DIR = COSYVOICE_DIR / "third_party" / "Matcha-TTS"
MODELS_DIR = PROJECT_ROOT / "models"
OUT_DIR = PROJECT_ROOT / "out"


def add_cosyvoice_to_syspath() -> None:
    """Prepend the vendored CosyVoice + Matcha-TTS directories to sys.path.

    Raises RuntimeError naming `make vendor` if either directory is missing,
    since a non-recursive clone silently omits Matcha-TTS and the resulting
    ModuleNotFoundError would otherwise surface much later, at model
    construction time, with no clue as to the actual cause.
    """
    missing = [p for p in (COSYVOICE_DIR, MATCHA_TTS_DIR) if not p.is_dir()]
    if missing:
        missing_str = ", ".join(str(p) for p in missing)
        raise RuntimeError(
            f"Vendored CosyVoice not found (missing: {missing_str}). "
            "Run `make vendor` from ME2/ to clone it."
        )

    for path in (COSYVOICE_DIR, MATCHA_TTS_DIR):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
