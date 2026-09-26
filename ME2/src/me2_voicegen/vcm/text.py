"""Text normalization and transcript resolution across the five
out/conversions/v2 test_set source datasets. Dispatches to each
per-experiment package (`vcm.optiona`, `vcm.optionb`) for that experiment's
own canonical phrase table/transcript prep.

See docs/VCM-CONTRACT.md for the full transcript-resolution table this
module implements; this file is its source of truth, not the other way
around.
"""

from __future__ import annotations

import csv
import string
from functools import lru_cache
from pathlib import Path
from typing import Mapping

from me2_voicegen.vcm.optiona.phrases import INTENT_PHRASES as _OPTIONA_INTENT_PHRASES
from me2_voicegen.vcm.optionb.transcript import prepare_ctc_transcript

# src/me2_voicegen/vcm/text.py -> parents[3] is the ME2 project root.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONVERSIONS_V2_DIR = PROJECT_ROOT / "out" / "conversions" / "v2"

_ALLOWED_CHARS = set(string.ascii_lowercase) | {" ", "'"}
_CURLY_APOSTROPHES = {"‘", "’", "ʼ"}


def normalize_text(text: str) -> str:
    """Lowercase; fold curly apostrophes to `'`; drop/space out any other
    punctuation or symbol not in the 29-token alphabet; collapse runs of
    whitespace to a single space; strip leading/trailing whitespace.

    Guarantees the result is encodable by `vcm.alphabet.encode` unchanged.
    """
    text = text.lower()
    chars = []
    for ch in text:
        if ch in _CURLY_APOSTROPHES:
            chars.append("'")
        elif ch in _ALLOWED_CHARS:
            chars.append(ch)
        else:
            # Punctuation/digits/symbols/unicode letters outside a-z are
            # dropped, replaced by a word boundary so adjacent words don't
            # get glued together (e.g. "lights,on" -> "lights on").
            chars.append(" ")
    return " ".join("".join(chars).split())


@lru_cache(maxsize=None)
def _load_source_manifest(source_dataset: str) -> Mapping[str, dict]:
    """Load out/conversions/v2/<source_dataset>/manifest.csv keyed by its
    own `filename` column. Cached per source_dataset for the process
    lifetime (these files are read-only inputs, not mutated at runtime).
    """
    path = CONVERSIONS_V2_DIR / source_dataset / "manifest.csv"
    with path.open(newline="", encoding="utf-8") as f:
        return {row["filename"]: row for row in csv.DictReader(f)}


def resolve_transcript(manifest_row: Mapping[str, str]) -> str | None:
    """Resolve the ground-truth transcript for one out/conversions/v2/
    test_set/manifest.csv row, joining back to that row's own source
    dataset's manifest. See docs/VCM-CONTRACT.md for the full rules table;
    summary:

    - sanitized_clean: no source manifest exists ->
      vcm.optiona.phrases.INTENT_PHRASES[label].
    - common_voice_negative: source manifest `transcript` column (per-chunk
      Whisper transcript; may be "").
    - youtube_institutional: source manifest `transcript` column (blank for
      non-speech `ambient`-bucket rows -- that's correct, not missing data).
    - background_noise: no transcript column, no speech present -> "".
    - filipino_speech_corpus: per decision (B), always None, regardless of
      whether the row is a whole-clip or `_cNN.wav` chunked row -- excluded
      from CTC loss uniformly, kept only as an eval rejection probe.
    - optionb / vcm_balanced / fil50_persona: own manifest's `transcript`
      column, read directly (no source manifest join -- unlike
      common_voice_negative/youtube_institutional, this row *is* the
      source row), passed through
      `vcm.optionb.transcript.prepare_ctc_transcript` first so digit-bearing
      slot values (e.g. "Alarm 6 AM") survive this module's own
      `normalize_text` instead of being silently dropped. `vcm_balanced`
      (VCM Dataset B, merged in by `vcm.vcmx_merge`) and `fil50_persona`
      (feature `accent-balance-fil50`, persona-synthesized Filipino-voiced
      commands whose text is copied verbatim from a same-split non-Filipino
      row -- `.scratch/accent-balance-fil50/tickets/00-RECAP.md` T5) both
      reuse this branch unchanged: their own rows are pre-normalized
      (alias-normalized, digit-kept text) the same way `optionb` rows
      already are, so no separate digit-spelling rule is needed for either.
      A noisy `fil50_persona` row is still `source_dataset="fil50_persona"`
      (noise is signaled via `_noisy` in `filename`, the same convention
      `optionb`'s own noisy rows use -- not a distinct source_dataset
      value the way wakeword's `_noisy` subsets are).
    """
    source_dataset = manifest_row["source_dataset"]

    if source_dataset == "sanitized_clean":
        return _OPTIONA_INTENT_PHRASES[manifest_row["label"]]

    if source_dataset == "background_noise":
        return ""

    if source_dataset == "filipino_speech_corpus":
        return None

    if source_dataset in ("optionb", "vcm_balanced", "fil50_persona"):
        return prepare_ctc_transcript(manifest_row["transcript"])

    if source_dataset in ("common_voice_negative", "youtube_institutional"):
        source_manifest = _load_source_manifest(source_dataset)
        basename = Path(manifest_row["source_relpath"]).name
        source_row = source_manifest[basename]
        return source_row["transcript"]

    raise ValueError(f"unknown source_dataset: {source_dataset!r}")
