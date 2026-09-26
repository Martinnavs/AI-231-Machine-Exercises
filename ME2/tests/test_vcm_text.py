"""Unit tests for vcm.alphabet and vcm.text.

Uses the real out/conversions/v2/test_set/manifest.csv and the real source
manifests it joins against (per ticket 01's "Must verify" acceptance
criteria) - these are repo-local, checked-in fixtures, not a multi-GB
external corpus, so reading them directly (rather than a synthetic subset)
is intentional here.
"""

from __future__ import annotations

import csv
import glob
import re
import subprocess
import sys
from pathlib import Path

import pytest

import me2_voicegen.vcm.alphabet as alphabet
import me2_voicegen.vcm.text as text
from me2_voicegen.vcm.optiona.phrases import INTENT_PHRASES

MANIFEST_PATH = text.PROJECT_ROOT / "out" / "conversions" / "v2" / "test_set" / "manifest.csv"
REPORTS_DIR = text.PROJECT_ROOT / "out" / "conversions" / "v2" / "reports"


def _manifest_rows() -> list[dict]:
    with MANIFEST_PATH.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# no-torch guarantee
# ---------------------------------------------------------------------------


def test_alphabet_and_text_import_with_no_torch():
    # Run in a fresh interpreter: other tests in the full suite may have
    # already imported torch by the time this test runs, so checking
    # sys.modules in-process would be contaminated by import order.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import me2_voicegen.vcm.alphabet; import me2_voicegen.vcm.text; "
            "print('torch' in sys.modules)",
        ],
        cwd=text.PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False", result.stdout + result.stderr


# ---------------------------------------------------------------------------
# alphabet
# ---------------------------------------------------------------------------


def test_alphabet_has_29_tokens():
    assert alphabet.ALPHABET_SIZE == 29
    assert alphabet.BLANK_ID == 0


@pytest.mark.parametrize("phrase", sorted(INTENT_PHRASES.values()))
def test_encode_decode_roundtrip_canonical_phrases(phrase):
    assert alphabet.decode(alphabet.encode(phrase)) == phrase


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Set Alarm!", "set alarm"),
        ("  play   music  ", "play music"),
        ("Don't Stop.", "don't stop"),
        ("WEATHER?!", "weather"),
        ("Call, now.", "call now"),
        ("It’s a Test", "it's a test"),
        ("100% brighter", "brighter"),
        ("lights,on", "lights on"),
    ],
)
def test_normalize_then_roundtrip(raw, expected):
    normalized = text.normalize_text(raw)
    assert normalized == expected
    assert alphabet.decode(alphabet.encode(normalized)) == expected


def test_encode_rejects_out_of_alphabet_char():
    with pytest.raises(ValueError):
        alphabet.encode("hello!")


def test_decode_rejects_invalid_id():
    with pytest.raises(ValueError):
        alphabet.decode([999])


def test_collapse_merges_repeats_and_drops_blanks():
    # blank=0, a=1, space=27
    raw = [0, 1, 1, 0, 0, 27, 27, 1, 1, 1, 0]
    assert alphabet.collapse(raw) == [1, 27, 1]


def test_collapse_empty():
    assert alphabet.collapse([]) == []


# ---------------------------------------------------------------------------
# normalize_text guarantees
# ---------------------------------------------------------------------------


def test_normalize_text_output_is_always_encodable():
    samples = [
        "SET ALARM",
        "call   me? maybe.",
        "  ",
        "",
        "100% Volume-Up!!",
        "\tTabs\nand\nnewlines\t",
    ]
    for s in samples:
        normalized = text.normalize_text(s)
        alphabet.encode(normalized)  # must not raise
        assert normalized == normalized.strip()
        assert "  " not in normalized


# ---------------------------------------------------------------------------
# resolve_transcript against real manifests
# ---------------------------------------------------------------------------


def _first_row_for_source(source_dataset: str) -> dict:
    for row in _manifest_rows():
        if row["source_dataset"] == source_dataset:
            return row
    raise AssertionError(f"no manifest row found for source_dataset={source_dataset!r}")


def test_resolve_transcript_sanitized_clean():
    row = _first_row_for_source("sanitized_clean")
    result = text.resolve_transcript(row)
    assert result == INTENT_PHRASES[row["label"]]


def test_resolve_transcript_common_voice_negative():
    row = _first_row_for_source("common_voice_negative")
    result = text.resolve_transcript(row)
    source_manifest = text._load_source_manifest("common_voice_negative")
    basename = Path(row["source_relpath"]).name
    assert result == source_manifest[basename]["transcript"]


def test_resolve_transcript_youtube_institutional():
    row = _first_row_for_source("youtube_institutional")
    result = text.resolve_transcript(row)
    source_manifest = text._load_source_manifest("youtube_institutional")
    basename = Path(row["source_relpath"]).name
    assert result == source_manifest[basename]["transcript"]


def test_resolve_transcript_background_noise_is_empty_string():
    row = _first_row_for_source("background_noise")
    assert text.resolve_transcript(row) == ""


def test_resolve_transcript_filipino_speech_corpus_is_always_none():
    rows = [r for r in _manifest_rows() if r["source_dataset"] == "filipino_speech_corpus"]
    assert len(rows) == 187
    for row in rows:
        assert text.resolve_transcript(row) is None


def test_resolve_transcript_common_voice_negative_empty_count():
    rows = [r for r in _manifest_rows() if r["source_dataset"] == "common_voice_negative"]
    resolved = [text.resolve_transcript(r) for r in rows]
    assert len(rows) == 187
    assert sum(1 for r in resolved if r == "") == 10
    assert all(r is not None for r in resolved)


def test_resolve_transcript_youtube_institutional_empty_count():
    rows = [r for r in _manifest_rows() if r["source_dataset"] == "youtube_institutional"]
    resolved = [text.resolve_transcript(r) for r in rows]
    assert len(rows) == 218
    assert sum(1 for r in resolved if r == "") == 34
    assert all(r is not None for r in resolved)


def test_resolve_transcript_unknown_source_dataset_raises():
    with pytest.raises(ValueError):
        text.resolve_transcript({"source_dataset": "not_a_real_source"})


def test_resolve_transcript_fil50_persona_matches_optionb_branch():
    """feature accent-balance-fil50 (.scratch/accent-balance-fil50/tickets/
    00-RECAP.md T5): a persona-synthesized row's transcript is its own
    `transcript` column, prepared the same way optionb/vcm_balanced rows
    are -- including a noisy row, which stays `source_dataset=fil50_persona`
    (noise is signaled via `_noisy` in filename, not a distinct source_dataset)."""
    for filename in ("ALARM_6_00AM_fsc_94_v1.wav", "ALARM_6_00AM_fsc_94_v1_noisy.wav"):
        row = {"source_dataset": "fil50_persona", "filename": filename, "transcript": "Alarm 6 AM"}
        assert text.resolve_transcript(row) == text.prepare_ctc_transcript("Alarm 6 AM")


# ---------------------------------------------------------------------------
# slow drift-guard: re-derive INTENT_PHRASES from the real QA reports
# ---------------------------------------------------------------------------

_QA_ROW_RE = re.compile(r"^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([\d.]+)\s*\|\s*(yes|no)\s*\|\s*$")


def _derive_intent_phrases_from_reports() -> dict[str, str]:
    derived: dict[str, str] = {}
    for intent in INTENT_PHRASES:
        glob_intent = intent.replace("_", "-")
        matches = sorted(glob.glob(str(REPORTS_DIR / f"tmp-qa-{glob_intent}-*.md")))
        assert len(matches) == 1, f"expected exactly one QA report for {intent}, found {matches}"
        report_path = Path(matches[0])

        expected_phrases: set[str] = set()
        in_flagged_table = False
        for line in report_path.read_text(encoding="utf-8").splitlines():
            if line.strip() == "## Flagged":
                in_flagged_table = True
                continue
            if not in_flagged_table:
                continue
            m = _QA_ROW_RE.match(line)
            if not m:
                continue
            _file, expected, _transcribed, _score, _exact = m.groups()
            if expected == "expected":
                continue
            expected_phrases.add(expected)

        assert len(expected_phrases) == 1, (
            f"{report_path.name}: expected exactly one distinct 'expected' phrase, "
            f"found {expected_phrases}"
        )
        derived[intent] = expected_phrases.pop()
    return derived


@pytest.mark.slow
def test_intent_phrases_matches_real_qa_reports():
    derived = _derive_intent_phrases_from_reports()
    assert derived == INTENT_PHRASES
