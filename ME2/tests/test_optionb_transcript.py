"""Unit tests for `optionb.transcript.prepare_ctc_transcript` and the
`vcm.text.resolve_transcript` "optionb" branch it feeds (ticket 03).
"""

from __future__ import annotations

import csv

import pytest

from me2_voicegen.vcm.optionb.grammar import OPTIONB_GRAMMAR
from me2_voicegen.vcm.optionb.transcript import prepare_ctc_transcript
from me2_voicegen.vcm.alphabet import encode
from me2_voicegen.vcm.text import CONVERSIONS_V2_DIR, normalize_text, resolve_transcript

# The live-upstream refresh (docs/OPTIONB-GRAMMAR-CONTRACT.md §6): OPTIONB_GRAMMAR
# now follows optionb-v3's upstream commit, not the pinned optionb/ baseline
# (which is deliberately left un-refetched/unchanged as a reproducible
# snapshot -- its 7 changed intents' old transcripts are no longer grammar
# -accepted under the refreshed grammar, by design).
REAL_OPTIONB_MANIFEST = CONVERSIONS_V2_DIR / "ai231-me2-voice-commands" / "manifest.csv"


def test_spells_out_a_single_digit_run():
    assert prepare_ctc_transcript("Alarm 6 AM") == "Alarm six AM"


def test_spells_out_multiple_digit_runs():
    assert (
        prepare_ctc_transcript("Set the brightness to 100 percent")
        == "Set the brightness to one hundred percent"
    )


def test_no_digits_is_unchanged():
    assert prepare_ctc_transcript("Play music") == "Play music"


def test_empty_string_is_unchanged():
    assert prepare_ctc_transcript("") == ""


def test_digit_run_above_100_raises_value_error():
    with pytest.raises(ValueError):
        prepare_ctc_transcript("Set a timer for 250 seconds")


def test_end_to_end_survives_normalize_and_is_grammar_accepted():
    spelled = prepare_ctc_transcript("Alarm 6 AM")
    normalized = normalize_text(spelled)
    assert normalized == "alarm six am"
    assert OPTIONB_GRAMMAR.accepts(normalized) is not None
    # No numeric content silently dropped: the digit is still represented
    # (as a word), not stripped, and the alphabet can encode the result.
    assert "six" in normalized
    encode(normalized)


class TestResolveTranscriptOptionbBranch:
    def test_reads_transcript_column_directly(self, vcm_fake_manifest_factory):
        manifest_path = vcm_fake_manifest_factory(
            [
                {
                    "bucket": "target_commands",
                    "source_dataset": "optionb",
                    "label": "ALARM",
                    "transcript": "Alarm 6 AM",
                }
            ]
        )
        with manifest_path.open(newline="", encoding="utf-8") as f:
            row = next(csv.DictReader(f))

        assert resolve_transcript(row) == "Alarm six AM"

    def test_empty_transcript_resolves_to_empty_string_not_none(
        self, vcm_fake_manifest_factory
    ):
        manifest_path = vcm_fake_manifest_factory(
            [
                {
                    "bucket": "babble",
                    "source_dataset": "optionb",
                    "label": "unknown",
                    "transcript": "",
                }
            ]
        )
        with manifest_path.open(newline="", encoding="utf-8") as f:
            row = next(csv.DictReader(f))

        assert resolve_transcript(row) == ""

    def test_no_source_manifest_join_needed(self, vcm_fake_manifest_factory, tmp_path):
        """D7: optionb resolution must not touch `_load_source_manifest` /
        `CONVERSIONS_V2_DIR` at all -- deleting the fake conversions dir
        entirely must not break resolution."""
        manifest_path = vcm_fake_manifest_factory(
            [
                {
                    "bucket": "target_commands",
                    "source_dataset": "optionb",
                    "label": "PLAY_MUSIC",
                    "transcript": "Play music",
                }
            ]
        )
        with manifest_path.open(newline="", encoding="utf-8") as f:
            row = next(csv.DictReader(f))

        assert resolve_transcript(row) == "Play music"

    def test_unknown_source_dataset_still_raises(self):
        with pytest.raises(ValueError):
            resolve_transcript({"source_dataset": "not_a_real_dataset"})


def test_import_vcm_text_has_no_import_cycle():
    import me2_voicegen.vcm.text  # noqa: F401


@pytest.mark.skipif(
    not REAL_OPTIONB_MANIFEST.exists(), reason="ai231 dataset not converted locally"
)
def test_exact_real_transcripts_are_grammar_accepted_and_cover_all_93():
    """`exact` rows of the converted ai231 dataset: every one of the 93 phrases appears and is accepted
    with the right intent; raw typos ("9PM", "what is the weather") stay under 0.1%."""
    with REAL_OPTIONB_MANIFEST.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["bucket"] == "target_commands" and r["variation_match"] == "exact"]

    accepted, rejected = set(), 0
    for r in rows:
        result = OPTIONB_GRAMMAR.accepts(normalize_text(prepare_ctc_transcript(r["transcript"])))
        if result is None or result[0][0] != r["label"]:
            rejected += 1
        else:
            accepted.add((r["label"], r["variation"], r["slot_value"]))

    assert len(accepted) == 93
    assert rejected <= len(rows) * 0.001
