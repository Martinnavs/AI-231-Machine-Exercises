import numpy as np
import pytest

from me2_voicegen.vcm import alphabet
from me2_voicegen.vcm.decoder import NEG_INF, decode
from me2_voicegen.vcm.optionb.grammar import OPTIONB_GRAMMAR
from me2_voicegen.vcm.segment_scorer import force_align, segment_scores
from me2_voicegen.vcm.segment_scorer_spike import (
    choose_threshold,
    crop_outer_excess_by_energy,
    decode_candidate,
    negative_periods,
    place_in_period,
    is_correct_candidate,
    threshold_sweep,
)


def _logp_for_path(path: list[int]) -> np.ndarray:
    logp = np.full((len(path), alphabet.ALPHABET_SIZE), -20.0)
    logp[np.arange(len(path)), path] = 0.0
    return logp


def test_force_align_requires_blank_between_repeated_letters():
    alignment = force_align(_logp_for_path([1, 0, 1]), "aa")
    assert alignment.frame_token_ids == (1, 0, 1)
    assert alignment.start_frame == 0
    assert alignment.end_frame == 2


def test_force_align_rejects_impossible_repeated_letter_path():
    with pytest.raises(ValueError, match="impossible"):
        force_align(_logp_for_path([1, 1]), "aa")


def test_segment_scores_ignore_outer_blank_duration():
    alignment = force_align(_logp_for_path([0, 0, 1, 2, 0, 0]), "ab")
    scores = segment_scores(_logp_for_path([0, 0, 1, 2, 0, 0]), alignment)
    assert (scores.start_frame, scores.end_frame) == (2, 3)
    assert scores.segment_path_mean == pytest.approx(0.0)
    assert scores.token_blank_margin > 10.0


def test_negative_windows_cover_final_offset_and_short_audio_is_padded():
    waveform = __import__("torch").arange(10, dtype=__import__("torch").float32)
    windows = negative_periods(waveform, period_samples=4, stride_samples=3)
    assert [window.tolist() for window in windows] == [[0, 1, 2, 3], [3, 4, 5, 6], [6, 7, 8, 9]]
    assert place_in_period(__import__("torch").ones(2), "left", 4).tolist() == [1, 1, 0, 0]


def test_outer_energy_crop_leaves_short_audio_unchanged_and_uses_contiguous_period():
    torch = __import__("torch")
    short = torch.tensor([1.0, 2.0])
    assert crop_outer_excess_by_energy(short, 3).tolist() == [1.0, 2.0]
    assert crop_outer_excess_by_energy(torch.tensor([0.0, 1.0, 5.0, 0.0, 0.0]), 3).tolist() == [0.0, 1.0, 5.0]


def test_threshold_selection_never_relaxes_zero_false_accept_rule():
    rows = [
        type("Row", (), {"kind": "positive", "correct_candidate": True, "segment_path_mean": 0.9})(),
        type("Row", (), {"kind": "positive", "correct_candidate": True, "segment_path_mean": 0.8})(),
        type("Row", (), {"kind": "negative", "correct_candidate": False, "segment_path_mean": 0.85})(),
    ]
    selected = choose_threshold(threshold_sweep(rows, "segment_path_mean"))
    assert selected == {"threshold": 0.9, "positive_correct_accepts": 1, "positive_recall": 0.5, "false_accepts": 0, "negative_windows": 1}


def test_semantic_correctness_accepts_a_spelled_numeric_surface_form():
    assert is_correct_candidate("BRIGHTNESS", {"PERCENT": "100 percent"}, "BRIGHTNESS", {"PERCENT": "100 percent"})


def test_one_pass_candidate_selection_matches_existing_permissive_decoder():
    phrase = "lights on"
    path = [alphabet.BLANK_ID]
    for token in alphabet.encode(phrase):
        path.extend((token, alphabet.BLANK_ID))
    logp = _logp_for_path(path)
    decoded = decode(logp, OPTIONB_GRAMMAR, threshold=NEG_INF, beam_width=25)
    assert not isinstance(decoded, list)
    text, intent, slots, confidence = decode_candidate(logp)
    assert (text, intent, slots) == (phrase, decoded.intent, decoded.slots)
    assert confidence == pytest.approx(decoded.confidence)
