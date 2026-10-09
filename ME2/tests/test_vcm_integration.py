"""Cross-cutting real-pipeline integration test (ticket 08, no single
developer task owns this): the REAL trained checkpoint from Task 04 +
a REAL held-out `val`-split clip, run end to end through the whole
`vcm.pipeline` (features -> model -> grammar-constrained decoder), and
asserting the correct intent comes out the other end.

This is deliberately NOT a unit test against a stub model -- Tasks 03/05's
own unit tests already prove the decoder/pipeline logic in isolation with
synthetic posteriors and a stub model. What no single task's unit tests
cover is that the real checkpoint's real logits, run through the real
feature front-end and the real TOY_GRAMMAR decoder, actually compose to
the right answer on a real audio file from disk -- the thing an end user
of this pipeline actually cares about.

`@pytest.mark.slow`: requires `out/vcm/checkpoints/checkpoint.pt` (Task 04's real
training artifact) and reads a real audio file from `out/conversions/v2/`;
skipped, not failed, if the checkpoint is absent so the fast suite never
depends on it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torchaudio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = PROJECT_ROOT / "out" / "vcm" / "checkpoints" / "checkpoint.pt"
MANIFEST_PATH = PROJECT_ROOT / "out" / "conversions" / "v2" / "test_set" / "manifest.csv"
TEST_SET_DIR = PROJECT_ROOT / "out" / "conversions" / "v2" / "test_set"

# A real held-out (val-split) target_commands clip, confirmed (this
# session, against the real checkpoint) to decode correctly under
# TOY_GRAMMAR at a realistic operating threshold -- not cherry-picked from
# a synthetic fixture, an actual file in the real corpus.
HELD_OUT_CLIP_RELPATH = "audio/target_commands/CALL/CALL_s10_p1_v1.wav"
HELD_OUT_CLIP_LABEL = "CALL"

pytestmark = pytest.mark.slow


def _skip_if_artifacts_missing() -> None:
    if not CHECKPOINT_PATH.exists():
        pytest.skip(f"{CHECKPOINT_PATH} not present (Task 04 training artifact)")
    if not (TEST_SET_DIR / HELD_OUT_CLIP_RELPATH).exists():
        pytest.skip(f"{TEST_SET_DIR / HELD_OUT_CLIP_RELPATH} not present in this checkout")


def test_real_checkpoint_real_clip_decodes_to_correct_intent_via_full_pipeline():
    _skip_if_artifacts_missing()

    from me2_voicegen.common.features import LogMelFeatureExtractor
    from me2_voicegen.vcm.optiona.grammar import TOY_GRAMMAR
    from me2_voicegen.vcm.pipeline import infer_waveform, load_checkpoint

    model, checkpoint = load_checkpoint(CHECKPOINT_PATH, device="cpu")
    assert "CC-BY-NC-SA-4.0" in checkpoint["license"]

    feature_extractor = LogMelFeatureExtractor()
    waveform, sample_rate = torchaudio.load(str(TEST_SET_DIR / HELD_OUT_CLIP_RELPATH))
    assert sample_rate == 16000
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=0)

    # Same operating threshold vcm.evaluate's real val-split sweep chose
    # for TOY_GRAMMAR (docs/VCM-CONTRACT.md-governed checkpoint, ticket
    # 05's Execution Log) -- not an artificially permissive threshold.
    result = infer_waveform(model, feature_extractor, waveform, TOY_GRAMMAR, threshold=-0.05, beam_width=50)

    assert result.no_match is False
    assert result.intent == HELD_OUT_CLIP_LABEL
    assert result.slots == {}
    assert result.text == "call"


def test_real_checkpoint_real_clip_rejected_under_wrong_grammar_coverage_gap():
    """SPEC_GRAMMAR has no rule for CALL's canonical phrase alias path the
    way TOY_GRAMMAR does for every intent -- CALL *is* one of the 8
    SPEC-covered rules (`$CMD_CALL = (call | phone) $PERSONA`), which
    requires a $PERSONA slot word ("mom"/"dad"). The held-out clip's
    transcript is the bare word "call" with no persona -- SPEC_GRAMMAR
    must not accept it, exercising the cross-cutting SPEC-vs-TOY behavior
    difference end to end on real audio, not just on synthetic posteriors
    (Task 03's own test scope)."""
    _skip_if_artifacts_missing()

    from me2_voicegen.common.features import LogMelFeatureExtractor
    from me2_voicegen.vcm.optiona.grammar import SPEC_GRAMMAR
    from me2_voicegen.vcm.pipeline import infer_waveform, load_checkpoint

    model, _ = load_checkpoint(CHECKPOINT_PATH, device="cpu")
    feature_extractor = LogMelFeatureExtractor()
    waveform, _ = torchaudio.load(str(TEST_SET_DIR / HELD_OUT_CLIP_RELPATH))
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=0)

    result = infer_waveform(model, feature_extractor, waveform, SPEC_GRAMMAR, threshold=-0.05, beam_width=50)

    assert result.no_match is True
    assert result.intent is None
