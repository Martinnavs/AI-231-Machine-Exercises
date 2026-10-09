"""Exactly one real, end-to-end slow test (whole-plan Ticket 04, part b).

Runs the REAL synthesis path through the factory against the real downloaded
CosyVoice2-0.5B weights and the real GPU - no mocking of CosyVoice internals,
AutoModel, or ONNX Runtime. Skipped (not failed) when the weights aren't
present on this machine, since that's an environment precondition, not a
code defect.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from me2_voicegen.generation.download_model import missing_manifest_entries, target_dir
from me2_voicegen.synthesis import save_wav
from me2_voicegen.synthesis.cosyvoice2_backend import CosyVoice2Synthesizer
from me2_voicegen.synthesis.factory import create_synthesizer
from me2_voicegen.synthesis.base import VoicePrompt

pytestmark = pytest.mark.slow


def _weights_missing() -> list[str]:
    model_dir = target_dir()
    if not model_dir.is_dir():
        return ["<entire model_dir missing>"]
    return missing_manifest_entries(model_dir)


@pytest.mark.skipif(
    bool(_weights_missing()),
    reason=(
        "real CosyVoice2-0.5B weights not present/complete at "
        f"{target_dir()} (missing: {_weights_missing()}); run `make download-model` "
        "first. Not mocked per this test's explicit scope."
    ),
)
def test_real_synthesis_through_factory_produces_valid_nonsilent_wav(tmp_path: Path) -> None:
    synthesizer = create_synthesizer(
        "cosyvoice2", model_dir=str(target_dir()), fp16=False
    )
    assert isinstance(synthesizer, CosyVoice2Synthesizer)

    prompt = VoicePrompt(
        wav_path=CosyVoice2Synthesizer.DEFAULT_PROMPT_WAV,
        text=CosyVoice2Synthesizer.DEFAULT_PROMPT_TEXT,
    )
    text = "Hey computer, could you please turn on the lights in the living room?"

    result = synthesizer.synthesize(text, prompt=prompt)

    assert result.sample_rate == 24000
    assert result.audio.dtype == np.float32
    assert result.audio.shape[-1] > 0

    out_path = tmp_path / "real_synthesis.wav"
    save_wav(result, out_path)

    assert out_path.exists()
    data, sr = sf.read(str(out_path), dtype="float32")
    assert sr == 24000
    assert len(data) > 0

    rms = np.sqrt(np.mean(data.astype(np.float64) ** 2))
    assert rms > 0.0, "output wav is silent (RMS == 0)"
