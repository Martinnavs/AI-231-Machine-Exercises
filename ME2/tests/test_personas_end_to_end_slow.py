"""Real, end-to-end slow test for the persona-batch CLI path (Ticket 02 of the
persona-batch-generation plan).

Invokes the actual shipped entry point, `me2_voicegen.generate_personas.main()`,
with real CLI-style argv - not a hand-rolled loop over the library pieces -
against the real downloaded CosyVoice2-0.5B weights and the real GPU, using a
2-entry persona manifest built in tmp_path from the vendored
zero_shot_prompt.wav asset. No mocking of CosyVoice internals, AutoModel,
ONNX Runtime, or of any part of generate_personas.py itself: argument parsing,
the shared-run-timestamp output naming, the per-persona try/except handling,
and the exit code all run for real here (R3-1 review fix - the prior version
of this test only exercised load_personas/create_synthesizer/synthesize/
save_wav directly and left the real CLI entry point with no automated
real-GPU regression coverage). Skipped (not failed) when the weights or the
vendored asset aren't present on this machine, mirroring
tests/test_end_to_end_slow.py's skip pattern, since that's an environment
precondition, not a code defect.

Both manifest entries point at the same vendored source clip
(zero_shot_prompt.wav) under two different persona names, because no
distinct-persona reference clips exist in this repo (see Ticket 01/02's
"Unknowns"). Identical voice timbre across the two output wavs is therefore
EXPECTED in this test and is not a bug - it only proves the mechanical path
(one model load, N distinctly-named non-silent outputs). Verifying that two
genuinely different reference clips actually produce two distinct timbres is
a manual, out-of-scope step the user performs separately with real persona
clips (see plan Non-Goals).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from me2_voicegen.generation import generate_personas
from me2_voicegen.generation.download_model import missing_manifest_entries, target_dir
from me2_voicegen.synthesis.cosyvoice2_backend import CosyVoice2Synthesizer

pytestmark = pytest.mark.slow


def _skip_reasons() -> list[str]:
    reasons = []
    model_dir = target_dir()
    if not model_dir.is_dir():
        reasons.append("<entire model_dir missing>")
    else:
        reasons.extend(missing_manifest_entries(model_dir))
    if not CosyVoice2Synthesizer.DEFAULT_PROMPT_WAV.is_file():
        reasons.append(
            f"vendored asset missing: {CosyVoice2Synthesizer.DEFAULT_PROMPT_WAV}"
        )
    return reasons


@pytest.mark.skipif(
    bool(_skip_reasons()),
    reason=(
        "real CosyVoice2-0.5B weights and/or the vendored CosyVoice asset are "
        f"not present/complete (missing: {_skip_reasons()}); run "
        "`make download-model` and ensure vendor/CosyVoice is cloned first. "
        "Not mocked per this test's explicit scope."
    ),
)
def test_persona_batch_cli_produces_distinct_wavs_from_single_model_load(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "personas.json"
    manifest_path.write_text(
        json.dumps(
            {
                "personas": [
                    {
                        "name": "english_woman",
                        "wav_path": str(CosyVoice2Synthesizer.DEFAULT_PROMPT_WAV),
                        "text": CosyVoice2Synthesizer.DEFAULT_PROMPT_TEXT,
                    },
                    {
                        "name": "indian_man",
                        "wav_path": str(CosyVoice2Synthesizer.DEFAULT_PROMPT_WAV),
                        "text": CosyVoice2Synthesizer.DEFAULT_PROMPT_TEXT,
                    },
                ]
            }
        )
    )

    out_dir = tmp_path / "out"
    text = "Hey computer, could you please turn on the lights in the living room?"

    exit_code = generate_personas.main(
        [
            "--manifest",
            str(manifest_path),
            "--backend",
            "cosyvoice2",
            "--text",
            text,
            "--out-dir",
            str(out_dir),
            "--opt",
            f"model_dir={target_dir()}",
            "--opt",
            "fp16=false",
        ]
    )

    assert exit_code == 0

    wavs = sorted(out_dir.glob("sample_*.wav"))
    assert len(wavs) == 2
    assert any(p.name.startswith("sample_english_woman_") for p in wavs)
    assert any(p.name.startswith("sample_indian_man_") for p in wavs)
    assert wavs[0].name != wavs[1].name

    timestamps = {p.name.rsplit("_", 1)[-1] for p in wavs}
    assert len(timestamps) == 1

    for wav_path in wavs:
        data, sr = sf.read(str(wav_path), dtype="float32")
        assert sr == 24000
        assert len(data) > 0
        rms = np.sqrt(np.mean(data.astype(np.float64) ** 2))
        assert rms > 0.0, f"{wav_path} is silent (RMS == 0)"
