from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from me2_voicegen.synthesis.base import SynthesisResult, save_wav


def test_save_wav_writes_mono_2d_audio_correctly(tmp_path: Path) -> None:
    audio = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)  # (1 channel, 4 samples)
    result = SynthesisResult(audio=audio, sample_rate=24000)
    out_path = tmp_path / "mono.wav"

    save_wav(result, out_path)

    data, sr = sf.read(str(out_path), dtype="float32")
    assert sr == 24000
    assert data.shape == (4,)
    assert np.allclose(data, audio[0], atol=1e-4)


def test_save_wav_writes_mono_1d_audio_correctly(tmp_path: Path) -> None:
    audio = np.array([0.5, -0.5, 0.25, -0.25], dtype=np.float32)
    result = SynthesisResult(audio=audio, sample_rate=16000)
    out_path = tmp_path / "mono_1d.wav"

    save_wav(result, out_path)

    data, sr = sf.read(str(out_path), dtype="float32")
    assert sr == 16000
    assert data.shape == (4,)
    assert np.allclose(data, audio, atol=1e-4)


def test_save_wav_writes_multichannel_audio_with_correct_layout(tmp_path: Path) -> None:
    # (2 channels, 3 samples): left channel all 0.1s, right channel all -0.1s.
    audio = np.array(
        [[0.1, 0.1, 0.1], [-0.1, -0.1, -0.1]],
        dtype=np.float32,
    )
    result = SynthesisResult(audio=audio, sample_rate=24000)
    out_path = tmp_path / "stereo.wav"

    save_wav(result, out_path)

    data, sr = sf.read(str(out_path), dtype="float32")
    assert sr == 24000
    assert data.shape == (3, 2)  # soundfile's (frames, channels) layout
    assert np.allclose(data[:, 0], audio[0], atol=1e-4)
    assert np.allclose(data[:, 1], audio[1], atol=1e-4)


def test_save_wav_creates_missing_parent_directories(tmp_path: Path) -> None:
    audio = np.zeros((1, 10), dtype=np.float32)
    result = SynthesisResult(audio=audio, sample_rate=24000)
    out_path = tmp_path / "nested" / "dirs" / "out.wav"

    assert not out_path.parent.exists()
    save_wav(result, out_path)

    assert out_path.exists()


def test_save_wav_result_is_non_silent_when_audio_is_non_zero(tmp_path: Path) -> None:
    audio = (np.arange(100, dtype=np.float32) / 100.0)[np.newaxis, :]
    result = SynthesisResult(audio=audio, sample_rate=24000)
    out_path = tmp_path / "nonsilent.wav"

    save_wav(result, out_path)

    data, _sr = sf.read(str(out_path), dtype="float32")
    rms = np.sqrt(np.mean(data.astype(np.float64) ** 2))
    assert rms > 0.0
