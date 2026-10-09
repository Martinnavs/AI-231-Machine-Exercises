"""Unit tests for vcm.features (the single log-mel front-end)."""

from __future__ import annotations

import torch
import torchaudio

from me2_voicegen.common.features import (
    HOP_LENGTH,
    N_MELS,
    LogMelFeatureExtractor,
    log_mel_features,
)


def _sine(duration_s: float = 0.5, sample_rate: int = 16000, freq_hz: float = 440.0) -> torch.Tensor:
    n = int(duration_s * sample_rate)
    t = torch.arange(n, dtype=torch.float32) / sample_rate
    return 0.3 * torch.sin(2 * torch.pi * freq_hz * t)


def test_output_shape_is_b_n_mels_t_contract():
    waveform = _sine(duration_s=0.5)
    features = LogMelFeatureExtractor()(waveform)
    assert features.shape[0] == N_MELS
    expected_frames = waveform.shape[-1] // HOP_LENGTH + 1
    assert features.shape[1] == expected_frames


def test_stereo_input_is_averaged_to_mono():
    mono = _sine(duration_s=0.3)
    stereo = torch.stack([mono, mono])
    extractor = LogMelFeatureExtractor()
    mono_out = extractor(mono)
    stereo_out = extractor(stereo)
    assert torch.allclose(mono_out, stereo_out, atol=1e-5)


def test_no_log_of_zero_on_silence():
    silence = torch.zeros(8000)
    features = LogMelFeatureExtractor()(silence)
    assert torch.isfinite(features).all()


def test_per_utterance_normalization_zero_mean_unit_ish_std():
    waveform = _sine(duration_s=0.4)
    features = LogMelFeatureExtractor()(waveform)
    assert abs(features.mean().item()) < 1e-4
    assert abs(features.std().item() - 1.0) < 0.05


def test_deterministic_for_same_input():
    waveform = _sine(duration_s=0.3)
    a = LogMelFeatureExtractor()(waveform)
    b = LogMelFeatureExtractor()(waveform)
    assert torch.equal(a, b)


def test_log_mel_features_functional_matches_module():
    waveform = _sine(duration_s=0.3)
    a = LogMelFeatureExtractor()(waveform)
    b = log_mel_features(waveform)
    assert torch.equal(a, b)


def test_single_front_end_is_the_one_torchaudio_melspectrogram_used(tmp_path):
    # Guards against a second, independently-configured MelSpectrogram
    # elsewhere drifting from this one (train/infer skew is the named
    # risk in docs/VCM-CONTRACT.md section 5 / ticket 02's Action step 1).
    extractor = LogMelFeatureExtractor()
    assert isinstance(extractor._mel, torchaudio.transforms.MelSpectrogram)
    assert extractor._mel.spectrogram.n_fft == 480
    assert extractor._mel.spectrogram.win_length == 480
    assert extractor._mel.spectrogram.hop_length == 160
    assert extractor._mel.n_mels == 40
