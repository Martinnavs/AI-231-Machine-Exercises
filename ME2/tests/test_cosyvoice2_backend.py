from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from me2_voicegen.synthesis import cosyvoice2_backend as backend_mod
from me2_voicegen.synthesis.base import VoicePrompt
from me2_voicegen.synthesis.cosyvoice2_backend import CosyVoice2Synthesizer


class _FakeModel:
    def __init__(self, model_dir, fp16=False, chunks=None, sample_rate=24000):
        self.model_dir = model_dir
        self.fp16 = fp16
        self.sample_rate = sample_rate
        self._chunks = chunks if chunks is not None else [torch.ones(1, 4), torch.ones(1, 6) * 2]
        self.calls: list[tuple] = []

    def inference_zero_shot(self, tts_text, prompt_text, prompt_wav):
        self.calls.append((tts_text, prompt_text, prompt_wav))
        for chunk in self._chunks:
            yield {"tts_speech": chunk}


def _make_fake_auto_model(**model_kwargs):
    def _auto_model(model_dir, fp16=False):
        return _FakeModel(model_dir, fp16=fp16, **model_kwargs)

    return _auto_model


def _patch_auto_model(monkeypatch: pytest.MonkeyPatch, **model_kwargs):
    monkeypatch.setattr(backend_mod, "_load_auto_model", lambda: _make_fake_auto_model(**model_kwargs))
    monkeypatch.setattr(backend_mod, "_default_model_dir", lambda: "/fake/model/dir")


def test_synthesize_concatenates_chunks_and_converts_to_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_auto_model(monkeypatch)
    synthesizer = CosyVoice2Synthesizer(model_dir="/fake/model/dir")

    prompt = VoicePrompt(wav_path=Path("/fake/prompt.wav"), text="hello prompt")
    result = synthesizer.synthesize("hello world", prompt=prompt)

    assert isinstance(result.audio, np.ndarray)
    assert result.audio.dtype == np.float32
    assert result.audio.shape == (1, 10)  # 4 + 6 samples concatenated
    assert result.sample_rate == 24000
    assert np.array_equal(result.audio[0, :4], np.ones(4, dtype=np.float32))
    assert np.array_equal(result.audio[0, 4:], np.full(6, 2.0, dtype=np.float32))


def test_synthesize_forwards_text_and_prompt_to_inference_zero_shot(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_auto_model(monkeypatch)
    synthesizer = CosyVoice2Synthesizer(model_dir="/fake/model/dir")

    prompt = VoicePrompt(wav_path=Path("/fake/prompt.wav"), text="a prompt transcript")
    synthesizer.synthesize("some text to say", prompt=prompt)

    assert synthesizer._model.calls == [("some text to say", "a prompt transcript", "/fake/prompt.wav")]


def test_synthesize_raises_clear_error_when_prompt_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_auto_model(monkeypatch)
    synthesizer = CosyVoice2Synthesizer(model_dir="/fake/model/dir")

    with pytest.raises(ValueError, match="requires a VoicePrompt"):
        synthesizer.synthesize("hello world", prompt=None)


def test_synthesize_raises_clear_error_when_prompt_text_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_auto_model(monkeypatch)
    synthesizer = CosyVoice2Synthesizer(model_dir="/fake/model/dir")

    prompt = VoicePrompt(wav_path=Path("/fake/prompt.wav"), text=None)
    with pytest.raises(ValueError, match="requires prompt.text"):
        synthesizer.synthesize("hello world", prompt=prompt)


def test_synthesize_raises_on_empty_chunk_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_auto_model(monkeypatch, chunks=[])
    synthesizer = CosyVoice2Synthesizer(model_dir="/fake/model/dir")

    prompt = VoicePrompt(wav_path=Path("/fake/prompt.wav"), text="a prompt transcript")
    with pytest.raises(RuntimeError, match="no audio chunks"):
        synthesizer.synthesize("hello world", prompt=prompt)


def test_default_model_dir_used_when_model_dir_not_given(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_auto_model(monkeypatch)
    synthesizer = CosyVoice2Synthesizer()

    assert synthesizer._model.model_dir == "/fake/model/dir"


def test_fp16_forwarded_to_auto_model(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_auto_model(monkeypatch)
    synthesizer = CosyVoice2Synthesizer(model_dir="/fake/model/dir", fp16=True)

    assert synthesizer._model.fp16 is True


def test_device_cuda_requested_but_unavailable_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_auto_model(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="no CUDA device is available"):
        CosyVoice2Synthesizer(model_dir="/fake/model/dir", device="cuda")


def test_device_auto_does_not_require_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_auto_model(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    synthesizer = CosyVoice2Synthesizer(model_dir="/fake/model/dir", device="auto")

    assert synthesizer._device == "auto"


def test_device_cpu_is_recorded_but_not_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_auto_model(monkeypatch)

    synthesizer = CosyVoice2Synthesizer(model_dir="/fake/model/dir", device="cpu")

    assert synthesizer._device == "cpu"
