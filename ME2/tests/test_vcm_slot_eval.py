import wave

import numpy as np
import pytest
import soundfile as sf

from me2_voicegen.generation.personas import Persona
from me2_voicegen.synthesis.base import SynthesisResult, Synthesizer, VoicePrompt
from me2_voicegen.vcm import slot_eval_set as mod


class StubSynthesizer(Synthesizer):
    """Deterministic stand-in for CosyVoice2Synthesizer: returns a real,
    audible sine wave at a configurable sample rate, so resample/verify/RMS
    logic gets exercised without torch or a vendor clone."""

    def __init__(self, sample_rate: int = 24000, silent: bool = False):
        self.sample_rate = sample_rate
        self.silent = silent
        self.calls: list[tuple[str, VoicePrompt | None]] = []

    def synthesize(self, text: str, prompt: VoicePrompt | None = None) -> SynthesisResult:
        self.calls.append((text, prompt))
        duration_s = 0.5
        n = int(self.sample_rate * duration_s)
        if self.silent:
            audio = np.zeros((1, n), dtype=np.float32)
        else:
            t = np.linspace(0, duration_s, n, endpoint=False)
            audio = (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)[np.newaxis, :]
        return SynthesisResult(audio=audio, sample_rate=self.sample_rate)


def test_slot_phrases_cover_all_five_categories():
    covered = mod.slot_categories_covered()
    assert {"NUMBER", "ARTIST", "PERSONA", "TIME_UNIT", "AMPM"} <= covered


def test_slot_phrases_count_is_twelve():
    assert len(mod.SLOT_PHRASES) == 12


def test_slot_phrases_use_only_documented_vocab():
    valid_numbers = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                      "fifteen", "twenty", "thirty"}
    valid_personas = {"mom", "dad"}
    valid_artists = {"taylor swift", "the weeknd", "bad bunny", "drake", "billie eilish"}
    valid_time_units = {"seconds", "second", "minutes", "minute", "hours", "hour"}
    valid_ampm = {"am", "pm"}

    for phrase in mod.SLOT_PHRASES:
        for slot, value in phrase.slots.items():
            if slot == "NUMBER":
                assert value in valid_numbers, value
            elif slot == "PERSONA":
                assert value in valid_personas, value
            elif slot == "ARTIST":
                assert value in valid_artists, value
            elif slot == "TIME_UNIT":
                assert value in valid_time_units, value
            elif slot == "AMPM":
                assert value in valid_ampm, value
            else:
                pytest.fail(f"unexpected slot category {slot!r}")


def test_eval_clip_filename_is_stable_and_unique():
    names = {mod.eval_clip_filename("english_woman", i) for i in range(len(mod.SLOT_PHRASES))}
    assert len(names) == len(mod.SLOT_PHRASES)


def test_write_eval_clip_resamples_to_16k_mono_16bit(tmp_path):
    synth = StubSynthesizer(sample_rate=24000)
    result = synth.synthesize("set an alarm for five am")
    dest = tmp_path / "clip.wav"

    sr, ch, sw = mod.write_eval_clip(result, dest)

    assert (sr, ch, sw) == (mod.REQUIRED_SR, mod.REQUIRED_CHANNELS, mod.REQUIRED_SAMPWIDTH)

    with wave.open(str(dest), "rb") as w:
        assert w.getframerate() == 16000
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2


def test_write_eval_clip_noop_resample_when_already_16k(tmp_path):
    synth = StubSynthesizer(sample_rate=16000)
    result = synth.synthesize("call mom")
    dest = tmp_path / "clip.wav"

    sr, ch, sw = mod.write_eval_clip(result, dest)
    assert (sr, ch, sw) == (16000, 1, 2)


def test_assert_non_silent_passes_on_real_audio(tmp_path):
    synth = StubSynthesizer(sample_rate=24000, silent=False)
    result = synth.synthesize("play music by drake")
    dest = tmp_path / "clip.wav"
    mod.write_eval_clip(result, dest)

    rms = mod.assert_non_silent(dest)
    assert rms > mod.RMS_SILENCE_FLOOR


def test_assert_non_silent_raises_on_silence(tmp_path):
    synth = StubSynthesizer(sample_rate=24000, silent=True)
    result = synth.synthesize("call dad")
    dest = tmp_path / "clip.wav"
    mod.write_eval_clip(result, dest)

    with pytest.raises(RuntimeError, match="silence floor"):
        mod.assert_non_silent(dest)


def test_build_manifest_row_matches_test_set_manifest_columns(tmp_path):
    dest = tmp_path / "english_woman_slot00.wav"
    sf.write(str(dest), np.zeros(8000, dtype=np.float32), 16000, subtype="PCM_16")

    row = mod.build_manifest_row(
        filename="english_woman_slot00.wav",
        dest_path=dest,
        out_root=tmp_path,
        intent="ALARM",
        persona_name="english_woman",
        index=0,
    )

    assert set(row) == set(mod.MANIFEST_FIELDS)
    assert row["split"] == "eval"
    assert row["sample_rate"] == "16000"
    assert row["resampled"] == "True"
    assert row["label"] == "ALARM"
    assert row["group_id"] == "english_woman"


def test_write_manifest_matches_test_set_manifest_header(tmp_path):
    row = {field: "x" for field in mod.MANIFEST_FIELDS}
    path = mod.write_manifest(tmp_path, [row])

    with path.open() as f:
        header = f.readline().strip().split(",")
    assert header == mod.MANIFEST_FIELDS
    assert header == [
        "filename",
        "path",
        "bucket",
        "label",
        "duration",
        "sample_rate",
        "resampled",
        "source_dataset",
        "source_relpath",
        "group_id",
        "split",
    ]


def test_synthesize_slot_eval_set_end_to_end_with_stub(tmp_path):
    synth = StubSynthesizer(sample_rate=24000)
    persona = Persona(name="p1", prompt=VoicePrompt(wav_path=tmp_path / "ref.wav", text="hi"))

    rows = mod.synthesize_slot_eval_set([persona], synth, tmp_path, phrases=mod.SLOT_PHRASES[:3])

    assert len(rows) == 3
    assert len(synth.calls) == 3
    for row in rows:
        clip_path = tmp_path / row["filename"]
        assert clip_path.is_file()
        with wave.open(str(clip_path), "rb") as w:
            assert w.getframerate() == 16000
            assert w.getnchannels() == 1
