"""Fast, CPU-only tests for `vcm.streaming.sources`. No test here spawns
a real subprocess or touches a real audio device: `MicrophoneSource` is
exercised via a fake `subprocess.Popen` substitute (monkeypatched),
never the real thing.
"""

from __future__ import annotations

import io
import struct

import numpy as np
import pytest
import torch
import torchaudio

from me2_voicegen.vcm.streaming import sources
from me2_voicegen.vcm.streaming.sources import (
    DEFAULT_MIC_COMMAND,
    MicrophoneSource,
    MicrophoneUnavailableError,
    RawPcmStreamSource,
    WavFileSource,
    open_microphone_source,
)
from me2_voicegen.common.features import SAMPLE_RATE


def s16_bytes(values):
    return struct.pack(f"<{len(values)}h", *values)


class TestRawPcmStreamSource:
    def test_round_trips_known_values_including_negative_full_scale(self):
        values = [0, 1, -1, 32767, -32768, 16384, -16384]
        stream = io.BytesIO(s16_bytes(values))
        src = RawPcmStreamSource(stream, block_samples=len(values))
        blocks = list(src.blocks())
        assert len(blocks) == 1
        expected = np.array(values, dtype=np.float32) / 32768.0
        np.testing.assert_allclose(blocks[0], expected, rtol=0, atol=1e-7)
        assert blocks[0][4] == pytest.approx(-1.0)
        src.close()

    def test_block_sizes_exact_except_final_short_block(self):
        values = list(range(-10, 10))  # 20 samples
        stream = io.BytesIO(s16_bytes(values))
        src = RawPcmStreamSource(stream, block_samples=7)
        blocks = list(src.blocks())
        sizes = [b.size for b in blocks]
        assert sizes == [7, 7, 6]
        src.close()

    def test_odd_trailing_byte_at_eof_does_not_raise(self):
        values = [1, 2, 3]
        raw = s16_bytes(values) + b"\x99"  # one dangling odd byte
        stream = io.BytesIO(raw)
        src = RawPcmStreamSource(stream, block_samples=4)
        blocks = list(src.blocks())
        assert len(blocks) == 1
        np.testing.assert_allclose(
            blocks[0], np.array(values, dtype=np.float32) / 32768.0
        )
        src.close()

    def test_tolerates_short_partial_reads(self):
        class ChunkyStream:
            """Returns at most 3 bytes per .read() call regardless of n,
            simulating a slow/partial pipe read."""

            def __init__(self, data: bytes) -> None:
                self._data = data
                self._pos = 0
                self.closed = False

            def read(self, n):
                chunk = self._data[self._pos : self._pos + min(n, 3)]
                self._pos += len(chunk)
                return chunk

            def close(self):
                self.closed = True

        values = list(range(10))
        stream = ChunkyStream(s16_bytes(values))
        src = RawPcmStreamSource(stream, block_samples=10)
        blocks = list(src.blocks())
        assert len(blocks) == 1
        assert blocks[0].size == 10
        np.testing.assert_allclose(
            blocks[0], np.array(values, dtype=np.float32) / 32768.0
        )
        src.close()
        assert stream.closed

    def test_close_is_idempotent(self):
        stream = io.BytesIO(s16_bytes([1, 2, 3, 4]))
        src = RawPcmStreamSource(stream, block_samples=4)
        src.close()
        src.close()  # must not raise

    def test_context_manager_closes_stream(self):
        stream = io.BytesIO(s16_bytes([1, 2]))
        with RawPcmStreamSource(stream, block_samples=2) as src:
            list(src.blocks())
        assert stream.closed

    def test_is_realtime_false(self):
        src = RawPcmStreamSource(io.BytesIO(b""), block_samples=4)
        assert src.is_realtime is False
        src.close()


class TestWavFileSource:
    def test_yields_expected_block_sizes(self, vcm_wav_factory):
        wav_path = vcm_wav_factory("tone.wav", duration_s=0.1, sample_rate=SAMPLE_RATE)
        n_samples = int(0.1 * SAMPLE_RATE)
        src = WavFileSource(wav_path, block_samples=512)
        blocks = list(src.blocks())
        assert sum(b.size for b in blocks) == n_samples
        for b in blocks[:-1]:
            assert b.size == 512
        assert blocks[-1].size == n_samples - 512 * (len(blocks) - 1)
        src.close()

    def test_is_realtime_false_by_default(self, vcm_wav_factory):
        wav_path = vcm_wav_factory("tone2.wav", duration_s=0.05)
        src = WavFileSource(wav_path, block_samples=256)
        assert src.is_realtime is False
        src.close()

    def test_resamples_non_16khz_wav(self, tmp_path):
        other_rate = 8000
        n_samples = int(0.2 * other_rate)
        waveform = 0.1 * torch.sin(
            2 * torch.pi * 440.0 * torch.arange(n_samples, dtype=torch.float32) / other_rate
        )
        path = tmp_path / "low_rate.wav"
        torchaudio.save(str(path), waveform.unsqueeze(0), other_rate)

        src = WavFileSource(path, block_samples=256)
        total_samples = sum(b.size for b in src.blocks())
        src.close()

        expected_samples = round(n_samples * SAMPLE_RATE / other_rate)
        assert total_samples == pytest.approx(expected_samples, abs=2)

    def test_resample_warns_on_stderr(self, tmp_path, capsys):
        other_rate = 8000
        waveform = torch.zeros(1, int(0.1 * other_rate))
        path = tmp_path / "low_rate2.wav"
        torchaudio.save(str(path), waveform, other_rate)

        src = WavFileSource(path, block_samples=64)
        list(src.blocks())
        src.close()

        captured = capsys.readouterr()
        assert "resampling" in captured.err

    def test_downmixes_stereo_to_mono(self, tmp_path):
        n_samples = int(0.1 * SAMPLE_RATE)
        left = torch.full((n_samples,), 0.2)
        right = torch.full((n_samples,), -0.2)
        waveform = torch.stack([left, right])
        path = tmp_path / "stereo.wav"
        torchaudio.save(str(path), waveform, SAMPLE_RATE)

        src = WavFileSource(path, block_samples=n_samples)
        blocks = list(src.blocks())
        src.close()
        assert len(blocks) == 1
        np.testing.assert_allclose(blocks[0], np.zeros(n_samples), atol=1e-6)

    def test_close_is_idempotent(self, vcm_wav_factory):
        wav_path = vcm_wav_factory("tone3.wav", duration_s=0.02)
        src = WavFileSource(wav_path, block_samples=64)
        src.close()
        src.close()

    def test_realtime_self_paces(self, vcm_wav_factory):
        import time

        wav_path = vcm_wav_factory("tone4.wav", duration_s=0.3, sample_rate=SAMPLE_RATE)
        src = WavFileSource(wav_path, block_samples=SAMPLE_RATE // 10, realtime=True)
        start = time.monotonic()
        list(src.blocks())
        elapsed = time.monotonic() - start
        src.close()
        assert elapsed >= 0.25  # ~0.3s of audio, allow scheduling slack


class FakePopen:
    """Stand-in for `subprocess.Popen` used to drive `MicrophoneSource`
    without ever spawning a real process."""

    instances = []

    def __init__(
        self,
        command,
        stdout=None,
        stderr=None,
        shell=False,
        pcm_values=None,
        exit_code=None,
        ignores_terminate=False,
    ):
        assert shell is False
        self.command = command
        self.pcm_values = pcm_values if pcm_values is not None else []
        self._exit_code = exit_code
        self.stdout = io.BytesIO(s16_bytes(self.pcm_values)) if self.pcm_values else io.BytesIO(b"")
        self.stderr = io.BytesIO(b"")
        self.terminated = False
        self.killed = False
        self.waited = False
        self._ignores_terminate = ignores_terminate
        FakePopen.instances.append(self)

    def poll(self):
        if self._exit_code is not None:
            return self._exit_code
        return None

    def terminate(self):
        self.terminated = True
        if not self._ignores_terminate:
            self._exit_code = -15  # SIGTERM-equivalent, our own shutdown

    def kill(self):
        self.killed = True
        self._exit_code = -9  # SIGKILL-equivalent

    def wait(self, timeout=None):
        self.waited = True
        if self._exit_code is None:
            if timeout is not None:
                raise sources.subprocess.TimeoutExpired(cmd=self.command, timeout=timeout)
            # A bare wait() with no timeout must not hang the test suite;
            # a well-behaved fake blocks only until kill()/an exit code is
            # set, which by construction (close()'s own control flow)
            # always happens before this branch is reached in practice.
            raise AssertionError("FakePopen.wait(timeout=None) called on a process with no exit code set")
        return self._exit_code


def make_fake_popen_factory(**kwargs):
    def factory(command, stdout=None, stderr=None, shell=False):
        return FakePopen(command, stdout=stdout, stderr=stderr, shell=shell, **kwargs)

    return factory


class TestMicrophoneSource:
    def test_is_realtime_true(self, monkeypatch):
        monkeypatch.setattr(
            sources.subprocess,
            "Popen",
            make_fake_popen_factory(pcm_values=[1, 2, 3, 4]),
        )
        mic = MicrophoneSource(["fake-arecord"], block_samples=4)
        assert mic.is_realtime is True
        mic.close()

    def test_blocks_reads_through_to_raw_pcm(self, monkeypatch):
        values = list(range(-4, 4))
        monkeypatch.setattr(
            sources.subprocess,
            "Popen",
            make_fake_popen_factory(pcm_values=values),
        )
        mic = MicrophoneSource(["fake-arecord"], block_samples=8)
        blocks = list(mic.blocks())
        assert len(blocks) == 1
        np.testing.assert_allclose(
            blocks[0], np.array(values, dtype=np.float32) / 32768.0
        )
        mic.close()

    def test_raises_microphone_unavailable_when_process_self_exits(self, monkeypatch):
        monkeypatch.setattr(
            sources.subprocess,
            "Popen",
            make_fake_popen_factory(exit_code=1),
        )
        with pytest.raises(MicrophoneUnavailableError, match="--source|--mic-command"):
            MicrophoneSource(["fake-arecord", "--bad-flag"], block_samples=4)

    def test_raises_microphone_unavailable_when_binary_missing(self, monkeypatch):
        def factory(command, stdout=None, stderr=None, shell=False):
            raise OSError("no such file or directory: 'fake-arecord'")

        monkeypatch.setattr(sources.subprocess, "Popen", factory)
        with pytest.raises(MicrophoneUnavailableError, match="--source|--mic-command"):
            MicrophoneSource(["fake-arecord"], block_samples=4)

    def test_deliberate_terminate_is_not_treated_as_failure(self, monkeypatch):
        """Our own close()->terminate() must never raise
        MicrophoneUnavailableError, even though the resulting returncode
        (SIGTERM, e.g. -15/1 on the real subprocess) is nonzero."""
        monkeypatch.setattr(
            sources.subprocess,
            "Popen",
            make_fake_popen_factory(pcm_values=[]),
        )
        mic = MicrophoneSource(["fake-arecord"], block_samples=4)
        mic.close()  # must not raise
        fake_proc = FakePopen.instances[-1]
        assert fake_proc.terminated
        assert fake_proc.waited

    def test_close_is_idempotent_and_reaps_process(self, monkeypatch):
        monkeypatch.setattr(
            sources.subprocess,
            "Popen",
            make_fake_popen_factory(pcm_values=[]),
        )
        mic = MicrophoneSource(["fake-arecord"], block_samples=4)
        mic.close()
        mic.close()  # must not raise
        fake_proc = FakePopen.instances[-1]
        assert fake_proc.waited

    def test_context_manager_closes_process(self, monkeypatch):
        monkeypatch.setattr(
            sources.subprocess,
            "Popen",
            make_fake_popen_factory(pcm_values=[1, 2]),
        )
        with MicrophoneSource(["fake-arecord"], block_samples=2) as mic:
            list(mic.blocks())
        fake_proc = FakePopen.instances[-1]
        assert fake_proc.terminated or fake_proc.poll() is not None

    def test_early_exit_failure_path_closes_pipes_no_fd_leak(self, monkeypatch):
        """R3-2 (1/2): when Popen succeeds but the process then exits
        immediately on its own, MicrophoneSource must raise
        MicrophoneUnavailableError WITHOUT leaking the stdout/stderr
        pipes -- there's no wrapped MicrophoneSource left afterward for
        close() to reach."""
        monkeypatch.setattr(
            sources.subprocess,
            "Popen",
            make_fake_popen_factory(exit_code=1),
        )
        with pytest.raises(MicrophoneUnavailableError):
            MicrophoneSource(["fake-arecord", "--bad-flag"], block_samples=4)

        fake_proc = FakePopen.instances[-1]
        assert fake_proc.stdout.closed
        assert fake_proc.stderr.closed
        assert fake_proc.waited

    def test_close_escalates_to_kill_when_process_ignores_terminate(self, monkeypatch):
        """R3-2 (2/2): a capture child that doesn't die on SIGTERM must
        not hang close() forever -- it gets escalated to kill() after the
        terminate-timeout, simulated here (no real unkillable process is
        spawned)."""
        monkeypatch.setattr(
            sources.subprocess,
            "Popen",
            make_fake_popen_factory(pcm_values=[], ignores_terminate=True),
        )
        mic = MicrophoneSource(["fake-arecord"], block_samples=4)
        mic.close()  # must return promptly, not hang

        fake_proc = FakePopen.instances[-1]
        assert fake_proc.terminated
        assert fake_proc.killed
        assert fake_proc.poll() == -9


class TestOpenMicrophoneSource:
    def test_shlex_splits_string_command_and_never_uses_shell(self, monkeypatch):
        captured = {}

        def factory(command, stdout=None, stderr=None, shell=False):
            captured["command"] = command
            captured["shell"] = shell
            return FakePopen(command, stdout=stdout, stderr=stderr, shell=shell, pcm_values=[])

        monkeypatch.setattr(sources.subprocess, "Popen", factory)
        mic = open_microphone_source("arecord -f S16_LE -r 16000 -c 1 -t raw -")
        assert captured["shell"] is False
        assert captured["command"] == ["arecord", "-f", "S16_LE", "-r", "16000", "-c", "1", "-t", "raw", "-"]
        mic.close()

    def test_default_command_is_used(self, monkeypatch):
        captured = {}

        def factory(command, stdout=None, stderr=None, shell=False):
            captured["command"] = command
            return FakePopen(command, stdout=stdout, stderr=stderr, shell=shell, pcm_values=[])

        monkeypatch.setattr(sources.subprocess, "Popen", factory)
        mic = open_microphone_source()
        assert captured["command"] == shlex_split_default()
        mic.close()

    def test_accepts_pre_parsed_argv_list(self, monkeypatch):
        captured = {}

        def factory(command, stdout=None, stderr=None, shell=False):
            captured["command"] = command
            return FakePopen(command, stdout=stdout, stderr=stderr, shell=shell, pcm_values=[])

        monkeypatch.setattr(sources.subprocess, "Popen", factory)
        mic = open_microphone_source(["custom-cmd", "--flag"])
        assert captured["command"] == ["custom-cmd", "--flag"]
        mic.close()


def shlex_split_default():
    import shlex

    return shlex.split(DEFAULT_MIC_COMMAND)
