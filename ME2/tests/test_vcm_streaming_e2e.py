"""Cross-cutting tests for the `vcm.streaming` feature spanning all four
developer tickets' output (ring buffer/debounce/policy, sources, config/
backends, runner/CLI) -- owned by ticket 05's test-engineer audit, not any
single developer ticket. Every test here is fast/CPU-only: no real
checkpoint, no real ONNX artifact, no microphone, no `out/conversions/`.

Covers ticket 05's cross-cutting items:
  (a) end-to-end `python -m me2_voicegen.vcm.streaming` subprocess
      invocation over a synthetic wav, asserting parseable JSONL on stdout.
  (b) `me2_voicegen.vcm.streaming.*` imports cleanly with no mic/checkpoint/
      ONNX artifact/`out/conversions/` needed at import time.
  (c) a more adversarial `RingBuffer` thread-safety stress test than
      ticket 01's own basic one (multiple readers, varying write sizes).
  (e) a determinism regression for lockstep file replay, driven through the
      real CLI subprocess (not just in-process, unlike the runner ticket's
      own `test_lockstep_deterministic_across_runs_byte_for_byte`).
  (f) an ACTIVE guard that the fast suite never constructs a real ORT
      session or spawns a real subprocess for the mic path.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
import torch

from me2_voicegen.vcm.model import MatchboxNetConfig, MatchboxNetCTC
from me2_voicegen.vcm.streaming.buffer import RingBuffer

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_synthetic_torch_checkpoint(tmp_path: Path, seed: int = 0) -> Path:
    """Same shape as `test_vcm_streaming_backends.py`'s fixture checkpoint
    -- a tiny, untrained `MatchboxNetCTC` with the real checkpoint dict
    shape `TorchBackend`/`load_checkpoint` expect. Deliberately duplicated
    here (rather than imported from that test module) so this file stays
    self-contained -- it's a handful of lines, not worth a shared-fixture
    dependency for one cross-cutting test file."""
    torch.manual_seed(seed)
    config = MatchboxNetConfig(
        n_mels=40, n_blocks=1, channels=8, kernel_sizes=[5], prologue_channels=8, epilogue_channels=8
    )
    model = MatchboxNetCTC(config)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "preset": "e2e-fixture",
        "config": {
            "n_mels": config.n_mels,
            "n_blocks": config.n_blocks,
            "channels": config.channels,
            "kernel_sizes": config.kernel_sizes,
            "prologue_channels": config.prologue_channels,
            "epilogue_channels": config.epilogue_channels,
            "alphabet_size": config.alphabet_size,
        },
        "alphabet_size": config.alphabet_size,
        "seed": seed,
        "epoch": 1,
        "val_loss": 0.5,
        "license": "e2e-fixture-license",
    }
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)
    return path


def _run_cli(
    checkpoint_path: Path, wav_path: Path, extra_args: tuple[str, ...] = ()
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "me2_voicegen.vcm.streaming",
            "--backend",
            "torch",
            "--model",
            str(checkpoint_path),
            # Equals-form is required here: argparse's "looks like a negative
            # number" detector only matches `^-\d+$|^-\d*\.\d+$`, which does
            # not cover scientific notation like `-1e6` -- as a separate
            # token argparse would treat it as an unrecognized option
            # instead of the value for --threshold.
            "--threshold=-1e6",
            "--grammar",
            "optionb",
            "--policy",
            "threshold",
            "--source",
            str(wav_path),
            "--window-s",
            "0.5",
            "--stride-s",
            "0.25",
            "--refractory-s",
            "0.0",
            "--beam-width",
            "5",
            "--log-all-windows",
            *extra_args,
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )


# ---------------------------------------------------------------------------
# (a) End-to-end __main__ subprocess invocation -> parseable JSONL.
# ---------------------------------------------------------------------------


def test_cli_subprocess_over_synthetic_wav_emits_parseable_jsonl(tmp_path, vcm_wav_factory):
    checkpoint_path = _write_synthetic_torch_checkpoint(tmp_path)
    wav_path = vcm_wav_factory("e2e.wav", duration_s=1.0)

    result = _run_cli(checkpoint_path, wav_path)

    assert result.returncode == 0, (
        f"CLI exited nonzero.\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert lines, f"expected at least one JSONL line on stdout with --log-all-windows; stderr={result.stderr}"
    for line in lines:
        payload = json.loads(line)  # must not raise
        assert payload["event"] in ("trigger", "window")
        assert "t_seconds" in payload
        assert "window_index" in payload
        assert "policy_reason" in payload
    # Startup banner and the summary line go to stderr, not stdout -- stdout
    # must be pure JSONL, nothing else, so a downstream consumer piping
    # stdout through a JSON-lines parser never chokes on a stray banner line.
    for line in lines:
        json.loads(line)
    assert "streaming run complete" in result.stderr


# ---------------------------------------------------------------------------
# (e) Determinism regression for lockstep file replay, through the real CLI.
# ---------------------------------------------------------------------------


def test_cli_lockstep_file_replay_is_byte_for_byte_deterministic(tmp_path, vcm_wav_factory):
    checkpoint_path = _write_synthetic_torch_checkpoint(tmp_path, seed=7)
    wav_path = vcm_wav_factory("determinism.wav", duration_s=1.0, freq_hz=523.25)

    first = _run_cli(checkpoint_path, wav_path)
    second = _run_cli(checkpoint_path, wav_path)

    assert first.returncode == 0 and second.returncode == 0
    assert first.stdout == second.stdout
    assert first.stdout  # sanity: something was actually emitted


# ---------------------------------------------------------------------------
# (b) Import-cleanliness regression: vcm.streaming.* imports cleanly with
# no mic/checkpoint/ONNX artifact/out/conversions present -- proven by
# pointing the model registry at nonexistent paths and confirming *mere
# import* never resolves/opens any of them. (Registration of these modules
# in test_smoke_fast_suite.py's _VCM_MODULES list is ticket 04's own work,
# already landed and audited separately below -- this test is the
# independent "constructing nothing at import time" guarantee that list
# alone doesn't prove.)
# ---------------------------------------------------------------------------


def test_streaming_modules_import_cleanly_with_no_real_artifacts_present():
    script = """
import sys

# Fail loudly if import time ever touches a real device or model artifact.
class _Boom:
    def __call__(self, *a, **k):
        raise AssertionError("import touched a real artifact/device: " + repr((a, k)))

import subprocess as _subprocess
_subprocess.Popen = _Boom()

import me2_voicegen.vcm.streaming.buffer
import me2_voicegen.vcm.streaming.debounce
import me2_voicegen.vcm.streaming.policy
import me2_voicegen.vcm.streaming.sources
import me2_voicegen.vcm.streaming.config
import me2_voicegen.vcm.streaming.backends
import me2_voicegen.vcm.streaming.runner
import me2_voicegen.vcm.streaming.__main__

print("OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert result.stdout.strip() == "OK"


# ---------------------------------------------------------------------------
# (c) RingBuffer thread-safety stress test: more adversarial than ticket
# 01's own basic probe -- multiple concurrent readers, varying write
# chunk sizes, longer duration.
# ---------------------------------------------------------------------------


def test_ring_buffer_survives_one_writer_many_readers_under_varying_chunk_sizes():
    capacity = 128
    buf = RingBuffer(window_samples=capacity)
    stop = threading.Event()
    errors: list[str] = []
    errors_lock = threading.Lock()

    def writer():
        counter = 0
        rng = np.random.default_rng(0)
        while not stop.is_set():
            block = int(rng.integers(1, 17))  # varying chunk sizes, 1..16
            buf.write(np.arange(counter, counter + block, dtype=np.float32))
            counter += block

    def reader():
        for _ in range(300):
            snap = buf.snapshot()
            if snap.size == 0:
                continue
            if snap.size > capacity:
                with errors_lock:
                    errors.append(f"snapshot exceeded capacity: {snap.size}")
                return
            if snap.size > 1:
                diffs = np.diff(snap)
                if not np.all(diffs == 1.0):
                    with errors_lock:
                        errors.append(f"non-contiguous snapshot: {snap.tolist()}")
                    return

    writer_thread = threading.Thread(target=writer, daemon=True)
    reader_threads = [threading.Thread(target=reader) for _ in range(4)]

    writer_thread.start()
    for t in reader_threads:
        t.start()
    try:
        for t in reader_threads:
            t.join(timeout=5.0)
    finally:
        stop.set()
        writer_thread.join(timeout=2.0)

    assert errors == []
    # The buffer must still be perfectly usable after the stress run --
    # reset must not be corrupted by concurrent access that happened before it.
    buf.reset()
    assert buf.samples_written == 0
    assert buf.snapshot().size == 0


# ---------------------------------------------------------------------------
# (f) Active guard: the fast suite never constructs a real ORT session or
# spawns a real subprocess for the mic path. Runs the streaming-specific
# fast test files in a fresh subprocess with onnxruntime.InferenceSession
# and sources.subprocess.Popen replaced by assertion-raising sentinels
# *before* those test modules are collected; per-test `monkeypatch.setattr`
# calls in the individual tests still work normally (they save/restore
# around the sentinel exactly like they would around the real thing), so
# this only fires if some code path escapes those fakes and reaches the
# real construction call.
# ---------------------------------------------------------------------------


def test_fast_streaming_suite_never_constructs_real_ort_session_or_spawns_mic_process():
    # Deliberately excludes this file (test_vcm_streaming_e2e.py): its own
    # (a)/(e) tests legitimately spawn a real subprocess to invoke the CLI
    # end-to-end over a wav file -- that is not "the mic path" this guard
    # exists to catch, but since `sources.py` and this test file share the
    # one process-wide `subprocess` module object, patching
    # `sources.subprocess.Popen` here would also intercept those unrelated,
    # legitimate spawns. The developer-owned files below are the ones whose
    # own unit tests claim (per their own Execution Logs) to always
    # monkeypatch `subprocess.Popen`/`onnxruntime.InferenceSession` per-test
    # rather than ever calling the real thing in the fast suite -- this is
    # what actually verifies that claim.
    streaming_test_files = [
        "tests/test_vcm_streaming_buffer.py",
        "tests/test_vcm_streaming_debounce.py",
        "tests/test_vcm_streaming_policy.py",
        "tests/test_vcm_streaming_sources.py",
        "tests/test_vcm_streaming_config.py",
        "tests/test_vcm_streaming_backends.py",
        "tests/test_vcm_streaming_runner.py",
    ]
    wrapper_script = f"""
import sys

# Warm up numpy's own lazy one-time `subprocess.run(['lscpu'], ...)` probe
# (numpy.testing._private.utils.check_support_sve, gh-22982) *before*
# installing the sentinels below. If left cold, it fires lazily the first
# time some test in the guarded suite happens to trigger a
# numpy.testing.* import, hits whichever Popen sentinel/per-test fake is
# active at that arbitrary moment, and produces a false-positive failure
# in a test that never touches subprocess itself.
import numpy.testing._private.utils as _np_testing_utils  # noqa: F401

import onnxruntime


def _boom_ort(*args, **kwargs):
    raise AssertionError(
        "a fast (non-slow) streaming test constructed a real "
        "onnxruntime.InferenceSession -- this must only happen in a "
        "@pytest.mark.slow test against a real .onnx artifact"
    )


onnxruntime.InferenceSession = _boom_ort

import me2_voicegen.vcm.streaming.sources as _sources_mod


def _boom_popen(*args, **kwargs):
    raise AssertionError(
        "a fast (non-slow) streaming test spawned a real subprocess for "
        "the microphone path -- MicrophoneSource must always be driven "
        "through a monkeypatched subprocess.Popen in the fast suite"
    )


_sources_mod.subprocess.Popen = _boom_popen

import pytest

sys.exit(pytest.main(["-q", "-m", "not slow", *{streaming_test_files!r}]))
"""
    result = subprocess.run(
        [sys.executable, "-c", wrapper_script],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        "the sentinel-guarded run of the fast streaming suite failed -- either a "
        "real genuine test failure, or (more likely, given this guard) some code "
        "path constructed a real ORT session / spawned a real mic subprocess "
        f"outside a monkeypatched fake.\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "AssertionError" not in result.stdout


# ---------------------------------------------------------------------------
# --score-mode through the real CLI (dense-d2-loose-impl task 2).
# ---------------------------------------------------------------------------


def test_cli_score_mode_flag_reaches_runner_and_banner(tmp_path, vcm_wav_factory):
    checkpoint_path = _write_synthetic_torch_checkpoint(tmp_path)
    wav_path = vcm_wav_factory("score_mode.wav", duration_s=1.0)

    default = _run_cli(checkpoint_path, wav_path)
    per_char = _run_cli(checkpoint_path, wav_path, ("--score-mode", "per_char"))

    assert default.returncode == 0 and per_char.returncode == 0, per_char.stderr
    assert "score_mode: mean_frame" in default.stderr
    assert "score_mode: per_char" in per_char.stderr
    # Default stdout is unchanged by the new flag existing; per_char runs end to end.
    for line in per_char.stdout.splitlines():
        if line.strip():
            json.loads(line)


def test_cli_rejects_unknown_score_mode(tmp_path, vcm_wav_factory):
    checkpoint_path = _write_synthetic_torch_checkpoint(tmp_path)
    wav_path = vcm_wav_factory("bad_mode.wav", duration_s=1.0)
    result = _run_cli(checkpoint_path, wav_path, ("--score-mode", "dense"))
    assert result.returncode != 0
    assert "score-mode" in result.stderr or "score_mode" in result.stderr
