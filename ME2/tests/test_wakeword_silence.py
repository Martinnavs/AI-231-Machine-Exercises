"""Tests for me2_voicegen.wakeword.generate_silence.

Pure local synthesis -- no network, no fixtures beyond numpy-generated
signals, so this suite runs fully offline. Spectral-slope assertions
independently measure the *power spectral density* of generated pink/
brown noise via a fresh FFT (not reusing generation's own spectrum-shaping
code), so a sign/exponent bug in `colored_noise` would be caught here
rather than assumed correct.
"""

from __future__ import annotations

import csv
import wave

import numpy as np
import pytest

from me2_voicegen.wakeword import generate_silence as gs


# ---------------------------------------------------------------------------
# Spectral-slope correctness (the acceptance criterion this ticket calls out
# explicitly: don't just trust the generation code).
# ---------------------------------------------------------------------------


def measure_psd_slope_db_per_octave(samples: np.ndarray, sample_rate: int) -> float:
    """Fit a line to log2(power spectral density) vs log2(frequency) over
    the mid-band (excludes DC/near-DC and the very top of the band, where
    windowing/aliasing effects distort the estimate), and return the slope
    in dB/octave (10*log10(2) per unit of the log2-power fit == a
    doubling of frequency)."""
    n = len(samples)
    spectrum = np.fft.rfft(samples)
    psd = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    # Restrict to a clean mid-band: above a low-frequency cutoff (avoid the
    # near-DC bins, which are noisy/undefined for the 1/f, 1/f^2 shaping)
    # and below Nyquist * 0.9 (avoid edge effects).
    mask = (freqs > 50.0) & (freqs < sample_rate * 0.45)
    freqs = freqs[mask]
    psd = psd[mask]

    log_f = np.log2(freqs)
    log_p = np.log2(psd + 1e-20)

    slope_per_log2f, _intercept = np.polyfit(log_f, log_p, 1)
    # slope_per_log2f is "dB-equivalent power change per octave" already
    # once expressed in log2(power); convert log2-power-units to dB
    # (dB = 10*log10(power)):
    slope_db_per_octave = slope_per_log2f * 10.0 * np.log10(2.0)
    return float(slope_db_per_octave)


@pytest.mark.parametrize(
    "color,expected_slope,tolerance",
    [
        ("white", 0.0, 1.5),
        ("pink", -3.0, 1.5),
        ("brown", -6.0, 1.5),
    ],
)
def test_colored_noise_spectral_slope(color, expected_slope, tolerance):
    rng = np.random.default_rng(12345)
    n_samples = 16000 * 8  # long chunk for a stable PSD estimate
    noise = gs.colored_noise(n_samples, color, rng)

    slope = measure_psd_slope_db_per_octave(noise, gs.REQUIRED_SR)
    assert slope == pytest.approx(expected_slope, abs=tolerance), (
        f"{color} noise measured {slope:.2f}dB/octave, expected "
        f"{expected_slope}dB/octave +/- {tolerance}"
    )


def test_pink_noise_is_not_actually_white():
    """Guards specifically against the bug this ticket calls out: a sign/
    exponent error that produces "pink" noise indistinguishable from
    white."""
    rng = np.random.default_rng(1)
    n_samples = 16000 * 8
    pink = gs.colored_noise(n_samples, "pink", rng)
    white = gs.colored_noise(n_samples, "white", rng)

    pink_slope = measure_psd_slope_db_per_octave(pink, gs.REQUIRED_SR)
    white_slope = measure_psd_slope_db_per_octave(white, gs.REQUIRED_SR)
    assert pink_slope < white_slope - 1.0


def test_colored_noise_rejects_unknown_color():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        gs.colored_noise(100, "purple", rng)


def test_colored_noise_is_unit_std():
    rng = np.random.default_rng(0)
    for color in gs.NOISE_COLORS:
        noise = gs.colored_noise(16000, color, np.random.default_rng(0))
        assert np.std(noise) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# render_chunk: RMS targeting + peak ceiling
# ---------------------------------------------------------------------------


def test_render_chunk_hits_target_rms():
    rng = np.random.default_rng(42)
    samples = gs.render_chunk("white", -20.0, 2.0, rng)
    measured_rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2))
    measured_dbfs = 20.0 * np.log10(measured_rms)
    assert measured_dbfs == pytest.approx(-20.0, abs=0.5)


def test_render_chunk_respects_duration():
    rng = np.random.default_rng(0)
    samples = gs.render_chunk("pink", -30.0, 1.5, rng)
    assert len(samples) == int(round(1.5 * gs.REQUIRED_SR))


def test_render_chunk_never_clips_past_peak_ceiling():
    rng = np.random.default_rng(0)
    # Loud target, near full scale, to exercise the peak-ceiling clamp.
    samples = gs.render_chunk("brown", -1.0, 1.0, rng)
    assert np.max(np.abs(samples)) <= gs.PEAK_CEILING + 1e-6


# ---------------------------------------------------------------------------
# largest_remainder_allocation
# ---------------------------------------------------------------------------


def test_largest_remainder_allocation_sums_to_total():
    result = gs.largest_remainder_allocation(100, [1.0] * 9)
    assert sum(result) == 100
    assert max(result) - min(result) <= 1


def test_largest_remainder_allocation_zero_weights_raises():
    with pytest.raises(ValueError):
        gs.largest_remainder_allocation(10, [0.0, 0.0])


# ---------------------------------------------------------------------------
# plan_chunks: seeded determinism, group coverage
# ---------------------------------------------------------------------------


def test_plan_chunks_is_deterministic_for_same_seed():
    plan1 = gs.plan_chunks(90, seed=7)
    plan2 = gs.plan_chunks(90, seed=7)
    assert plan1 == plan2


def test_plan_chunks_differs_across_seeds():
    plan1 = gs.plan_chunks(90, seed=7)
    plan2 = gs.plan_chunks(90, seed=8)
    assert plan1 != plan2


def test_plan_chunks_covers_all_color_band_groups():
    plan = gs.plan_chunks(90, seed=0)
    groups = {gs.group_id_for(e["color"], e["rms_band"]) for e in plan}
    expected = {gs.group_id_for(c, b) for c in gs.NOISE_COLORS for b in gs.RMS_BANDS}
    assert groups == expected


def test_plan_chunks_total_count_matches():
    plan = gs.plan_chunks(123, seed=0)
    assert len(plan) == 123


def test_plan_chunks_duration_within_real_ambient_range():
    plan = gs.plan_chunks(300, seed=0)
    for entry in plan:
        assert gs.MIN_DURATION_S <= entry["duration_s"] <= gs.MAX_DURATION_S


# ---------------------------------------------------------------------------
# End-to-end: dry-run, real run, byte-identical reproducibility, format
# verification.
# ---------------------------------------------------------------------------


def test_dry_run_reports_without_writing(tmp_path):
    out_root = tmp_path / "out" / "silence_synthetic"
    rc = gs.main(["--out-root", str(out_root), "--total-count", "18", "--seed", "1", "--dry-run"])
    assert rc == 0
    assert not out_root.exists()


def test_full_run_writes_manifest_with_all_colors_and_verified_wavs(tmp_path):
    out_root = tmp_path / "out" / "silence_synthetic"
    rc = gs.main(["--out-root", str(out_root), "--total-count", "18", "--seed", "1"])
    assert rc == 0

    with (out_root / "manifest.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 18
    assert list(rows[0].keys()) == gs.MANIFEST_FIELDS
    assert all(r["label"] == gs.LABEL_SILENCE for r in rows)
    assert all(r["source_dataset"] == gs.SOURCE_SYNTHETIC for r in rows)
    assert all(r["split"] == "" for r in rows)
    assert all(r["resampled"] == "False" for r in rows)

    colors_seen = {r["source_relpath"].split("color=", 1)[1].split(";", 1)[0] for r in rows}
    assert colors_seen == set(gs.NOISE_COLORS)

    for row in rows:
        path = out_root / row["path"]
        assert path.is_file()
        with wave.open(str(path), "rb") as w:
            assert w.getframerate() == gs.REQUIRED_SR
            assert w.getnchannels() == gs.REQUIRED_CHANNELS
            assert w.getsampwidth() == gs.REQUIRED_SAMPWIDTH
        assert float(row["duration"]) == pytest.approx(w.getnframes() / gs.REQUIRED_SR, abs=1e-4)

    assert (out_root / "summary.md").is_file()
    summary = (out_root / "summary.md").read_text(encoding="utf-8")
    assert "white" in summary and "pink" in summary and "brown" in summary


def test_full_run_is_byte_identical_across_runs_with_same_seed(tmp_path):
    out_root1 = tmp_path / "run1" / "silence_synthetic"
    out_root2 = tmp_path / "run2" / "silence_synthetic"
    rc1 = gs.main(["--out-root", str(out_root1), "--total-count", "18", "--seed", "5"])
    rc2 = gs.main(["--out-root", str(out_root2), "--total-count", "18", "--seed", "5"])
    assert rc1 == 0
    assert rc2 == 0

    manifest1 = (out_root1 / "manifest.csv").read_bytes()
    manifest2 = (out_root2 / "manifest.csv").read_bytes()
    assert manifest1 == manifest2

    with (out_root1 / "manifest.csv").open(newline="", encoding="utf-8") as f:
        rows1 = list(csv.DictReader(f))
    for row in rows1:
        assert (out_root1 / row["path"]).read_bytes() == (out_root2 / row["path"]).read_bytes()


def test_full_run_differs_across_seeds(tmp_path):
    out_root1 = tmp_path / "run1" / "silence_synthetic"
    out_root2 = tmp_path / "run2" / "silence_synthetic"
    gs.main(["--out-root", str(out_root1), "--total-count", "18", "--seed", "1"])
    gs.main(["--out-root", str(out_root2), "--total-count", "18", "--seed", "2"])

    manifest1 = (out_root1 / "manifest.csv").read_bytes()
    manifest2 = (out_root2 / "manifest.csv").read_bytes()
    assert manifest1 != manifest2


def test_full_run_is_idempotent_rerun_in_place(tmp_path):
    out_root = tmp_path / "out" / "silence_synthetic"
    argv = ["--out-root", str(out_root), "--total-count", "18", "--seed", "3"]
    rc1 = gs.main(argv)
    assert rc1 == 0
    manifest1 = (out_root / "manifest.csv").read_bytes()

    rc2 = gs.main(argv)
    assert rc2 == 0
    manifest2 = (out_root / "manifest.csv").read_bytes()
    assert manifest1 == manifest2


def test_main_rejects_non_positive_total_count(tmp_path):
    out_root = tmp_path / "out" / "silence_synthetic"
    rc = gs.main(["--out-root", str(out_root), "--total-count", "0"])
    assert rc == 1
    assert not out_root.exists()


def test_main_preserves_prior_good_output_on_mid_run_failure(tmp_path, monkeypatch):
    out_root = tmp_path / "out" / "silence_synthetic"
    argv = ["--out-root", str(out_root), "--total-count", "18", "--seed", "1"]
    rc = gs.main(argv)
    assert rc == 0
    good_manifest = (out_root / "manifest.csv").read_bytes()

    def broken_render_and_write(plan, rows, out_root, seed):
        raise RuntimeError("simulated mid-run failure")

    monkeypatch.setattr(gs, "render_and_write", broken_render_and_write)
    rc2 = gs.main(argv)
    assert rc2 == 1

    assert out_root.is_dir()
    assert (out_root / "manifest.csv").read_bytes() == good_manifest
    staging_root = out_root.parent / f".{out_root.name}.staging"
    assert not staging_root.exists()


# ---------------------------------------------------------------------------
# group_id granularity
# ---------------------------------------------------------------------------


def test_group_id_scheme_is_color_and_rms_band():
    assert gs.group_id_for("pink", "moderate") == "pink_moderate"


def test_build_manifest_rows_rejects_duplicate_destination():
    plan = gs.plan_chunks(2, seed=0)
    plan = [plan[0], plan[0]]  # force a duplicate (color, band, index)
    with pytest.raises(gs.ManifestValidationError, match="duplicate"):
        gs.build_manifest_rows(plan, out_root=None)
