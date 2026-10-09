"""Cross-cutting smoke test (ticket 09): the real CLI entry points ticket
07/08's Makefile targets will invoke -- `incomplete_probes.main` and
`incomplete_calibration.main`'s `select`/`holdout` subcommands -- driven
end to end (probes -> select -> holdout) on a tiny synthetic manifest and a
randomly initialized checkpoint under `tmp_path`.

Rewritten per tech-lead Review Feedback R2-4 (ticket 09's Execution Log):
the original version called library pieces (`select_target_rows`,
`generate_probes`, `build_val_report`, ...) directly instead of going
through `main()`'s own `argparse` wiring, which is exactly the seam where
independently-developed CLIs disagree (that gap is what let 06's R1-1
filename mismatch and R2-3 missing guards through undetected). This version
never calls an internal function directly except to read back and assert
on the JSON/CSV artifacts the CLIs themselves wrote.

Marked `slow`: it loads a real (if tiny) `MatchboxNetCTC`, runs `main()`
four/five times end to end (each CLI invocation reloads the checkpoint,
decodes every row, force-aligns), and exercises the full beam-search
decoder at the real beam width (25). No real project checkpoint, dataset,
or `out/` artifact is touched anywhere here.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
import torch

from me2_voicegen.vcm import incomplete_calibration as ic
from me2_voicegen.vcm.model import MatchboxNetConfig, MatchboxNetCTC
from me2_voicegen.vcm.optionb import incomplete_probes as ip

pytestmark = pytest.mark.slow

BEAM_WIDTH = "25"  # matches the real select/holdout run's beam width (Established, ticket 09)


def _assert_probe_far_by_silence_source_matches_manifest(metrics: dict, probe_rows: list[dict]) -> None:
    """R2-5 follow-up (tech-lead re-review): the CLI-written report's
    `probe_far_by_silence_source` breakdown must actually be grouped by the
    real `silence_source` values the probe manifest carries -- same key set,
    same per-key `n` -- not merely present as a field."""
    breakdown = metrics["probe_far_by_silence_source"]
    expected_counts = Counter(row["silence_source"] for row in probe_rows)
    assert set(breakdown) == set(expected_counts)
    for key, count in expected_counts.items():
        assert breakdown[key]["n"] == count


def _tiny_config() -> MatchboxNetConfig:
    return MatchboxNetConfig(
        n_mels=40, n_blocks=1, channels=8, kernel_sizes=[5], prologue_channels=8, epilogue_channels=8
    )


def _write_synthetic_checkpoint(path: Path, seed: int = 0) -> None:
    """Same shape/idiom as `tests/test_vcm_streaming_backends.py`'s own
    synthetic-checkpoint helper (duplicated per convention: `tests/` is not
    a package)."""
    torch.manual_seed(seed)
    config = _tiny_config()
    model = MatchboxNetCTC(config)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "preset": "test-fixture",
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
        "license": "test-fixture-license",
    }
    torch.save(checkpoint, path)


@pytest.fixture
def _fixture(tmp_path, vcm_fake_manifest_factory):
    checkpoint_path = tmp_path / "checkpoint.pt"
    _write_synthetic_checkpoint(checkpoint_path)

    # "remind me to study" is a real OPTIONB_GRAMMAR target phrase containing
    # three designated incomplete prefixes ("remind", "remind me",
    # "remind me to"), one of which ("remind") is also a documented
    # character-overlap prefix -- so a single source row per split exercises
    # both the aggregate FAR and the overlap-flag/intent-confusion columns.
    target_transcript = "remind me to study"
    specs = [
        {
            "bucket": "target_commands",
            "source_dataset": "optionb",
            "split": "val",
            "label": "REMINDER",
            "transcript": target_transcript,
            "duration_s": 2.0,
        },
        {
            "bucket": "babble",
            "source_dataset": "common_voice_negative",
            "split": "val",
            "transcript": "some unrelated babble speech",
            "duration_s": 1.0,
        },
        {
            "bucket": "target_commands",
            "source_dataset": "optionb",
            "split": "test",
            "label": "REMINDER",
            "transcript": target_transcript,
            "duration_s": 2.0,
        },
        {
            "bucket": "babble",
            "source_dataset": "common_voice_negative",
            "split": "test",
            "transcript": "other unrelated babble speech",
            "duration_s": 1.0,
        },
    ]
    manifest_path = vcm_fake_manifest_factory(specs)
    return tmp_path, manifest_path, checkpoint_path


def test_probes_select_holdout_cli_end_to_end(_fixture):
    tmp_path, manifest_path, checkpoint_path = _fixture

    # --- ip.main: generate real probe manifests for val and test ----------
    probes_dir = tmp_path / "probes"
    for split in ("val", "test"):
        split_out_dir = probes_dir / split
        rc = ip.main(
            [
                "--manifest", str(manifest_path),
                "--checkpoint", str(checkpoint_path),
                "--split", split,
                "--out-dir", str(split_out_dir),
                "--device", "cpu",
            ]
        )
        assert rc == 0
        probe_manifest = split_out_dir / "manifest.csv"
        assert probe_manifest.exists()
        assert (split_out_dir / "generation_report.json").exists()
        probe_rows = ip._load_manifest_rows(probe_manifest)
        assert probe_rows, f"expected at least one generated probe row for split {split!r}"
        assert {row["split"] for row in probe_rows} == {split}

    val_probe_manifest = probes_dir / "val" / "manifest.csv"
    test_probe_manifest = probes_dir / "test" / "manifest.csv"

    # --- ic.main select: refused against the wrong (test) probe split -----
    wrong_split_out_dir = tmp_path / "select_wrong_split"
    with pytest.raises(SystemExit):
        ic.main(
            [
                "select",
                "--manifest", str(manifest_path),
                "--probe-manifest", str(test_probe_manifest),
                "--checkpoint", str(checkpoint_path),
                "--out-dir", str(wrong_split_out_dir),
                "--device", "cpu",
                "--beam-width", BEAM_WIDTH,
            ]
        )
    assert not (wrong_split_out_dir / "val_selection.json").exists()

    # --- ic.main select: the real val calibration run ----------------------
    select_out_dir = tmp_path / "select"
    rc = ic.main(
        [
            "select",
            "--manifest", str(manifest_path),
            "--probe-manifest", str(val_probe_manifest),
            "--checkpoint", str(checkpoint_path),
            "--out-dir", str(select_out_dir),
            "--device", "cpu",
            "--beam-width", BEAM_WIDTH,
        ]
    )
    assert rc == 0
    val_selection_path = select_out_dir / "val_selection.json"
    assert val_selection_path.exists()
    assert (select_out_dir / "val_report.md").exists()

    val_report = json.loads(val_selection_path.read_text())
    assert "no_eligible_margin" not in val_report
    assert val_report["beam_width"] == 25
    # ticket 05's generation-report failure counts must actually reach the
    # val report (06 R1-1: the two CLIs previously disagreed on this
    # filename and this field silently read as "unavailable").
    probegen = val_report["probe_generation_failures"]
    assert probegen["available"] is True
    assert probegen["by_reason"] == {
        "crop_too_short": 0, "force_align_error": 0, "missing_audio": 0, "no_gap": 0
    }

    # R2-5 follow-up (tech-lead re-review of ticket 05/06's R2-5 fix): the
    # per-silence_source FAR breakdown must actually show up in the
    # CLI-written val report, grouped by the real values the probe manifest
    # carries -- not just present as a field. This synthetic fixture's random
    # (untrained) checkpoint forces alignment across virtually the entire
    # clip (verified separately: forced alignment against a random model has
    # no acoustic signal to anchor a shorter span, so `start_frame`/
    # `end_frame` degenerate to ~0/~T-1), leaving no lead-in/tail region for
    # `find_quiet_window` to ever accept -- so this fixture reliably produces
    # exactly two `silence_source` values ("none" for the 0.0s bucket,
    # "digital_zero" for every padded bucket), never the two room-tone
    # values. That is a property of using a *randomly initialized*
    # checkpoint (this ticket's own Established constraint), not a test gap:
    # `tests/test_vcm_incomplete_calibration.py`'s
    # `_real_probe_manifest_rows_with_all_silence_sources` already covers
    # all four real values (including both room-tone sources) by calling
    # 05's actual `generate_probes` with engineered synthetic
    # waveforms/logp for deterministic per-branch control, which a random
    # live model cannot offer here.
    val_probe_rows = ip._load_manifest_rows(val_probe_manifest)
    val_silence_sources = {row["silence_source"] for row in val_probe_rows}
    assert val_silence_sources == {"none", "digital_zero"}
    for key in ("baseline_metrics", "gated_metrics"):
        _assert_probe_far_by_silence_source_matches_manifest(val_report[key], val_probe_rows)

    # --- ic.main holdout: refused with a mismatched --beam-width -----------
    with pytest.raises(SystemExit, match="does not match"):
        ic.main(
            [
                "holdout",
                "--manifest", str(manifest_path),
                "--probe-manifest", str(test_probe_manifest),
                "--checkpoint", str(checkpoint_path),
                "--out-dir", str(select_out_dir),
                "--device", "cpu",
                "--beam-width", "1",
                "--selection", str(val_selection_path),
            ]
        )
    assert not (select_out_dir / "test_report.json").exists()

    # --- ic.main holdout: the real held-out report --------------------------
    rc = ic.main(
        [
            "holdout",
            "--manifest", str(manifest_path),
            "--probe-manifest", str(test_probe_manifest),
            "--checkpoint", str(checkpoint_path),
            "--out-dir", str(select_out_dir),
            "--device", "cpu",
            "--selection", str(val_selection_path),
        ]
    )
    assert rc == 0
    test_report_path = select_out_dir / "test_report.json"
    assert test_report_path.exists()
    assert (select_out_dir / "test_report.md").exists()

    test_report = json.loads(test_report_path.read_text())
    assert test_report["beam_width"] == 25

    # per-prefix FAR with overlap flags: "remind" is a documented
    # character-overlap prefix and must show up both in the flagged-prefix
    # list and in the per-prefix FAR breakdown.
    gated = test_report["gated_metrics"]
    assert "remind" in gated["character_overlap_prefixes"]
    assert "remind" in gated["incomplete_prefix_far"]["per_prefix"]

    # Same R2-5 follow-up check, for the held-out report.
    test_probe_rows = ip._load_manifest_rows(test_probe_manifest)
    test_silence_sources = {row["silence_source"] for row in test_probe_rows}
    assert test_silence_sources == {"none", "digital_zero"}
    for key in ("baseline_metrics", "gated_metrics"):
        _assert_probe_far_by_silence_source_matches_manifest(test_report[key], test_probe_rows)

    # --- ic.main holdout: run-once guard actually refuses a second run -----
    with pytest.raises(SystemExit, match="already exists"):
        ic.main(
            [
                "holdout",
                "--manifest", str(manifest_path),
                "--probe-manifest", str(test_probe_manifest),
                "--checkpoint", str(checkpoint_path),
                "--out-dir", str(select_out_dir),
                "--device", "cpu",
                "--selection", str(val_selection_path),
            ]
        )
