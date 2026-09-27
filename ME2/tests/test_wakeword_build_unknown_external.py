import csv

import pytest

from me2_voicegen.wakeword import build_unknown_external as bue
from me2_voicegen.wakeword.fetch_positives import ManifestValidationError


def _write_common_voice(root, vcm_wav_factory, rows: list[dict]):
    audio_dir = root / "audio"
    for row in rows:
        vcm_wav_factory(audio_dir / row["filename"], duration_s=0.3, sample_rate=48000)
    manifest_path = root / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["filename", "accent", "duration", "transcript", "sentence", "client_id", "source_file"]
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _row(i, transcript="", sentence="", source_file=None):
    return {
        "filename": f"cv{i:04d}.wav",
        "accent": "us",
        "duration": "0.3",
        "transcript": transcript,
        "sentence": sentence,
        "client_id": "",
        "source_file": source_file or f"raw{i // 2:04d}.mp3",
    }


@pytest.mark.parametrize(
    "text,expected",
    [
        ("turn on the computer", True),
        ("my computers are broken", True),
        ("that's the computer's fan", True),
        ("please stop the timer", False),
        ("a computerized system", False),
        ("COMPUTER, override", True),
        ("", False),
    ],
)
def test_contains_computer_token(text, expected):
    row = {"transcript": text, "sentence": ""}
    assert bue.contains_computer_token(row) is expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("open sesame now", True),
        ("the sesames are ready", True),
        ("that is sesame's fault", True),
        ("SESAME, open", True),
        ("turn on the computer", False),
        ("sesamean does not contain the bare word", False),
        ("", False),
    ],
)
def test_contains_scrub_token_sesame_scoping(text, expected):
    row = {"transcript": text, "sentence": ""}
    assert bue.contains_scrub_token(row, "sesame") is expected


def test_scrub_words_are_independently_scoped_never_ored():
    """A 'sesame' scrub drops only sesame variants (a computer-only row
    survives it) and the default 'computer' scrub drops only computer
    variants (a sesame-only row survives it) -- the two scrubs must never
    be OR'd together."""
    rows = [
        _row(0, sentence="open sesame please"),
        _row(1, sentence="turn on the computer"),
    ]
    sampled_sesame, dropped_sesame = bue.scrub_and_sample(rows, target_count=1, seed=0, scrub_word="sesame")
    assert dropped_sesame == 1
    assert sampled_sesame[0]["sentence"] == "turn on the computer"

    sampled_default, dropped_default = bue.scrub_and_sample(rows, target_count=1, seed=0)
    assert dropped_default == 1
    assert sampled_default[0]["sentence"] == "open sesame please"


def test_scrub_and_sample_sesame_drops_all_three_forms():
    rows = [
        _row(0, sentence="sesame"),
        _row(1, sentence="sesames"),
        _row(2, sentence="sesame's"),
        _row(3, sentence="plain speech"),
        _row(4, sentence="turn on the computer"),
    ]
    sampled, dropped = bue.scrub_and_sample(rows, target_count=2, seed=0, scrub_word="sesame")
    assert dropped == 3
    sentences = {r["sentence"] for r in sampled}
    assert sentences == {"plain speech", "turn on the computer"}


def test_scrub_and_sample_drops_and_samples_deterministically():
    rows = [_row(i, transcript="turn on the computer" if i % 5 == 0 else "hello there") for i in range(20)]
    sampled_a, dropped_a = bue.scrub_and_sample(rows, target_count=5, seed=1)
    sampled_b, dropped_b = bue.scrub_and_sample(rows, target_count=5, seed=1)

    assert dropped_a == dropped_b == 4  # i in {0,5,10,15}
    assert [r["filename"] for r in sampled_a] == [r["filename"] for r in sampled_b]
    assert len(sampled_a) == 5
    assert all("computer" not in r["transcript"] for r in sampled_a)


def test_scrub_and_sample_raises_when_target_exceeds_pool():
    rows = [_row(i) for i in range(3)]
    with pytest.raises(ManifestValidationError):
        bue.scrub_and_sample(rows, target_count=10, seed=1)


def test_load_common_voice_manifest_missing_raises(tmp_path):
    with pytest.raises(ManifestValidationError):
        bue.load_common_voice_manifest(tmp_path / "does_not_exist")


def test_main_end_to_end_resamples_and_groups(tmp_path, vcm_wav_factory):
    root = tmp_path / "common_voice_negative"
    rows = [_row(i, sentence="hello there", source_file=f"raw{i // 2:04d}.mp3") for i in range(10)]
    _write_common_voice(root, vcm_wav_factory, rows)

    out_root = tmp_path / "common_voice_negative_sample"
    exit_code = bue.main(
        [
            "--common-voice-root",
            str(root),
            "--out-root",
            str(out_root),
            "--target-count",
            "4",
            "--seed",
            "0",
        ]
    )

    assert exit_code == 0
    manifest_path = out_root / "manifest.csv"
    assert manifest_path.is_file()
    with manifest_path.open(newline="", encoding="utf-8") as f:
        out_rows = list(csv.DictReader(f))
    assert len(out_rows) == 4
    for row in out_rows:
        assert row["label"] == "_unknown_"
        assert row["source_dataset"] == "common_voice_negative"
        assert row["sample_rate"] == "16000"
        assert row["resampled"] == "True"
        assert (out_root / row["path"]).is_file()
        assert row["group_id"] == row["group_id"]  # non-empty, derived from source_file
        assert row["group_id"]


def test_main_sesame_scrub_end_to_end_and_summary_reports_real_word(tmp_path, vcm_wav_factory):
    root = tmp_path / "common_voice_negative"
    rows = [
        _row(0, sentence="open sesame now"),
        _row(1, sentence="hello there"),
        _row(2, sentence="the sesames are great"),
        _row(3, sentence="turn on the computer"),
    ]
    _write_common_voice(root, vcm_wav_factory, rows)

    out_root = tmp_path / "common_voice_negative_sample"
    exit_code = bue.main(
        [
            "--common-voice-root",
            str(root),
            "--out-root",
            str(out_root),
            "--target-count",
            "2",
            "--seed",
            "0",
            "--scrub-word",
            "sesame",
        ]
    )

    assert exit_code == 0
    with (out_root / "manifest.csv").open(newline="") as f:
        out_rows = list(csv.DictReader(f))
    # exactly the two non-sesame rows survived and were sampled (the
    # computer-only row is NOT scrubbed by a sesame scrub)
    assert {r["filename"] for r in out_rows} == {"cv0001.wav", "cv0003.wav"}
    for r in out_rows:
        assert (out_root / r["path"]).is_file()
    summary = (out_root / "summary.md").read_text(encoding="utf-8")
    assert "'sesame'-token scrub" in summary
    assert "catches 'sesame'/'sesames'/'sesame's" in summary
    assert "2 dropped by the" in summary  # the two sesame rows, not the computer row


def test_dry_run_writes_nothing(tmp_path, vcm_wav_factory):
    root = tmp_path / "common_voice_negative"
    rows = [_row(i, sentence="hello there") for i in range(5)]
    _write_common_voice(root, vcm_wav_factory, rows)
    out_root = tmp_path / "common_voice_negative_sample"

    exit_code = bue.main(
        [
            "--common-voice-root",
            str(root),
            "--out-root",
            str(out_root),
            "--target-count",
            "2",
            "--seed",
            "0",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    assert not out_root.exists()
