import csv
from pathlib import Path

import pytest
import torch

from me2_voicegen.common.augment import Augmenter
from me2_voicegen.common.features import LogMelFeatureExtractor
from me2_voicegen.wakeword.dataset import (
    PRECOMPUTED_SUBSETS,
    RARE_FALLBACK_SUBSETS,
    WakewordDataset,
    collate_fn,
)

FIELDS = [
    "filename",
    "path",
    "label",
    "duration",
    "sample_rate",
    "resampled",
    "source_dataset",
    "source_relpath",
    "group_id",
    "split",
    "speech_start_s",
    "speech_end_s",
]


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def wakeword_manifest(tmp_path, vcm_wav_factory):
    """Final-assembled-manifest shape: `path` is prefixed with the owning
    subset name (`<subset>/audio/<filename>`), matching
    `build_dataset.py`'s real output -- not the single-subset shape
    `wakeword_fake_manifest_factory` builds, which this module doesn't
    consume directly (that fixture is documented as standing in for a
    single subset's own manifest, not the ticket-04 merge)."""
    root = tmp_path / "wakeword"
    rows: list[dict] = []

    def add(subset, filename, label, split, duration_s=1.0, span=None, silence=False, group_id=None):
        rel = f"{subset}/audio/{filename}"
        vcm_wav_factory(root / rel, duration_s=duration_s, silence=silence)
        rows.append(
            {
                "filename": filename,
                "path": rel,
                "label": label,
                "duration": f"{duration_s:.6f}",
                "sample_rate": "16000",
                "resampled": "False",
                "source_dataset": subset,
                "source_relpath": filename,
                "group_id": group_id or Path(filename).stem,
                "split": split,
                "speech_start_s": f"{span[0]:.6f}" if span else "",
                "speech_end_s": f"{span[1]:.6f}" if span else "",
            }
        )

    add("positives_real", "wk1.wav", "_wakeword_", "train", span=(0.2, 0.8))
    add("positives_real", "wk2.wav", "_wakeword_", "val", span=(0.1, 0.9))
    add("adversaries", "adv1.wav", "_unknown_", "train")  # no precomputed span, by design
    add("silence_synthetic", "sil1.wav", "_silence_", "train", silence=True)

    manifest_path = root / "manifest.csv"
    _write_manifest(manifest_path, rows)
    return manifest_path


def test_split_filtering_and_label_mapping(wakeword_manifest):
    train_ds = WakewordDataset(wakeword_manifest, split="train")
    assert len(train_ds) == 3  # wk1, adv1, sil1
    labels = {ex_row["label"] for ex_row in train_ds.rows}
    assert labels == {"_wakeword_", "_unknown_", "_silence_"}

    val_ds = WakewordDataset(wakeword_manifest, split="val")
    assert len(val_ds) == 1
    assert val_ds.rows[0]["filename"] == "wk2.wav"


def test_raises_on_manifest_with_only_unassigned_splits(tmp_path, vcm_wav_factory):
    root = tmp_path / "wakeword"
    vcm_wav_factory(root / "positives_real/audio/wk1.wav", duration_s=1.0)
    _write_manifest(
        root / "manifest.csv",
        [
            {
                "filename": "wk1.wav",
                "path": "positives_real/audio/wk1.wav",
                "label": "_wakeword_",
                "duration": "1.000000",
                "sample_rate": "16000",
                "resampled": "False",
                "source_dataset": "positives_real",
                "source_relpath": "wk1.wav",
                "group_id": "wk1",
                "split": "",  # unassigned -- pre-ticket-04 manifest
                "speech_start_s": "",
                "speech_end_s": "",
            }
        ],
    )
    with pytest.raises(ValueError, match="unassigned"):
        WakewordDataset(root / "manifest.csv", split="train")


def test_getitem_returns_fixed_window_length(wakeword_manifest):
    ds = WakewordDataset(wakeword_manifest, split="train", window_seconds=0.5)
    for i in range(len(ds)):
        example = ds[i]
        assert example.waveform.shape[-1] == int(0.5 * 16000)
        assert isinstance(example.label, int)


def test_precomputed_span_used_without_live_vad(wakeword_manifest, monkeypatch):
    calls = []

    def fake_detect(waveform, sample_rate):
        calls.append(1)
        return None

    monkeypatch.setattr("me2_voicegen.wakeword.dataset.detect_speech_span", fake_detect)

    ds = WakewordDataset(wakeword_manifest, split="train", window_seconds=0.5)
    wk1_index = next(i for i, r in enumerate(ds.rows) if r["filename"] == "wk1.wav")
    ds[wk1_index]
    assert calls == []  # precomputed span present -- detect_speech_span never called


def test_live_vad_fallback_invoked_for_rows_without_precomputed_span(wakeword_manifest, monkeypatch):
    calls = []

    def fake_detect(waveform, sample_rate):
        calls.append(1)
        return None

    monkeypatch.setattr("me2_voicegen.wakeword.dataset.detect_speech_span", fake_detect)

    ds = WakewordDataset(wakeword_manifest, split="train", window_seconds=0.5)
    adv_index = next(i for i, r in enumerate(ds.rows) if r["filename"] == "adv1.wav")
    ds[adv_index]
    assert calls == [1]
    assert ds.fallback_counts.get("adversaries") == 1


def test_silence_label_never_calls_detect_speech_span(wakeword_manifest, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "me2_voicegen.wakeword.dataset.detect_speech_span",
        lambda waveform, sample_rate: calls.append(1),
    )

    ds = WakewordDataset(wakeword_manifest, split="train", window_seconds=0.5)
    sil_index = next(i for i, r in enumerate(ds.rows) if r["filename"] == "sil1.wav")
    ds[sil_index]
    assert calls == []


def test_fallback_summary_logs_only_non_precomputed_subsets(wakeword_manifest, monkeypatch, capsys):
    monkeypatch.setattr(
        "me2_voicegen.wakeword.dataset.detect_speech_span", lambda waveform, sample_rate: None
    )
    ds = WakewordDataset(wakeword_manifest, split="train", window_seconds=0.5)
    for i in range(len(ds)):
        ds[i]

    ds.log_fallback_summary()
    out = capsys.readouterr().out
    assert "adversaries" in out
    assert "positives_real" not in out  # in PRECOMPUTED_SUBSETS -- no aggregate line for it
    assert "adversaries" not in PRECOMPUTED_SUBSETS


def test_common_voice_negative_sample_is_precomputed_but_not_rare_fallback():
    """Regression guard for a real-pipeline discovery: common_voice_negative_sample
    gets a derive_speech_spans.py precompute attempt (its ~55% natural
    VAD-empty rate is a property of the data, not a bug), but it must stay
    out of RARE_FALLBACK_SUBSETS -- per-row warnings for a subset that
    fails ~half the time would be log noise, same reasoning as adversaries."""
    assert "common_voice_negative_sample" in PRECOMPUTED_SUBSETS
    assert "common_voice_negative_sample" not in RARE_FALLBACK_SUBSETS


def test_collate_fn_produces_batch_shaped_features_and_labels(wakeword_manifest):
    ds = WakewordDataset(wakeword_manifest, split="train", window_seconds=0.5)
    extractor = LogMelFeatureExtractor()
    batch = [ds[i] for i in range(len(ds))]
    out = collate_fn(batch, extractor)
    assert out["features"].shape[0] == len(batch)
    assert out["features"].shape[1] == 40
    assert out["labels"].shape == (len(batch),)
    assert out["labels"].dtype == torch.long


def test_ambient_rows_skip_online_noise_step(tmp_path, vcm_wav_factory):
    # SPEC Proof #2 (ambient-noise-overlay): a row whose source_dataset ends
    # in `_ambient` is already noised offline, so the train-time augmenter
    # must NOT apply the online noise step to it -- its waveform is invariant
    # under a forced p_noise=1.0 -- while a non-ambient row is still noised.
    # p_rir=0.0 isolates noise as the only waveform-level variable;
    # shift=False + the same window makes the augment block the only
    # difference between the plain and noised datasets.
    root = tmp_path / "wakeword"
    rows: list[dict] = []

    def add(subset, filename, label, split, duration_s=1.0, span=None):
        rel = f"{subset}/audio/{filename}"
        vcm_wav_factory(root / rel, duration_s=duration_s)
        rows.append(
            {
                "filename": filename,
                "path": rel,
                "label": label,
                "duration": f"{duration_s:.6f}",
                "sample_rate": "16000",
                "resampled": "False",
                "source_dataset": subset,
                "source_relpath": filename,
                "group_id": Path(filename).stem,
                "split": split,
                "speech_start_s": f"{span[0]:.6f}" if span else "",
                "speech_end_s": f"{span[1]:.6f}" if span else "",
            }
        )

    add("positives_real", "wk1.wav", "_wakeword_", "train", span=(0.2, 0.8))
    add("positives_real_ambient", "wk2.wav", "_wakeword_", "train", span=(0.2, 0.8))
    manifest_path = root / "manifest.csv"
    _write_manifest(manifest_path, rows)

    noise_dir = tmp_path / "noise"
    noise_dir.mkdir()
    vcm_wav_factory(noise_dir / "noise1.wav", duration_s=1.0)

    augmenter = Augmenter(p_rir=0.0, p_noise=1.0, seed=42)
    plain_ds = WakewordDataset(manifest_path, split="train", window_seconds=0.5, noise_root=noise_dir)
    noised_ds = WakewordDataset(
        manifest_path,
        split="train",
        window_seconds=0.5,
        augmenter=augmenter,
        noise_root=noise_dir,
        shift=False,
        generator=torch.Generator().manual_seed(0),
    )

    ambient_i = next(i for i, r in enumerate(noised_ds.rows) if r["source_dataset"].endswith("_ambient"))
    # Already noised offline: the forced online noise step must be a no-op.
    assert torch.equal(plain_ds[ambient_i].waveform, noised_ds[ambient_i].waveform)

    normal_i = next(i for i, r in enumerate(noised_ds.rows) if r["source_dataset"] == "positives_real")
    # A non-ambient row must still receive the online noise.
    assert not torch.equal(plain_ds[normal_i].waveform, noised_ds[normal_i].waveform)


def test_noise_pool_finds_wavs_nested_under_an_audio_subdirectory(tmp_path, vcm_wav_factory):
    # Real-world bug (found 2026-09-28, ambient-reverb-cooccurrence T4): every
    # wakeword training invocation's --noise-root default/Makefile value is
    # out/conversions/v2/background_noise, whose wavs live under its own
    # audio/ subdirectory (the same corpus-root/audio/ layout
    # mix_background_noise.py already resolves correctly via
    # noise_root/"audio"/filename) -- but `_noise_pool` globbed only
    # noise_root's top level, so the pool was silently empty for every
    # production and sesame run to date. This corpus layout, not a flat
    # directory, is the real one always passed in practice.
    noise_root = tmp_path / "background_noise"
    (noise_root / "audio").mkdir(parents=True)
    vcm_wav_factory(noise_root / "audio" / "noise1.wav", duration_s=1.0)
    vcm_wav_factory(noise_root / "audio" / "noise2.wav", duration_s=1.0)

    manifest_path = tmp_path / "wakeword" / "manifest.csv"
    _write_manifest(manifest_path, [])  # empty manifest: only noise_root matters here
    ds = WakewordDataset(manifest_path, split="train", window_seconds=0.5, noise_root=noise_root)
    pool = ds._noise_pool()
    assert len(pool) == 2


def test_noise_pool_still_finds_flat_top_level_wavs(tmp_path, vcm_wav_factory):
    # Backward-compat: a noise_root whose wavs sit directly at its top level
    # (no nested subdirectory) must keep working after switching to a
    # recursive glob.
    noise_root = tmp_path / "flat_noise"
    noise_root.mkdir()
    vcm_wav_factory(noise_root / "noise1.wav", duration_s=1.0)

    manifest_path = tmp_path / "wakeword2" / "manifest.csv"
    _write_manifest(manifest_path, [])
    ds = WakewordDataset(manifest_path, split="train", window_seconds=0.5, noise_root=noise_root)
    pool = ds._noise_pool()
    assert len(pool) == 1
