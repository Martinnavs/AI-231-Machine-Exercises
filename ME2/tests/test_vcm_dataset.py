"""Unit tests for vcm.dataset: VCMDataset + collate_fn, against the
tests/conftest.py `vcm_fake_manifest_factory` shared fixture (owned by
this same task -- see ticket 02)."""

from __future__ import annotations

import torch

from me2_voicegen.common.augment import Augmenter
from me2_voicegen.vcm.dataset import VCMDataset, collate_fn


def _target_command_specs(n: int = 3, split: str = "train") -> list[dict]:
    labels = ["ALARM", "CALL", "STOP"]
    return [
        {
            "bucket": "target_commands",
            "source_dataset": "sanitized_clean",
            "label": labels[i % len(labels)],
            "split": split,
            "duration_s": 0.6,
        }
        for i in range(n)
    ]


def test_dataset_loads_rows_and_resolves_transcripts(vcm_fake_manifest_factory):
    specs = _target_command_specs(3)
    manifest_path = vcm_fake_manifest_factory(specs)

    dataset = VCMDataset(manifest_path, split="train")
    assert len(dataset) == 3

    waveform, target_ids, target_len = dataset[0]
    assert waveform.dim() == 1
    assert target_len == len(target_ids)
    assert target_len > 0


def test_split_filtering(vcm_fake_manifest_factory):
    specs = _target_command_specs(2, split="train") + _target_command_specs(2, split="val")
    manifest_path = vcm_fake_manifest_factory(specs)

    train_ds = VCMDataset(manifest_path, split="train")
    val_ds = VCMDataset(manifest_path, split="val")
    assert len(train_ds) == 2
    assert len(val_ds) == 2


def test_filipino_rows_excluded_from_loss_bearing_but_loadable(vcm_fake_manifest_factory):
    specs = _target_command_specs(2) + [
        {
            "bucket": "babble",
            "source_dataset": "filipino_speech_corpus",
            "label": "unknown",
            "split": "train",
            "duration_s": 0.5,
            "sentence": "hindi na naiintindihan",
        }
    ]
    manifest_path = vcm_fake_manifest_factory(specs)
    dataset = VCMDataset(manifest_path, split="train")

    assert len(dataset) == 3
    assert len(dataset.loss_bearing_indices) == 2
    assert 2 not in dataset.loss_bearing_indices

    # Still loadable by index for eval-only use (rejection probe).
    waveform, target_ids, target_len = dataset[2]
    assert waveform.dim() == 1
    assert target_len == 0
    assert target_ids.numel() == 0


def test_common_voice_negative_transcript_resolution(vcm_fake_manifest_factory):
    specs = [
        {
            "bucket": "babble",
            "source_dataset": "common_voice_negative",
            "label": "unknown",
            "split": "train",
            "duration_s": 0.5,
            "transcript": "turn on the lights please",
        }
    ]
    manifest_path = vcm_fake_manifest_factory(specs)
    dataset = VCMDataset(manifest_path, split="train")
    assert len(dataset.loss_bearing_indices) == 1

    _, target_ids, target_len = dataset[0]
    assert target_len > 0
    from me2_voicegen.vcm import alphabet

    assert alphabet.decode(target_ids.tolist()) == "turn on the lights please"


def test_background_noise_resolves_empty_target_and_is_loss_bearing(vcm_fake_manifest_factory):
    specs = [
        {
            "bucket": "silence",
            "source_dataset": "background_noise",
            "label": "unknown",
            "split": "train",
            "duration_s": 0.4,
        }
    ]
    manifest_path = vcm_fake_manifest_factory(specs)
    dataset = VCMDataset(manifest_path, split="train")

    assert dataset.loss_bearing_indices == [0]
    _, target_ids, target_len = dataset[0]
    assert target_len == 0
    assert target_ids.numel() == 0


# ---------------------------------------------------------------------------
# collate_fn: padded (B, 40, T) batches, input_lengths >= target_lengths.
# ---------------------------------------------------------------------------


def test_collate_fn_produces_padded_batches_with_valid_lengths(vcm_fake_manifest_factory):
    specs = [
        {
            "bucket": "target_commands",
            "source_dataset": "sanitized_clean",
            "label": "ALARM",
            "split": "train",
            "duration_s": 0.4,
        },
        {
            "bucket": "target_commands",
            "source_dataset": "sanitized_clean",
            "label": "LIST_REMINDERS",
            "split": "train",
            "duration_s": 1.2,
        },
        {
            "bucket": "target_commands",
            "source_dataset": "sanitized_clean",
            "label": "STOP",
            "split": "train",
            "duration_s": 0.7,
        },
    ]
    manifest_path = vcm_fake_manifest_factory(specs)
    dataset = VCMDataset(manifest_path, split="train")

    batch = [dataset[i] for i in range(len(dataset))]
    out = collate_fn(batch)

    assert out["features"].dim() == 3
    b, n_mels, max_frames = out["features"].shape
    assert b == 3
    assert n_mels == 40
    assert out["input_lengths"].shape == (3,)
    assert out["target_len"].shape == (3,)
    assert int(out["input_lengths"].max()) == max_frames

    for i in range(3):
        assert out["input_lengths"][i] >= out["target_len"][i]

    # Padding: frames beyond each example's own input_length are exactly 0.
    for i in range(3):
        valid = int(out["input_lengths"][i])
        if valid < max_frames:
            assert torch.equal(
                out["features"][i, :, valid:], torch.zeros(n_mels, max_frames - valid)
            )


def test_noise_pool_loaded_from_disk_at_most_once_across_many_getitem_calls(
    vcm_fake_manifest_factory, monkeypatch
):
    # Regression test for the bug Task 04 diagnosed via real training:
    # _noise_pool() originally reloaded every background_noise wav from
    # disk on every __getitem__ call whenever p_noise > 0, capping real
    # epoch throughput hard (~17 vs ~90+ epochs in a fixed wall-clock
    # budget). The fix caches the pool on first access, same pattern as
    # Augmenter.rir_pool.
    specs = _target_command_specs(3) + [
        {
            "bucket": "silence",
            "source_dataset": "background_noise",
            "label": "unknown",
            "split": "train",
            "duration_s": 0.3,
        }
        for _ in range(5)
    ]
    manifest_path = vcm_fake_manifest_factory(specs)
    augmenter = Augmenter(p_noise=1.0, seed=5)
    dataset = VCMDataset(manifest_path, split="train", augmenter=augmenter)

    call_count = 0
    original_load = VCMDataset._load_wav_path

    def spy_load(self, wav_path):
        nonlocal call_count
        call_count += 1
        return original_load(self, wav_path)

    monkeypatch.setattr(VCMDataset, "_load_wav_path", spy_load)

    n_getitem_calls = 20
    for i in range(n_getitem_calls):
        dataset[i % len(dataset)]

    # 5 background_noise wavs loaded exactly once each, on the first
    # __getitem__ call that needs them -- never again across the other 19.
    assert call_count == 5, (
        f"expected the 5-wav noise pool to be loaded from disk exactly "
        f"once each across {n_getitem_calls} __getitem__ calls, got "
        f"{call_count} disk loads (pool is being reloaded per-item)"
    )


def test_noise_pool_caching_is_actually_fast(vcm_fake_manifest_factory):
    # Real timing sanity check, not just a call-count assertion: confirm
    # many __getitem__ calls with p_noise>0 complete in a time budget
    # consistent with loading the noise pool once, not once per call.
    import time

    specs = _target_command_specs(5) + [
        {
            "bucket": "silence",
            "source_dataset": "background_noise",
            "label": "unknown",
            "split": "train",
            "duration_s": 0.3,
        }
        for _ in range(20)
    ]
    manifest_path = vcm_fake_manifest_factory(specs)
    augmenter = Augmenter(p_noise=1.0, seed=7)
    dataset = VCMDataset(manifest_path, split="train", augmenter=augmenter)

    # Warm the pool once, then time 30 further __getitem__ calls -- with
    # caching this should stay well under a naive per-item reload of 20
    # wavs x 30 calls = 600 disk reads.
    dataset[0]
    start = time.perf_counter()
    for i in range(30):
        dataset[i % len(dataset)]
    elapsed = time.perf_counter() - start

    assert elapsed < 2.0, f"30 cached-pool __getitem__ calls took {elapsed:.3f}s -- looks uncached"


def test_augmenter_wired_into_dataset_noise_pool_does_not_error(vcm_fake_manifest_factory):
    specs = _target_command_specs(1) + [
        {
            "bucket": "silence",
            "source_dataset": "background_noise",
            "label": "unknown",
            "split": "train",
            "duration_s": 0.5,
        }
    ]
    manifest_path = vcm_fake_manifest_factory(specs)
    augmenter = Augmenter(p_rir=1.0, p_noise=1.0, p_specaugment=1.0, seed=5, rir_pool_size=3)
    dataset = VCMDataset(manifest_path, split="train", augmenter=augmenter)

    waveform, _target_ids, _target_len = dataset[0]
    assert torch.isfinite(waveform).all()

    log_mel = dataset.features_for(0)
    assert torch.isfinite(log_mel).all()
    assert log_mel.shape[0] == 40


def test_augmenter_not_applied_outside_train_split(vcm_fake_manifest_factory):
    specs = _target_command_specs(1, split="val")
    manifest_path = vcm_fake_manifest_factory(specs)
    augmenter = Augmenter(p_rir=1.0, p_noise=1.0, p_specaugment=1.0, seed=5, rir_pool_size=3)

    plain_dataset = VCMDataset(manifest_path, split="val", augmenter=None)
    augmented_dataset = VCMDataset(manifest_path, split="val", augmenter=augmenter)

    plain_wave, _, _ = plain_dataset[0]
    augmented_wave, _, _ = augmented_dataset[0]
    # split != "train" -> augmenter must not be applied even though it's wired in.
    assert torch.equal(plain_wave, augmented_wave)


def test_ambient_rows_skip_online_noise_step(vcm_fake_manifest_factory):
    # SPEC Proof #2 (ambient-noise-overlay): a row whose source_dataset ends
    # in `_ambient` is already noised offline, so the train-time augmenter
    # must NOT apply the online noise step to it -- its waveform is invariant
    # under a forced p_noise=1.0 -- while a non-ambient row is still noised.
    # p_rir=0.0 isolates noise as the only waveform-level variable.
    specs = [
        {"bucket": "target_commands", "source_dataset": "optionb", "label": "PLAY_MUSIC",
         "split": "train", "duration_s": 0.6, "transcript": "play music"},
        {"bucket": "target_commands", "source_dataset": "optionb_ambient", "label": "PLAY_MUSIC",
         "split": "train", "duration_s": 0.6, "transcript": "play music"},
        {"bucket": "silence", "source_dataset": "background_noise", "label": "unknown",
         "split": "train", "duration_s": 0.5},
    ]
    manifest_path = vcm_fake_manifest_factory(specs)
    augmenter = Augmenter(p_rir=0.0, p_noise=1.0, seed=42)

    plain_dataset = VCMDataset(manifest_path, split="train", augmenter=None)
    noised_dataset = VCMDataset(manifest_path, split="train", augmenter=augmenter)

    ambient_i = next(i for i, r in enumerate(noised_dataset.rows) if r["source_dataset"].endswith("_ambient"))
    plain_ambient = plain_dataset[ambient_i].waveform
    noised_ambient = noised_dataset[ambient_i].waveform
    # Already noised offline: the forced online noise step must be a no-op.
    assert torch.equal(plain_ambient, noised_ambient)

    non_ambient_i = next(i for i, r in enumerate(noised_dataset.rows) if r["source_dataset"] == "optionb")
    plain_normal = plain_dataset[non_ambient_i].waveform
    noised_normal = noised_dataset[non_ambient_i].waveform
    # A non-ambient row must still receive the online noise.
    assert not torch.equal(plain_normal, noised_normal)


def test_collate_fn_excludes_nothing_itself_caller_must_filter(vcm_fake_manifest_factory):
    # collate_fn operates on whatever VCMExamples it's given; filtering to
    # loss_bearing_indices is the caller's (e.g. a training Sampler's) job.
    specs = _target_command_specs(1) + [
        {
            "bucket": "babble",
            "source_dataset": "filipino_speech_corpus",
            "label": "unknown",
            "split": "train",
            "duration_s": 0.5,
        }
    ]
    manifest_path = vcm_fake_manifest_factory(specs)
    dataset = VCMDataset(manifest_path, split="train")

    loss_bearing_batch = [dataset[i] for i in dataset.loss_bearing_indices]
    out = collate_fn(loss_bearing_batch)
    assert out["features"].shape[0] == 1
