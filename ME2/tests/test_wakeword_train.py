"""Fast CPU-only tests for wakeword.train's core plumbing, mirroring
test_vcm_train.py's split: arg-parsing + isolated loss/metrics wiring on
tiny synthetic tensors (no manifest, no DataLoader), plus one true
end-to-end `main()` run on a tiny fake manifest -- the DS-CNN default
preset is small enough (~30k params) and the window short enough that this
stays fast, unlike vcm's CTC/RIR-heavy pipeline, so it's kept in the fast
suite rather than marked `slow`.
"""

import csv
import json
from pathlib import Path

import torch
from torch import nn

from me2_voicegen.wakeword.dataset import WakewordDataset
from me2_voicegen.wakeword.model import LABELS, DSCNNConfig, DSCNN
from me2_voicegen.wakeword.train import (
    build_accent_unknown_sampler,
    build_arg_parser,
    classify_ww_row_is_filipino,
    license_note,
    main,
    per_class_metrics,
    run_eval,
    wakeword_accent_recall,
)
from torch.utils.data import DataLoader

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
    "ref_voice",
    "speech_start_s",
    "speech_end_s",
]


def test_arg_parser_defaults():
    args = build_arg_parser().parse_args([])
    assert args.preset == "default"
    assert args.shift is True
    assert args.p_noise == 0.5
    assert args.oversample_accent_unknown == 1.0


def test_arg_parser_no_shift_flag():
    args = build_arg_parser().parse_args(["--no-shift"])
    assert args.shift is False


def _tiny_loader(n_batches: int, batch_size: int = 3):
    """A list of `{"features": ..., "labels": ...}` batches shaped exactly
    like `wakeword.dataset.collate_fn`'s output -- proving run_eval/
    per_class_metrics' wiring in isolation from any real Dataset/DataLoader."""
    torch.manual_seed(0)
    batches = []
    for _ in range(n_batches):
        features = torch.randn(batch_size, 40, 20)
        labels = torch.randint(0, len(LABELS), (batch_size,))
        batches.append({"features": features, "labels": labels})
    return batches


def test_run_eval_returns_finite_loss_and_valid_accuracy():
    config = DSCNNConfig(n_blocks=1, channels=8, kernel_sizes=[5], prologue_channels=8)
    model = DSCNN(config)
    criterion = nn.CrossEntropyLoss()
    loss, acc = run_eval(model, _tiny_loader(3), criterion, torch.device("cpu"))
    assert torch.isfinite(torch.tensor(loss))
    assert 0.0 <= acc <= 1.0


def test_per_class_metrics_confusion_matrix_shape_and_bounds():
    config = DSCNNConfig(n_blocks=1, channels=8, kernel_sizes=[5], prologue_channels=8)
    model = DSCNN(config)
    metrics = per_class_metrics(model, _tiny_loader(3), torch.device("cpu"))
    assert len(metrics["confusion_matrix"]) == len(LABELS)
    assert metrics["labels"] == list(LABELS)
    for label in LABELS:
        m = metrics["per_class"][label]
        assert 0.0 <= m["precision"] <= 1.0
        assert 0.0 <= m["recall"] <= 1.0
        assert 0.0 <= m["f1"] <= 1.0
        assert m["support"] >= 0


def test_overfit_tiny_synthetic_batch_to_low_cross_entropy_loss():
    torch.manual_seed(0)
    config = DSCNNConfig(n_blocks=1, channels=16, kernel_sizes=[5], prologue_channels=16)
    model = DSCNN(config)
    features = torch.randn(4, 40, 20)
    labels = torch.tensor([0, 1, 2, 0])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)

    model.train()
    first_loss = None
    last_loss = None
    for _ in range(60):
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        if first_loss is None:
            first_loss = float(loss.item())
        last_loss = float(loss.item())

    assert last_loss < 0.1, f"expected near-zero final loss, got {last_loss}"
    assert last_loss < first_loss / 5


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _tiny_manifest(tmp_path, vcm_wav_factory) -> Path:
    """Every non-silence row carries a precomputed span, so
    `WakewordDataset` never falls back to live per-sample
    `torchaudio.functional.vad` here -- this test is about the training
    loop's plumbing (loss/checkpoint/eval-report wiring), not VAD
    behavior (already covered by test_wakeword_derive_speech_spans.py),
    and skipping live VAD keeps this fast."""
    root = tmp_path / "wakeword"
    rows = []

    def add(subset, filename, label, split, group_id, duration_s=1.5):
        rel = f"{subset}/audio/{filename}"
        vcm_wav_factory(root / rel, duration_s=duration_s, freq_hz=440.0 if label != "_silence_" else 0.0, silence=(label == "_silence_"))
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
                "group_id": group_id,
                "split": split,
                "speech_start_s": "0.200000" if label != "_silence_" else "",
                "speech_end_s": "1.300000" if label != "_silence_" else "",
            }
        )

    for i in range(2):
        add("positives_real", f"wk{i}.wav", "_wakeword_", "train", f"wk{i}")
        add("adversaries", f"adv{i}.wav", "_unknown_", "train", f"adv{i}")
        add("silence_synthetic", f"sil{i}.wav", "_silence_", "train", f"sil{i}")
    add("positives_real", "wkv.wav", "_wakeword_", "val", "wkv")
    add("adversaries", "advv.wav", "_unknown_", "val", "advv")
    add("silence_synthetic", "silv.wav", "_silence_", "val", "silv")

    manifest_path = root / "manifest.csv"
    _write_manifest(manifest_path, rows)
    return manifest_path


def test_main_runs_one_short_training_pass_and_writes_expected_outputs(tmp_path, vcm_wav_factory):
    manifest_path = _tiny_manifest(tmp_path, vcm_wav_factory)
    out_dir = tmp_path / "out"

    main(
        [
            "--manifest",
            str(manifest_path),
            "--out-dir",
            str(out_dir),
            "--max-epochs",
            "1",
            "--max-minutes",
            "1",
            "--batch-size",
            "3",
            "--num-workers",
            "0",
            "--p-noise",
            "0.0",
            "--p-specaugment",
            "0.0",
            "--seed",
            "0",
        ]
    )

    checkpoint_path = out_dir / "checkpoints" / "checkpoint.pt"
    loss_history_path = out_dir / "metadata" / "loss_history.json"
    eval_report_json = out_dir / "metadata" / "eval_report.json"
    eval_report_md = out_dir / "metadata" / "eval_report.md"

    assert checkpoint_path.exists()
    assert loss_history_path.exists()
    assert eval_report_json.exists()
    assert eval_report_md.exists()

    expected_license = license_note(manifest_path)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    assert ckpt["license"] == expected_license
    assert ckpt["preset"] == "default"
    assert set(ckpt["labels"]) == set(LABELS)

    loss_history = json.loads(loss_history_path.read_text())
    assert loss_history["license"] == expected_license
    assert loss_history["epochs_run"] >= 1

    eval_report = json.loads(eval_report_json.read_text())
    assert eval_report["license"] == expected_license
    assert set(eval_report["per_class"].keys()) == set(LABELS)

    md_text = eval_report_md.read_text()
    assert expected_license in md_text
    assert "accent_recall" in eval_report
    assert set(eval_report["accent_recall"]) == {"filipino", "non_filipino"}
    assert "`_wakeword_` recall by voice accent" in md_text


def test_license_note_names_the_actual_manifest_root_not_a_hardcoded_one():
    """Regression for the "computer"-hardcoded prose bug: two different
    phrase-instances' manifests must each get their own root named in the
    note, not a fixed 'out/conversions/v2/wakeword/' string."""
    computer_note = license_note(Path("out/conversions/v2/wakeword/manifest.csv"))
    sesame_note = license_note(Path("out/conversions/v2/wakeword-sesame/manifest.csv"))
    assert "out/conversions/v2/wakeword" in computer_note
    assert "wakeword-sesame" not in computer_note
    assert "out/conversions/v2/wakeword-sesame" in sesame_note
    # both still carry the actual license terms, unchanged
    assert "CC-BY-NC-SA-4.0" in computer_note
    assert "CC-BY-NC-SA-4.0" in sesame_note


# ---------------------------------------------------------------------------
# feature accent-balance-fil50 (.scratch/accent-balance-fil50/tickets/
# 00-RECAP.md T6): classify_ww_row_is_filipino / wakeword_accent_recall /
# --eval-only.
# ---------------------------------------------------------------------------


def test_classify_ww_row_is_filipino_by_ref_voice():
    assert classify_ww_row_is_filipino({"ref_voice": "tagalog3"}) is True
    assert classify_ww_row_is_filipino({"ref_voice": "ilonggo1"}) is True
    assert classify_ww_row_is_filipino({"ref_voice": "picovoice"}) is False
    assert classify_ww_row_is_filipino({"ref_voice": ""}) is False
    assert classify_ww_row_is_filipino({}) is False


def test_classify_ww_row_is_filipino_by_source_dataset():
    assert classify_ww_row_is_filipino({"source_dataset": "fil50_persona", "ref_voice": ""}) is True
    assert classify_ww_row_is_filipino({"source_dataset": "fil50_persona_noisy", "ref_voice": ""}) is True
    assert classify_ww_row_is_filipino({"source_dataset": "picovoice", "ref_voice": ""}) is False
    # iteration-3 accent-mined negatives are Filipino-accented rows too
    assert classify_ww_row_is_filipino({"source_dataset": "accent_mined_unknown", "ref_voice": ""}) is True
    assert classify_ww_row_is_filipino({"source_dataset": "accent_mined_conversion", "ref_voice": ""}) is True


def _accent_manifest(tmp_path, vcm_wav_factory) -> Path:
    root = tmp_path / "wakeword_accent"
    rows = []

    def add(filename, label, group_id, ref_voice, source_dataset="positives_real"):
        rel = f"positives_real/audio/{filename}"
        vcm_wav_factory(root / rel, duration_s=1.5, freq_hz=440.0 if label != "_silence_" else 0.0, silence=(label == "_silence_"))
        rows.append({
            "filename": filename, "path": rel, "label": label, "duration": "1.500000", "sample_rate": "16000",
            "resampled": "False", "source_dataset": source_dataset, "source_relpath": filename, "group_id": group_id,
            "split": "val", "ref_voice": ref_voice,
            "speech_start_s": "0.200000" if label != "_silence_" else "", "speech_end_s": "1.300000" if label != "_silence_" else "",
        })

    add("wk_fil_0.wav", "_wakeword_", "g0", "tagalog3")
    add("wk_fil_1.wav", "_wakeword_", "g1", "ilonggo1")
    add("wk_nonfil_0.wav", "_wakeword_", "g2", "")
    add("adv_0.wav", "_unknown_", "g3", "")  # non-_wakeword_ row -- must not count toward either bucket

    manifest_path = root / "manifest.csv"
    _write_manifest(manifest_path, rows)
    return manifest_path


def test_wakeword_accent_recall_splits_by_ref_voice(tmp_path, vcm_wav_factory):
    from me2_voicegen.wakeword.dataset import WakewordDataset

    manifest_path = _accent_manifest(tmp_path, vcm_wav_factory)
    dataset = WakewordDataset(manifest_path, split="val", augmenter=None, shift=False)
    config = DSCNNConfig(n_blocks=1, channels=8, kernel_sizes=[5], prologue_channels=8)
    model = DSCNN(config)

    result = wakeword_accent_recall(model, dataset, torch.device("cpu"))

    assert set(result) == {"filipino", "non_filipino"}
    assert result["filipino"]["total"] == 2  # the two tagalog/ilonggo _wakeword_ rows
    assert result["non_filipino"]["total"] == 1  # the one ref_voice="" _wakeword_ row
    for bucket in result.values():
        assert 0 <= bucket["correct"] <= bucket["total"]
        assert bucket["recall"] is None or 0.0 <= bucket["recall"] <= 1.0


def test_eval_only_scores_an_existing_checkpoint_without_training(tmp_path, vcm_wav_factory):
    manifest_path = _tiny_manifest(tmp_path, vcm_wav_factory)
    train_out_dir = tmp_path / "out_trained"
    main([
        "--manifest", str(manifest_path), "--out-dir", str(train_out_dir), "--max-epochs", "1",
        "--max-minutes", "1", "--batch-size", "3", "--num-workers", "0", "--p-noise", "0.0",
        "--p-specaugment", "0.0", "--seed", "0",
    ])
    checkpoint_path = train_out_dir / "checkpoints" / "checkpoint.pt"
    assert checkpoint_path.is_file()

    eval_out_dir = tmp_path / "out_eval_only"
    main([
        "--manifest", str(manifest_path), "--out-dir", str(eval_out_dir), "--eval-only",
        "--checkpoint", str(checkpoint_path), "--num-workers", "0",
    ])

    eval_report_json = eval_out_dir / "metadata" / "eval_report.json"
    assert eval_report_json.is_file()
    assert not (eval_out_dir / "checkpoints" / "checkpoint.pt").exists()  # eval-only never trains/writes a checkpoint

    eval_report = json.loads(eval_report_json.read_text())
    assert eval_report["checkpoint_path"] == str(checkpoint_path)
    assert "accent_recall" in eval_report


def test_eval_only_without_checkpoint_raises():
    import pytest

    with pytest.raises(SystemExit):
        main(["--eval-only"])


# ---------------------------------------------------------------------------
# feature iteration-3 (docs/20260925_suggestions.md "Wakeword Training Shift"):
# --oversample-accent-unknown.
# ---------------------------------------------------------------------------


def test_oversample_default_1_0_keeps_today_exact_shuffle(tmp_path, vcm_wav_factory):
    """factor=1.0 must leave the existing sampling behavior byte-identical:
    same sampler type, same per-epoch index sequence for the same seed as
    today's plain `shuffle=True` DataLoader."""
    manifest_path = _tiny_manifest(tmp_path, vcm_wav_factory)
    dataset = WakewordDataset(manifest_path, split="train", augmenter=None, shift=False)

    torch.manual_seed(7)
    loader_today = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=0)
    # two full epochs, back to back (one RNG timeline)
    seq_today = [list(loader_today.sampler) for _ in range(2)]

    torch.manual_seed(7)
    sampler = build_accent_unknown_sampler(dataset, 1.0)
    assert sampler is None
    loader_new = DataLoader(
        dataset, batch_size=3, shuffle=(sampler is None), sampler=sampler, num_workers=0
    )
    seq_new = [list(loader_new.sampler) for _ in range(2)]

    assert seq_new == seq_today


def _write_train_split_manifest(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_oversample_factor_upsamples_accent_mined_rows(tmp_path):
    """Statistical check (fixed seed): with factor=2.0 each accent-mined row
    is drawn ~2x as often per epoch as a non-mined row. (Manifest rows only --
    the sampler never loads audio, so no wav files are needed here.)"""
    n_other, n_mined = 80, 20
    rows = []
    for i in range(n_other):
        rows.append({
            "filename": f"oth{i}.wav", "path": f"adversaries/audio/oth{i}.wav", "label": "_unknown_",
            "duration": "1.000000", "sample_rate": "16000", "resampled": "False",
            "source_dataset": "common_voice_negative", "source_relpath": f"oth{i}.wav",
            "group_id": f"oth{i}", "split": "train", "ref_voice": "",
            "speech_start_s": "0.100000", "speech_end_s": "0.900000",
        })
    for i in range(n_mined):
        rows.append({
            "filename": f"mined{i}.wav", "path": f"accent_mined_unknown/audio/mined{i}.wav",
            "label": "_unknown_", "duration": "1.000000", "sample_rate": "16000", "resampled": "False",
            "source_dataset": "accent_mined_unknown" if i % 2 else "accent_mined_conversion",
            "source_relpath": f"mined{i}.wav", "group_id": f"wwneg_fsc_{i:02d}", "split": "train",
            "ref_voice": "ref_tagalog1" if i % 2 else "", "speech_start_s": "", "speech_end_s": "",
        })
    manifest_path = _write_train_split_manifest(tmp_path / "m.csv", rows)
    dataset = WakewordDataset(manifest_path, split="train", augmenter=None, shift=False)
    assert len(dataset) == 100

    sampler = build_accent_unknown_sampler(dataset, 2.0)
    assert sampler is not None
    mined_idxs = set(range(n_other, n_other + n_mined))  # mined rows are last in the manifest

    torch.manual_seed(0)
    mined_draws = other_draws = 0
    for _ in range(200):
        epoch = list(sampler)
        assert len(epoch) == 100  # epoch length unchanged -> OneCycleLR steps untouched
        mined_draws += sum(1 for i in epoch if i in mined_idxs)
        other_draws += sum(1 for i in epoch if i not in mined_idxs)

    ratio = (mined_draws / n_mined) / (other_draws / n_other)
    # expected ratio is exactly 2.0; ~1.4% CV at this draw count, so a tight band
    assert 1.85 < ratio < 2.15, f"mined/non-mined per-row draw ratio {ratio:.3f} not ~2.0"


def test_oversample_factor_below_one_rejected(tmp_path, vcm_wav_factory):
    manifest_path = _tiny_manifest(tmp_path, vcm_wav_factory)
    dataset = WakewordDataset(manifest_path, split="train", augmenter=None, shift=False)
    import pytest

    with pytest.raises(ValueError, match=">= 1.0"):
        build_accent_unknown_sampler(dataset, 0.5)


def _oversample_manifest(tmp_path, vcm_wav_factory) -> Path:
    """6 train rows (3 of them accent-mined) + 4 val rows (1 mined), all with
    precomputed spans so no live VAD fires."""
    root = tmp_path / "ww_oversample"
    rows = []

    def add(filename, label, split, group_id, source_dataset, ref_voice=""):
        rel = f"accent_mined_unknown/audio/{filename}" if source_dataset.startswith("accent_mined") else f"adversaries/audio/{filename}"
        vcm_wav_factory(root / rel, duration_s=1.5)
        rows.append({
            "filename": filename, "path": rel, "label": label, "duration": "1.500000",
            "sample_rate": "16000", "resampled": "False", "source_dataset": source_dataset,
            "source_relpath": filename, "group_id": group_id, "split": split, "ref_voice": ref_voice,
            "speech_start_s": "0.200000", "speech_end_s": "1.300000",
        })

    add("t0.wav", "_unknown_", "train", "oth0", "common_voice_negative")
    add("t1.wav", "_unknown_", "train", "oth1", "common_voice_negative")
    add("t2.wav", "_unknown_", "train", "oth2", "common_voice_negative")
    add("m0.wav", "_unknown_", "train", "wwneg_fsc_01", "accent_mined_unknown")
    add("m1.wav", "_unknown_", "train", "wwneg_fsc_02", "accent_mined_unknown")
    add("m2.wav", "_unknown_", "train", "wwneg_ref_tagalog1", "accent_mined_conversion", ref_voice="ref_tagalog1")
    add("v0.wav", "_unknown_", "val", "othv0", "common_voice_negative")
    add("v1.wav", "_unknown_", "val", "othv1", "common_voice_negative")
    add("v2.wav", "_silence_", "val", "othv2", "common_voice_negative")
    add("vm.wav", "_unknown_", "val", "wwneg_fsc_03", "accent_mined_unknown")

    manifest_path = root / "manifest.csv"
    _write_train_split_manifest(manifest_path, rows)
    return manifest_path


def test_main_oversample_records_provenance_and_leaves_val_unaffected(tmp_path, vcm_wav_factory):
    manifest_path = _oversample_manifest(tmp_path, vcm_wav_factory)
    out_dir = tmp_path / "out"

    main(
        [
            "--manifest", str(manifest_path), "--out-dir", str(out_dir), "--max-epochs", "1",
            "--max-minutes", "1", "--batch-size", "3", "--num-workers", "0", "--p-noise", "0.0",
            "--p-specaugment", "0.0", "--seed", "0", "--oversample-accent-unknown", "2.0",
        ]
    )

    # provenance recorded in both artifacts
    loss_history = json.loads((out_dir / "metadata" / "loss_history.json").read_text())
    assert loss_history["oversample_accent_unknown"] == 2.0
    ckpt = torch.load(out_dir / "checkpoints" / "checkpoint.pt", map_location="cpu")
    assert ckpt["oversample_accent_unknown"] == 2.0

    # val split is evaluated over exactly its 4 rows (no up-sampling there)
    eval_report = json.loads((out_dir / "metadata" / "eval_report.json").read_text())
    total_val_support = sum(m["support"] for m in eval_report["per_class"].values())
    assert total_val_support == 4
