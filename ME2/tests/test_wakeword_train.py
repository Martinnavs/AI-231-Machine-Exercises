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

from me2_voicegen.wakeword.model import LABELS, DSCNNConfig, DSCNN
from me2_voicegen.wakeword.train import (
    LICENSE_NOTE,
    build_arg_parser,
    classify_ww_row_is_filipino,
    main,
    per_class_metrics,
    run_eval,
    wakeword_accent_recall,
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
    "ref_voice",
    "speech_start_s",
    "speech_end_s",
]


def test_arg_parser_defaults():
    args = build_arg_parser().parse_args([])
    assert args.preset == "default"
    assert args.shift is True
    assert args.p_noise == 0.5


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

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    assert ckpt["license"] == LICENSE_NOTE
    assert ckpt["preset"] == "default"
    assert set(ckpt["labels"]) == set(LABELS)

    loss_history = json.loads(loss_history_path.read_text())
    assert loss_history["license"] == LICENSE_NOTE
    assert loss_history["epochs_run"] >= 1

    eval_report = json.loads(eval_report_json.read_text())
    assert eval_report["license"] == LICENSE_NOTE
    assert set(eval_report["per_class"].keys()) == set(LABELS)

    md_text = eval_report_md.read_text()
    assert LICENSE_NOTE in md_text
    assert "accent_recall" in eval_report
    assert set(eval_report["accent_recall"]) == {"filipino", "non_filipino"}
    assert "`_wakeword_` recall by voice accent" in md_text


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
