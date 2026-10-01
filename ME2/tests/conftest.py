import csv
import math
import sys
from pathlib import Path

import pytest
import torch
import torchaudio

import me2_voicegen.generation.cosyvoice_env as cosyvoice_env
import me2_voicegen.vcm.text as vcm_text
from me2_voicegen.wakeword.fetch_positives import MANIFEST_FIELDS as WAKEWORD_MANIFEST_FIELDS


@pytest.fixture
def fake_project(tmp_path, monkeypatch):
    """Patch cosyvoice_env's module-level paths onto a throwaway tmp_path tree.

    Callers create whichever of vendor/CosyVoice, vendor/CosyVoice/third_party/
    Matcha-TTS they need for a given test; nothing is created by default.
    """
    project_root = tmp_path
    vendor_root = project_root / "vendor"
    cosyvoice_dir = vendor_root / "CosyVoice"
    matcha_dir = cosyvoice_dir / "third_party" / "Matcha-TTS"
    models_dir = project_root / "models"
    out_dir = project_root / "out"

    monkeypatch.setattr(cosyvoice_env, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(cosyvoice_env, "VENDOR_ROOT", vendor_root)
    monkeypatch.setattr(cosyvoice_env, "COSYVOICE_DIR", cosyvoice_dir)
    monkeypatch.setattr(cosyvoice_env, "MATCHA_TTS_DIR", matcha_dir)
    monkeypatch.setattr(cosyvoice_env, "MODELS_DIR", models_dir)
    monkeypatch.setattr(cosyvoice_env, "OUT_DIR", out_dir)

    before = list(sys.path)
    yield cosyvoice_env, project_root, cosyvoice_dir, matcha_dir
    sys.path[:] = before


# ---------------------------------------------------------------------------
# Shared VCM fixtures (owned by feature `vcm-toy` task 02).
#
# Tasks 03/04/05/07 and ticket 08's audit all depend on these three
# fixtures existing here with this shape -- don't invent competing
# per-test-file versions. See ME2/docs/VCM-CONTRACT.md for the manifest
# schema and transcript-resolution rules these fixtures stand in for.
# ---------------------------------------------------------------------------


@pytest.fixture
def vcm_wav_factory(tmp_path):
    """Factory: write a tiny synthetic 16kHz mono PCM wav file, return its
    Path. `silence=True` writes all-zero samples (for `background_noise`/
    silence-bucket rows); otherwise a low-amplitude sine tone."""

    def make(
        rel_path: str | Path,
        duration_s: float = 0.5,
        sample_rate: int = 16000,
        freq_hz: float = 440.0,
        silence: bool = False,
    ) -> Path:
        path = tmp_path / rel_path if not Path(rel_path).is_absolute() else Path(rel_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        n_samples = max(1, int(duration_s * sample_rate))
        if silence:
            waveform = torch.zeros(1, n_samples)
        else:
            t = torch.arange(n_samples, dtype=torch.float32) / sample_rate
            waveform = (0.1 * torch.sin(2 * math.pi * freq_hz * t)).unsqueeze(0)
        torchaudio.save(str(path), waveform, sample_rate)
        return path

    return make


@pytest.fixture
def vcm_tiny_wav(vcm_wav_factory):
    """A single ready-made tiny 16kHz mono wav (0.5s sine tone), for tests
    that just need "a" wav without building a full fake manifest."""
    return vcm_wav_factory("tiny.wav")


@pytest.fixture
def vcm_fake_manifest_factory(tmp_path, monkeypatch, vcm_wav_factory):
    """Factory: build a small synthetic `test_set/manifest.csv` (real
    column schema, per docs/VCM-CONTRACT.md section 4) plus matching audio
    files and, when needed, fake source-dataset manifests so
    `vcm.text.resolve_transcript` resolves against them instead of the
    real `out/conversions/v2/` tree.

    `build(specs)` takes a list of dicts, one per manifest row, each with:
      - `bucket` (str, required)
      - `source_dataset` (str, required) -- one of the 5 real values.
      - `label` (str, default "unknown") -- required to be a real
        `INTENT_PHRASES` key when `source_dataset == "sanitized_clean"`.
      - `split` (str, default "train")
      - `duration_s` (float, default 0.5)
      - `transcript` (str, default "") -- used as the fake source
        manifest's `transcript` column for `common_voice_negative`/
        `youtube_institutional` rows; written directly into the built
        manifest's own `transcript` column (no source manifest, no join)
        for `optionb` rows -- see docs/VCM-CONTRACT.md section 4; ignored
        for other sources.
      - `sentence` (str, default "") -- written to the fake
        `filipino_speech_corpus` source manifest's `sentence` column, for
        completeness only (resolve_transcript ignores it unconditionally).

    Returns the built `manifest.csv` Path. `VCMDataset(manifest_path=...)`
    can load it directly (its default `audio_root` is the manifest's own
    parent directory, matching the real `test_set/` layout).
    """

    def build(specs: list[dict]) -> Path:
        root = tmp_path / "fake_test_set"
        root.mkdir(exist_ok=True)
        fake_conversions_dir = tmp_path / "fake_conversions_v2"
        fake_conversions_dir.mkdir(exist_ok=True)

        source_manifest_rows: dict[str, list[dict]] = {}
        manifest_rows: list[dict] = []

        for i, spec in enumerate(specs):
            source_dataset = spec["source_dataset"]
            bucket = spec["bucket"]
            duration_s = spec.get("duration_s", 0.5)
            filename = f"{i:05d}.wav"
            rel_path = f"audio/{bucket}/{filename}"
            # `background_noise` rows are real (non-silent) ESC-50 ambient
            # noise clips in the actual corpus -- the "silence" here is a
            # bucket/label name, not literal digital silence. Only honor an
            # explicit `spec["silence"]` override (e.g. a caller
            # deliberately testing a true-zero-amplitude edge case).
            vcm_wav_factory(
                root / rel_path, duration_s=duration_s, silence=spec.get("silence", False)
            )

            source_relpath = f"audio/{filename}"
            if source_dataset in ("common_voice_negative", "youtube_institutional"):
                source_manifest_rows.setdefault(source_dataset, []).append(
                    {"filename": filename, "transcript": spec.get("transcript", "")}
                )
            elif source_dataset == "filipino_speech_corpus":
                source_manifest_rows.setdefault(source_dataset, []).append(
                    {"filename": filename, "sentence": spec.get("sentence", "")}
                )

            manifest_rows.append(
                {
                    "filename": filename,
                    "path": rel_path,
                    "bucket": bucket,
                    "label": spec.get("label", "unknown"),
                    "duration": f"{duration_s:.6f}",
                    "sample_rate": "16000",
                    "resampled": "False",
                    "source_dataset": source_dataset,
                    "source_relpath": source_relpath,
                    "group_id": str(i),
                    "split": spec.get("split", "train"),
                    # Only `optionb` reads this directly off the built
                    # manifest row (no source-manifest join, D7); harmless
                    # unused column for the other four source_datasets.
                    "transcript": spec.get("transcript", "") if source_dataset == "optionb" else "",
                }
            )

        manifest_path = root / "manifest.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)

        for source_dataset, rows in source_manifest_rows.items():
            source_dir = fake_conversions_dir / source_dataset
            source_dir.mkdir(parents=True, exist_ok=True)
            with (source_dir / "manifest.csv").open(
                "w", newline="", encoding="utf-8"
            ) as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

        monkeypatch.setattr(vcm_text, "CONVERSIONS_V2_DIR", fake_conversions_dir)
        vcm_text._load_source_manifest.cache_clear()

        return manifest_path

    yield build
    vcm_text._load_source_manifest.cache_clear()


class _StubCTCModel(torch.nn.Module):
    """Trivial stand-in for `vcm.model`'s trained CTC model.

    Given `(B, n_mels, T)` log-mel features, returns `(B, T, alphabet_size)`
    logits. With `forced_ids=None` every frame is all-zero logits (uniform
    posterior after softmax). With `forced_ids=[id0, id1, ...]`, frame `t`
    gets a large peak at `forced_ids[t % len(forced_ids)]` and near-zero
    elsewhere -- i.e. controllable, near-one-hot-per-frame, so a caller can
    script an exact target decode deterministically without a real
    checkpoint. `forced_ids` is repeated to fill however many frames the
    input actually has, so callers don't need to know T in advance.
    """

    def __init__(
        self,
        alphabet_size: int = 29,
        forced_ids: list[int] | None = None,
        peak: float = 12.0,
    ) -> None:
        super().__init__()
        self.alphabet_size = alphabet_size
        self.forced_ids = forced_ids
        self.peak = peak

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, _, frames = features.shape
        logits = torch.zeros(batch, frames, self.alphabet_size)
        if self.forced_ids:
            for t in range(frames):
                token_id = self.forced_ids[t % len(self.forced_ids)]
                logits[:, t, token_id] = self.peak
        return logits


@pytest.fixture
def vcm_stub_model_factory():
    """Factory: `make(forced_ids=None, alphabet_size=29, peak=12.0) ->
    _StubCTCModel`. CPU-only, no real checkpoint required; see
    `_StubCTCModel` docstring for behavior."""

    def make(
        forced_ids: list[int] | None = None,
        alphabet_size: int = 29,
        peak: float = 12.0,
    ) -> _StubCTCModel:
        return _StubCTCModel(alphabet_size=alphabet_size, forced_ids=forced_ids, peak=peak)

    return make


# ---------------------------------------------------------------------------
# Shared wakeword fixture (owned by feature `wakeword-computer-dataset` ticket
# 01). Tickets 03/04 both depend on this fixture existing here with this
# shape -- don't invent competing per-test-file versions. See
# ME2/docs/WAKEWORD-DATASET-CONTRACT.md for the manifest schema this stands
# in for.
# ---------------------------------------------------------------------------


@pytest.fixture
def wakeword_fake_manifest_factory(tmp_path, vcm_wav_factory):
    """Factory: build a small synthetic wakeword `manifest.csv` (real
    10-column schema, per docs/WAKEWORD-DATASET-CONTRACT.md section 2)
    plus matching audio files, under a throwaway subset directory.

    `build(specs, subset_name="positives_real")` takes a list of dicts,
    one per manifest row, each with:
      - `source_dataset` (str, default "picovoice")
      - `label` (str, default "_wakeword_")
      - `duration_s` (float, default 0.5)
      - `filename` (str, default f"{i:05d}.wav")
      - `source_relpath` (str, default f"{source_dataset}/{filename}")
      - `group_id` (str, default the filename's stem)
      - `split` (str, default "" -- unassigned, per contract section 2)
      - `silence` (bool, default False) -- written straight through to
        `vcm_wav_factory`.
      - `ref_voice` / `noise_source_file` / `snr_db` / `speech_start_s` /
        `speech_end_s` (str, no default) -- optional per-subset extension
        columns (contract section 2; `speech_start_s`/`speech_end_s` added
        for feature `wakeword-dscnn`, section 8). Only added to the
        written manifest's column set at all if at least one spec in this
        `build()` call sets one, so a caller building a plain 10-column
        subset sees exactly the same output as before this extension was
        added; specs that omit an extension key some other spec in the
        same call sets get `""` for it.

    Returns the built `manifest.csv` Path; audio lives alongside it at
    `audio/<source_dataset>/<filename>`, matching the real output layout
    (docs/WAKEWORD-DATASET-CONTRACT.md section 6).
    """

    def build(specs: list[dict], subset_name: str = "positives_real") -> Path:
        root = tmp_path / f"fake_wakeword_{subset_name}"
        root.mkdir(exist_ok=True, parents=True)

        extension_keys = [
            key
            for key in ("ref_voice", "noise_source_file", "snr_db", "speech_start_s", "speech_end_s")
            if any(key in spec for spec in specs)
        ]
        fieldnames = WAKEWORD_MANIFEST_FIELDS + extension_keys

        rows: list[dict] = []
        for i, spec in enumerate(specs):
            source_dataset = spec.get("source_dataset", "picovoice")
            filename = spec.get("filename", f"{i:05d}.wav")
            duration_s = spec.get("duration_s", 0.5)
            rel_path = f"audio/{source_dataset}/{filename}"

            vcm_wav_factory(
                root / rel_path, duration_s=duration_s, silence=spec.get("silence", False)
            )

            row = {
                "filename": filename,
                "path": rel_path,
                "label": spec.get("label", "_wakeword_"),
                "duration": f"{duration_s:.6f}",
                "sample_rate": "16000",
                "resampled": "False",
                "source_dataset": source_dataset,
                "source_relpath": spec.get("source_relpath", f"{source_dataset}/{filename}"),
                "group_id": spec.get("group_id", Path(filename).stem),
                "split": spec.get("split", ""),
            }
            for key in extension_keys:
                row[key] = spec.get(key, "")
            rows.append(row)

        manifest_path = root / "manifest.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        return manifest_path

    return build


@pytest.fixture
def vcm_quartznet_checkpoint(tmp_path, vcm_fake_manifest_factory):
    """A real QuartzNet checkpoint produced by calling `vcm.train.main` (one
    CPU epoch on a tiny fake manifest) -- never a hand-built dict. Returns
    `(checkpoint_path, manifest_path, out_dir)`."""
    from me2_voicegen.vcm import train

    specs = [
        {"bucket": "target_commands", "source_dataset": "optionb", "label": "STOP",
         "split": split, "transcript": "stop", "duration_s": 1.0}
        for split in ["train"] * 6 + ["val"] * 4
    ]
    manifest = vcm_fake_manifest_factory(specs)
    out_dir = tmp_path / "quartznet_run"
    train.main(
        [
            "--manifest", str(manifest), "--out-dir", str(out_dir),
            "--preset", "quartznet5x3", "--max-epochs", "1", "--device", "cpu",
            "--num-workers", "0", "--batch-size", "2", "--max-minutes", "5",
        ]
    )
    return out_dir / "checkpoints" / "checkpoint.pt", manifest, out_dir
