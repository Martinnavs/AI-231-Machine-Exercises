"""Download CosyVoice2-0.5B weights into models/CosyVoice2-0.5B.

Idempotent: if the expected file manifest is already present, this is a
fast no-op (no network call). Run via `make download-model` /
`uv run python -m me2_voicegen.generation.download_model`.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from . import cosyvoice_env

logger = logging.getLogger(__name__)

MODELSCOPE_MODEL_ID = "iic/CosyVoice2-0.5B"
HUGGINGFACE_REPO_ID = "FunAudioLLM/CosyVoice2-0.5B"

# CosyVoice2.__init__ overrides qwen_pretrain_path to <model_dir>/CosyVoice-BlankEN,
# so that subdirectory is mandatory, not optional, despite not being a top-level file.
#
# spk2info.pt is deliberately NOT in this manifest: verified against both live
# sources (modelscope HubApi.get_model_files / huggingface_hub.list_repo_files)
# that neither ships it for CosyVoice2-0.5B, and vendor/CosyVoice's
# cli/frontend.py already treats it as optional (os.path.exists check, falls
# back to an empty spk2info dict = zero-shot-only, no pretrained speakers).
MANIFEST = (
    "cosyvoice2.yaml",
    "llm.pt",
    "flow.pt",
    "hift.pt",
    "campplus.onnx",
    "speech_tokenizer_v2.onnx",
    "CosyVoice-BlankEN",
)


def target_dir() -> Path:
    return cosyvoice_env.MODELS_DIR / "CosyVoice2-0.5B"


def missing_manifest_entries(model_dir: Path) -> list[str]:
    """Return manifest entries missing from model_dir, following symlinks.

    modelscope's snapshot_download(local_dir=...) may populate local_dir with
    symlinks into its own cache rather than real files; Path.exists()/is_dir()
    already follow symlinks, so no special-casing is needed beyond that.
    """
    missing = []
    for entry in MANIFEST:
        path = model_dir / entry
        if entry == "CosyVoice-BlankEN":
            if not path.is_dir() or not any(path.iterdir()):
                missing.append(entry)
        elif not path.is_file():
            missing.append(entry)
    return missing


def download_modelscope(model_dir: Path) -> None:
    from modelscope import snapshot_download

    snapshot_download(MODELSCOPE_MODEL_ID, local_dir=str(model_dir))


def download_huggingface(model_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(HUGGINGFACE_REPO_ID, local_dir=str(model_dir))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=("modelscope", "huggingface"),
        default="modelscope",
        help="download source (default: modelscope)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="enable debug-level logging"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    model_dir = target_dir()

    missing = missing_manifest_entries(model_dir)
    if not missing:
        logger.info("model already present and complete at %s, skipping download", model_dir)
        return 0

    model_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "downloading CosyVoice2-0.5B via %s to %s (missing: %s)",
        args.source,
        model_dir,
        ", ".join(missing),
    )
    if args.source == "modelscope":
        download_modelscope(model_dir)
    else:
        download_huggingface(model_dir)

    missing_after = missing_manifest_entries(model_dir)
    if missing_after:
        missing_str = ", ".join(missing_after)
        raise RuntimeError(
            f"model download incomplete at {model_dir}: missing {missing_str}"
        )

    logger.info("download complete and manifest verified at %s", model_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
