from __future__ import annotations

import logging

import numpy as np

from me2_voicegen.generation.cosyvoice_env import COSYVOICE_DIR

from .base import Synthesizer, SynthesisResult, VoicePrompt

logger = logging.getLogger(__name__)


def _load_auto_model():
    """Import AutoModel lazily so this module stays importable (list_backends(),
    --help, the fast test suite) with no vendor clone, no weights, and no GPU.
    Factored out of __init__ (rather than inlined) so tests can monkeypatch this
    one seam instead of needing a real `cosyvoice` package on sys.path."""
    from me2_voicegen.generation.cosyvoice_env import add_cosyvoice_to_syspath

    add_cosyvoice_to_syspath()
    from cosyvoice.cli.cosyvoice import AutoModel

    return AutoModel


def _default_model_dir() -> str:
    from me2_voicegen.generation.download_model import target_dir

    return str(target_dir())


class CosyVoice2Synthesizer(Synthesizer):
    # Discovered generically by generate_sample.py via getattr() - see its
    # _build_prompt() - so the CLI never hardcodes a vendor-specific asset path.
    DEFAULT_PROMPT_WAV = COSYVOICE_DIR / "asset" / "zero_shot_prompt.wav"
    DEFAULT_PROMPT_TEXT = "希望你以后能够做的比我还好呦。"

    def __init__(
        self,
        model_dir: str | None = None,
        device: str = "auto",
        fp16: bool = False,
    ) -> None:
        """model_dir defaults to models/CosyVoice2-0.5B (see Ticket 02's
        download_model.py). device is accepted for interface consistency with
        the CLI's common flags, but CosyVoice2 has no device override hook of
        its own - CosyVoiceModel always auto-detects cuda internally
        (torch.cuda.is_available()) with no way to force CPU-only inference.
        An explicit device="cuda" is validated eagerly (fail loud) rather than
        silently falling back once inference starts."""
        if device == "cuda":
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "device='cuda' requested but no CUDA device is available"
                )
        elif device == "cpu":
            logger.warning(
                "CosyVoice2 has no CPU-only inference mode; device='cpu' is "
                "recorded but not enforced - it will still use cuda if available"
            )

        self._device = device
        resolved_model_dir = model_dir if model_dir is not None else _default_model_dir()
        auto_model = _load_auto_model()
        self._model = auto_model(model_dir=resolved_model_dir, fp16=fp16)

    def synthesize(self, text: str, prompt: VoicePrompt | None = None) -> SynthesisResult:
        if prompt is None:
            raise ValueError(
                "CosyVoice2Synthesizer requires a VoicePrompt (zero-shot voice "
                "cloning only; no non-cloning inference path is wired up)"
            )
        if prompt.text is None:
            raise ValueError(
                "CosyVoice2Synthesizer requires prompt.text (a transcript of "
                "prompt.wav_path) for inference_zero_shot"
            )

        import torch

        chunks = [
            chunk["tts_speech"]
            for chunk in self._model.inference_zero_shot(
                text, prompt.text, str(prompt.wav_path)
            )
        ]
        if not chunks:
            raise RuntimeError("CosyVoice2 inference_zero_shot yielded no audio chunks")

        audio_tensor = torch.cat(chunks, dim=-1)
        audio = audio_tensor.detach().cpu().numpy().astype(np.float32)
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]

        return SynthesisResult(audio=audio, sample_rate=self._model.sample_rate)

    def convert_voice(self, source_wav_path: str, prompt_wav_path: str) -> SynthesisResult:
        """Voice conversion (no text): re-renders source_wav_path's own speech
        content in prompt_wav_path's timbre via CosyVoice2's inference_vc. Not
        part of the Synthesizer ABC (that contract is text-in) - this is an
        additional capability specific to this backend, discovered via
        getattr() the same way DEFAULT_PROMPT_WAV is, so a backend without it
        simply doesn't support voice conversion."""
        import torch

        chunks = [
            chunk["tts_speech"]
            for chunk in self._model.inference_vc(source_wav_path, prompt_wav_path)
        ]
        if not chunks:
            raise RuntimeError("CosyVoice2 inference_vc yielded no audio chunks")

        audio_tensor = torch.cat(chunks, dim=-1)
        audio = audio_tensor.detach().cpu().numpy().astype(np.float32)
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]

        return SynthesisResult(audio=audio, sample_rate=self._model.sample_rate)
