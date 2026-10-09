Mixing a highly curated real-audio base with massive synthetic TTS generation is the exact blueprint used by modern edge KWS frameworks like openWakeWord, which successfully trains its feature extraction and classification backbones on synthetic data. Recent literature confirms that combining a small baseline of real audio (e.g., ~2,000 utterances across 100 speakers) with heavily augmented TTS data minimizes development cost while keeping accuracy highly competitive with models trained on millions of real recordings.

**Top Offline TTS Models (≤ 8 GB VRAM)**

* **XTTS-v2 (Coqui):** Requires ~3.5–4.5 GB VRAM in FP16. It supports zero-shot voice cloning from just a 3-second reference audio clip. You can pull thousands of random speaker clips from public datasets and feed them into XTTS to generate your 16 commands in distinct, unique voices.
* **CosyVoice 2 (Alibaba):** Comfortably fits within an 8 GB VRAM footprint. Recent studies generating synthetic multi-command datasets (like the SynTTS-Commands dataset) leverage CosyVoice 2 combined with speaker embeddings from public corpora to achieve scalable, high-fidelity data generation.
* **Piper TTS:** Requires under 1 GB VRAM and runs exceptionally fast, making it ideal for bulk generation. While it lacks dynamic zero-shot cloning, it provides hundreds of highly optimized, pre-trained voice checkpoints across various accents to quickly build a baseline.

**The Synthetic Augmentation Pipeline for your MLOps Projects file**

* **Acoustic Diversity Injection:** Because base TTS output lacks the natural variance of human speech, feed reference audio from diverse public corpora (e.g., Common Voice, VoxCeleb) into a zero-shot TTS model to force variations in pitch, prosody, and accent.
* **Phonetic Adversaries for the `_unknown_` Class:** Use the TTS engine to generate words that share syllables or sound phonetically similar to your wakewords (e.g., generating "play musket" to contrast with "play music"). This forces the classifier to pick up on exact phonetic differences rather than general voice spectra.
* **Environment Degradation:** Because TTS generation completely bypasses microphone acoustics, the clean synthetic audio must be systematically degraded. Apply synthetic Room Impulse Responses (RIRs) and background noise (SNR variations) over the generated TTS audio to simulate real-world distance and hardware distortion.


Gaussian noise at different frequencies (white, pink, or brown noise) is a fast way to bootstrap the `_silence_` class for a quick POC, but relying solely on mathematical noise will eventually cause real-world false positives. A robust `_silence_` class must include actual environmental audio—HVAC hums, distant traffic, keyboard clacking, and microphone static—because edge microphones capture transient room dynamics that flat Gaussian distributions cannot replicate.

**POC 6-Command Schema & Adversaries**

| Target Command (`wakeword`) | Phonetic Adversaries (`_unknown_` class) |
| --- | --- |
| `play music` | `play musket`, `clay magic`, `pay mucus` |
| `lights on` | `kites on`, `bites on`, `lights gone` |
| `lights off` | `kites off`, `bites off`, `lights cough` |
| `set timer` | `pet tiger`, `wet primer`, `step higher` |
| `weather` | `leather`, `feather`, `measure` |
| `stop` | `stomp`, `shop`, `slop` |

**POC Implementation Steps**

* **Baseline Generation:** Use your chosen TTS model (like Piper or CosyVoice 2) to generate 50–100 variations of the 6 target commands using different voices and speeds.
* **Adversarial Generation:** Generate 20–50 variations of the phonetic adversaries above to populate the `_unknown_` class. This forces the model to learn the exact consonants of your commands.
* **Negative Speech:** Mix in a few minutes of random, conversational audio (e.g., from Common Voice or a public domain audiobook) into the `_unknown_` class so the model learns to ignore natural talking.
* **Silence Construction:** For this specific POC, generate 1.5-second chunks of Gaussian, pink, and brown noise at varying volume levels to serve as your `_silence_` baseline.

You can log this initial dataset matrix directly into your MLOps Projects file to establish your v1.0 data lineage.

Are you planning to write a Python script to inject that Gaussian noise dynamically on the fly during the training loop, or pre-render the noise files into a static folder?
