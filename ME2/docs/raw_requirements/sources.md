https://github.com/voytekresearch/Tutorials/blob/master/06-ColoredNoise.ipynb - how to do colored noise
https://www.kaggle.com/datasets/minsithu/audio-noise-dataset - background noise
https://www.kaggle.com/datasets/lazyrac00n/speech-activity-detection-datasets - Noizeus dataset for noise


To sample synthetic voices
P1 (if able to be executed here):* **CosyVoice 2 (Alibaba):** Comfortably fits within an 8 GB VRAM footprint. Recent studies generating synthetic multi-command datasets (like the SynTTS-Commands dataset) leverage CosyVoice 2 combined with speaker embeddings from public corpora to achieve scalable, high-fidelity data generation.

P2 (if not possible): 
* **XTTS-v2 (Coqui):** Requires ~3.5–4.5 GB VRAM in FP16. It supports zero-shot voice cloning from just a 3-second reference audio clip. You can pull thousands of random speaker clips from public datasets and feed them into XTTS to generate your 16 commands in distinct, unique voices.
