**ENGINEERING SPECIFICATION: VCM AND WAKEWORD RE-ARCHITECTURE (BRANCH: optionb-grammar-v2)**

**1. DATA ARCHITECTURE & PIPELINE**

* **Positive Class Restructuring (Recall Optimization):** The Voice Command Model (VCM) requires a 50/50 target-class balance between Filipino and non-Filipino speech to close the empirical recall gap.
* **Phase 1 (Immediate Execution):** Utilize 17 verified Filipino-English "References" voices via zero-shot Text-to-Speech (TTS) using CosyVoice (Du et al., 2024). This guarantees transcript integrity, as pilot testing revealed that zero-shot TTS on Tagalog prompts (`sapinsapin` corpus) causes cross-lingual hallucination and failure rates exceeding 50%.
* **Phase 2 (Fast-Follow Experiment):** Process the 124 `sapinsapin` Tagalog speakers using Voice Conversion (`convert_voice` timbre transfer) rather than zero-shot generation. This captures the spectral characteristics of the diverse speaker pool without compromising the underlying English phonetic cadence.


* **Dual Negative Mining (Precision Optimization):**
* **Wakeword Accent-Matched Negatives:** The Depthwise Separable CNN (DS-CNN) (Zhang et al., 2017) requires accent-matched negative mining. The exact 17 References and 124 `sapinsapin` voices used for the positive class must be injected into the `_unknown_` class speaking conversational phrases to prevent the DS-CNN from degrading into a simple accent detector.
* **VCM Open-Vocabulary Hard Negatives:** The 4-hour podcast soak test generated 142 Acoustic Ghosts, largely driven by short intents like STOP, TIME, and PAUSE. These identical audio clips will be harvested and injected directly into the CTC training manifest with empty transcripts (`""`). This explicitly forces the network to predict the CTC `blank` class for these precise phonetic collisions.




* **Acoustic and Environmental Perturbation:**
* Break TTS prosodic homogeneity by applying aggressive time-stretching ($0.85\times$ to $1.15\times$) prior to log-mel feature extraction, which currently uses a 30 ms window and 10 ms hop on a 16 kHz waveform.


* Convolve ESC-50 background noise strictly within split-isolated folds to close the far-field physical realism gap. All dataset iterations and yields must be tracked verbatim in the MLOps Projects file.



**2. TRAINING CONFIGURATION**

* **Architectural Boundary Enforcement:** The system strictly maintains its split Deep Learning and Natural Language Processing architecture. The acoustic model will not be scaled up to absorb the grammar constraint.


* The VCM remains a ~1.01M parameter 1D Time-Channel-Separable Convolutional Neural Network built on the MatchboxNet architecture (Majumdar & Ginsburg, 2020).


* It serves exclusively as a closed-grammar phonetic aligner, structurally constrained to 19 intents and the 129 valid phrasings defined by the trie.


* The DS-CNN assumes the entirety of the open-mic false-accept suppression burden.


* **Hyperparameter Lock (MatchboxNet-CTC):**
* **Architecture:** Retain the `optiond` configuration utilizing 5 residual blocks and 128 channels.


* **Optimizer:** AdamW at a learning rate of 1e-4, weight decay of 1e-2, and gradient clipping at 5.0.


* **Schedule & Loss:** OneCycleLR schedule with CTC loss (`blank=0, zero_infinity=True`), a batch size of 16, and mixed precision training.




* **Wakeword Training Shift:** Over-sample the harvested hard conversational negatives during DS-CNN training to ensure robust continuous-listening performance in the presence of overlapping regional speech.

**3. DECODING & REJECTION GATING**

* **Unconstrained Greedy Gating (Lexical Intrusion Defense):** The parallel unconstrained CTC greedy decode pass, which currently runs purely to report raw text and gap scores, will be promoted to an active rejection gate.


* **Logic:** Compute the log-likelihood delta ($\Delta = \text{Score}_{\text{unconstrained}} - \text{Score}_{\text{constrained}}$). If the unconstrained path discovers significantly higher probability mass in non-grammar characters (e.g., predicting `p-l-a-y-i-n-g` against the forced `s-t-o-p`), the activation is immediately rejected.


* **Dense Phonetic Scoring (Anti-Blank Dilution):** The current confidence metric normalizes the beam's total log-probability by the entire window's frame count. This mathematically allows near-1.0 confidence on CTC blank frames to mask weak character activations.


* **New Metric:** Acceptance confidence will be calculated exclusively over non-blank emission frames: $\text{Score}_{\text{dense}} = \frac{1}{\vert{}T_{\text{non-blank}}\vert{}} \sum_{t \in T_{\text{non-blank}}} \log P(c_t \mid x_t)$.


* **Intent-Length Dependent Thresholding:**
* The static global confidence evaluation grid (which optimized at points like -0.1 or -0.075) failed to suppress generic false accepts. This global approach will be discarded.


* Bespoke, stricter dense thresholds will be assigned to short, low-entropy intents (STOP, TIME, PAUSE). Longer intents (e.g., CREATE_REMINDER) will utilize looser thresholds due to higher natural phonetic entropy.
* The incomplete-prefix margin gate remains specifically designed to guard against half-spoken or truncated commands by tracking the best-scoring designated incomplete prefix.





**4. SERVING & DEPLOYMENT**

* **Sequential Hardware Gate:** Deployment on the target Raspberry Pi 4/5 hardware will execute via a strict sequential logic flow. The acoustic model, its shared frontend feature module, and the CPU-only beam search must remain entirely dormant until the DS-CNN wakeword engine explicitly triggers.


* **Quantization and Memory Footprint:** The VCM will maintain its INT8 export size of 0.26 MB to preserve sub-real-time streaming latency. The new dense phonetic scoring and greedy gate checks will be natively compiled into the existing pure-Python CTC prefix search loop to avoid introducing execution overhead.
