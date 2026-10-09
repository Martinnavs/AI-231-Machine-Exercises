Here is the literature breakdown to guide the preliminary approach for your MLOps Projects file.

**Core Architectures (The "Tiny" Backbones)**

| Model / Literature | Key Innovation | Why It Fits Your Use Case |
| --- | --- | --- |
| **DS-CNN** (*"Hello Edge," Zhang et al.*) | Depthwise Separable Convolutions | The canonical baseline for edge keyword spotting. Runs comfortably under 5ms on a Raspberry Pi with an incredibly small memory footprint (~400KB quantized). |
| **BC-ResNet** (*Kim et al.*) | Broadcasted Residual Learning | Achieves state-of-the-art accuracy by combining 1D temporal convolutions with 2D frequency features. Hits >96% accuracy on standard benchmarks with under 10k parameters. |
| **MatchboxNet** (*Majumdar et al.*) | 1D Time-Channel Separable Convolutions | Extremely parameter-efficient residual network. Scales down beautifully for 1.5-second audio windows without sacrificing accuracy on 10-16 class tasks. |

**Dataset Construction & Framing**

* **The Speech Commands Blueprint** (*Warden, 2018*): This is the industry-standard methodology for framing the exact problem you are solving. It details how to balance target words with `_silence_` (ambient room noise) and `_unknown_` (out-of-vocabulary conversational speech) classes to drastically reduce false positives in always-on environments.
* **Synthetic Data Bootstrapping**: Recent TinyML workflows heavily utilize zero-shot TTS engines to generate custom wakewords across hundreds of synthetic voices, pitches, and speeds. This prevents the "cold start" problem before you collect real human samples.
* **Spectrogram Masking**: Papers detailing **SpecAugment** (*Park et al., 2019*) show that randomly masking blocks of time and frequency channels directly on the Mel-spectrogram during training forces the network to rely on the entire phonetic structure, making the model highly robust to varying microphone hardware and room acoustics.

**Deployment & Edge Optimization**

* **INT8 Quantization**: Standard edge literature emphasizes converting FP32 models to 8-bit integers via TensorFlow Lite or ONNX Runtime. This reduces model size by 4x and ensures the Raspberry Pi experiences zero thermal throttling during 24/7 continuous listening.
* **Streaming Inference Pipelines**: Production systems utilize a sliding FIFO buffer. You extract 30ms frames with a 10ms stride, continuously updating a 1.5-second spectrogram window. A posterior handling module (rolling average) is then applied to the output probabilities to prevent duplicate triggers from a single command.
