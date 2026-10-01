# Edge CTC alternatives to the current MatchboxNet-style model

> **Outcome (2026-10-01):** the proposed QuartzNet student was built and measured, and beat the
> production `optiond` model on the noisy gate, size and latency in two seeds. Width ended at 192 channels
> (911k parameters), set by the INT8 size gate, and the baseline was `optiond` rather than `optionc`. See
> `docs/QUARTZNET-STUDENT.md`.

Research date: 2026-09-22

## Decision

Benchmark a **scaled QuartzNet-5x3-style character CTC student with 2x temporal
subsampling**, capped at 1.0M parameters, against the current `optionc` model.
This is a new locally trained configuration, not the stock 6.7M-parameter
QuartzNet-5x5 checkpoint.

No maintained, pretrained general-ASR CTC checkpoint found in this review meets
all of this repository's constraints at once: comparable accuracy on this
closed-command task, roughly the current latency, and the existing 5 MB INT8
model budget. Stock Citrinet, Conformer, QuartzNet, wav2vec2, and distilled
self-supervised models are useful references or teachers, but are too large to
be the deployable peer.

## What is deployed now

The repository's model is a custom **MatchboxNet-style acoustic encoder trained
with CTC**, not the classifier evaluated in the MatchboxNet paper.

- Frontend: 16 kHz mono, 40 log-mel bins, 30 ms window, 10 ms hop, utterance
  normalization (`common/features.py`).
- Model seam: `(B, 40, T)` features to `(B, T, 29)` logits; the 29 symbols are
  CTC blank, `a-z`, space, and apostrophe (`vcm/model.py`, `vcm/alphabet.py`).
- Encoder: a regular Conv1d prologue, residual depthwise/pointwise TCS blocks,
  a regular Conv1d epilogue, and a linear head. Every layer has stride 1, so the
  output has one step per input feature frame.
- Training: character CTC with `input_lengths` copied directly from frontend
  lengths. The checkpoint records the concrete model config.
- Deployment: ONNX Runtime receives features and returns logits; NumPy
  log-softmax produces `(T,29)` log-posteriors. The grammar-constrained Python
  CTC prefix beam search consumes that stable interface.
- Streaming: a ring buffer is evaluated at each stride, then decoded at
  threshold `-inf`; policy thresholding and debounce happen afterward.

Measured local baselines (1.5 s / 151-frame model-forward benchmark, one ORT
thread, AMD EPYC—not Raspberry Pi):

| Model | Parameters | INT8 ONNX | FP32 p50 | INT8 p50 | Test exact intent | Test babble/silence FA |
|---|---:|---:|---:|---:|---:|---:|
| `default` | 254,477 | 0.264 MB | 1.059 ms | 1.032 ms | 94.8% | 7.1% / 0% |
| `optionc` | 753,965 | 0.746 MB | 2.820 ms | 2.717 ms | 96.0% | 0% / 0% |

These are task metrics, not LibriSpeech WER. Published open-vocabulary WERs are
therefore not directly comparable. The runner also documents that, at its
shipped 2.5 s / beam-25 settings, about 97% of per-window compute was the Python
beam search and about 3% was ONNX forward. Model-only latency is not the right
optimization target.

## Literature and artifact screen

| Candidate | Evidence | Fit here |
|---|---|---|
| Stock QuartzNet-5x5 | Character CTC with repeated time-channel-separable convolution modules; the published small model is about 6.7M parameters. The paper demonstrates transfer learning, but on much larger corpora. [QuartzNet paper](https://arxiv.org/abs/1910.10261) | Architecture is highly compatible, but the stock model exceeds the 5 MB INT8 budget before graph overhead. A width-scaled 5x3 student is the best experiment. |
| Citrinet-256 | Citrinet adds squeeze-excitation and stronger time reduction to separable-convolution CTC. The smallest official English card reports about 10M parameters and 3.8/9.8 greedy WER on LibriSpeech test-clean/test-other; the `.nemo` artifact is 36.7 MB. [Paper](https://arxiv.org/abs/2104.01721), [official model card/files](https://huggingface.co/nvidia/stt_en_citrinet_256_ls/tree/main) | Credible accuracy reference or teacher, but an estimated ~10 MB of INT8 weights alone misses the model-size budget. It also uses a 256-piece tokenizer and a different frontend, so it is not a drop-in backend. |
| Conformer-CTC small | NVIDIA's maintained model card reports about 13M parameters, 3.7/8.1 LibriSpeech WER, and several-thousand-hour pretraining; its `.nemo` artifact is 49.1 MB. A community sherpa-onnx export is 85.3 MB FP32 / 46.4 MB INT8. [Official model card/files](https://huggingface.co/nvidia/stt_en_conformer_ctc_small/tree/main), [ONNX files](https://huggingface.co/csukuangfj/sherpa-onnx-nemo-ctc-en-conformer-small/tree/main) | Better general ASR does not offset a roughly 17x parameter increase over `optionc`; attention/LayerNorm also add an ARM runtime risk. Reject for this footprint. |
| wav2vec2 / distilled SSL CTC | wav2vec2 gives excellent low-label transfer, but the base encoder is about 95M parameters; even reduced students remain far beyond the current sub-1M target. [wav2vec2 paper](https://arxiv.org/abs/2006.11477), [footprint-reduction study](https://arxiv.org/abs/2103.15760) | Useful only as a training teacher or pseudo-labeler. Not a Pi peer under this budget. |
| Streaming Zipformer-CTC | sherpa-onnx has maintained streaming CTC examples and supports ARM/Raspberry Pi. [CTC example](https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-addon-examples/README.md#streaming-speech-recognition-with-zipformer-ctc), [platform support](https://github.com/k2-fsa/sherpa-onnx) | Strong deployment ecosystem, but available models/runtime contracts are materially larger and use their own frontend/tokenization/decoder. Better as a future runtime path than this A/B model. |
| EdgeSpeechNet / TinySpeech | These report excellent tiny-device results on Google Speech Commands, but they are fixed-class command classifiers, not sequence CTC acoustic models. [EdgeSpeechNets](https://arxiv.org/abs/1810.08559), [TinySpeech](https://arxiv.org/abs/2008.04245) | Relevant only if the product can abandon slots and open phrase composition. They do not satisfy the requested CTC comparison. |

The original MatchboxNet paper likewise evaluates limited-vocabulary
classification, not this repository's character-CTC adaptation. Its accuracy
should not be presented as evidence for the current model. [MatchboxNet
paper](https://arxiv.org/abs/2004.08531)

## Engineering evidence and caveats

- ONNX Runtime's current quantizer guidance recommends symmetric QInt8
  activations and weights with `reduce_range=False` on ARM, and says
  `per_channel=False` normally gives best CPU throughput. That matches this
  repository's current static QDQ choices. [ORT quantizer source guidance](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/quantize.py)
- Do not assume FP16 helps on Pi CPU. A Raspberry Pi 5 / AArch64 ORT issue
  reports FP16 inference four times slower than FP32 because inserted casts and
  slower MatMul/Gemm dominated. This is one report, not a universal benchmark.
  [ORT issue #25824](https://github.com/microsoft/onnxruntime/issues/25824)
- Do not assume more threads help. A Zipformer-CTC user reports ONNX inference
  slowing as thread count increased. Again, this is anecdotal and requires
  target-device measurement. [sherpa-onnx issue #2620](https://github.com/k2-fsa/sherpa-onnx/issues/2620)
- sherpa-onnx confirms CTC + HLG/WFST decoding is available on CPU. If Python
  beam search remains dominant after temporal subsampling, moving the decoder
  to a native implementation is likely a larger latency win than replacing a
  0.75M-parameter encoder. [sherpa-onnx discussion #1377](https://github.com/k2-fsa/sherpa-onnx/discussions/1377)

Forum/issues evidence above identifies risks and implementation options; it is
not used as proof of accuracy or Raspberry Pi latency.

## Proposed comparison model

Use a **QuartzNet-5x3-tiny** experiment with these hard constraints:

- Preserve the current 40-bin frontend and 29-character output vocabulary.
- Five residual macro-blocks, three depthwise-separable Conv1d modules per
  block, BatchNorm/ReLU/dropout, and channels selected to stay at or below
  1.0M parameters.
- Apply total temporal stride 2 initially. This changes a 151-step posterior
  to roughly 76 steps and should nearly halve the dominant decoder work while
  leaving enough CTC alignment room for long command transcripts. Treat stride
  4 as a later measured variant, not the default.
- Export only common ONNX operators (`Conv`, `BatchNormalization` folded at
  inference, `Relu`, `Add`, `Dropout` removed in eval, `Transpose`, `Gemm`).
- Produce both FP32 and statically calibrated QInt8 artifacts.

Required integration changes:

1. Introduce a model protocol/factory rather than hard-coding
   `MatchboxNetConfig` and `MatchboxNetCTC` in checkpoint loading and export.
2. Have each model calculate output lengths. The current training loop passes
   feature lengths directly to `CTCLoss`, which is wrong after striding.
3. Keep the ONNX boundary names and shapes (`features` to `logits`) unchanged;
   the current inference backend and decoder already accept arbitrary output
   time `T`.
4. Re-export, calibrate INT8, verify torch/ONNX posterior parity, and re-sweep
   thresholds on validation data. Confidence scores are normalized over output
   timesteps, so old thresholds are not portable across temporal resolutions.

## Acceptance gate

Run both models from the same seed/data/augmentation budget and select the new
model only if all conditions hold:

- test exact-intent accuracy >= 96.0%, with slot accuracy reported separately;
- babble and silence false-accept rates no worse than `optionc` at a threshold
  chosen on validation only;
- INT8 ONNX <= 1.0 MB (tighter than the original 5 MB ceiling so the comparison
  truly preserves the current footprint class);
- end-to-end p95 on a Raspberry Pi 4 and/or 5, including feature extraction,
  ONNX forward, log-softmax, and grammar decoding, no slower than `optionc`;
- no dropped streaming windows at the shipped window/stride settings;
- process RSS measured in a fresh process after model load, not inferred from
  model-file size or the current EPYC process high-water mark.

If the tiny QuartzNet student does not beat `optionc`, the next experiment
should be native/WFST decoder optimization—not a larger stock ASR encoder.
