# Wakeword DS-CNN ONNX export/quantization benchmark

**Hardware: AMD EPYC 7742 (256-thread node) estimate. NOT a Raspberry Pi 4/5 (no such hardware exists on this node) measurement.**

Window: 1.5s (151 log-mel frames). onnxruntime threads: intra_op=1, inter_op=1. Checkpoint: `out/wakeword-fil50/checkpoints/checkpoint.pt` (preset=default).
**Confirmed target device: Raspberry Pi 4 (2018), 4GB RAM (Broadcom BCM2711, quad-core Cortex-A72) -- confirmed target, not yet measured on this node.**

| variant | size (MB) | p50 latency (ms) | p95 latency (ms) | process peak RSS (MB) | inference-only RSS delta (KB) | note |
|---|---|---|---|---|---|---|
| fp32 ONNX | 0.119 | 0.252 | 0.261 | 432.102 | 0 | AMD EPYC 7742 (256-thread node) estimate |
| INT8 ONNX (static, val-calibrated) | 0.047 | 0.144 | 0.155 | 472.102 | 0 | AMD EPYC 7742 (256-thread node) estimate |

**Reading the RSS columns (tech-lead review R3-2):** "process peak RSS" is the absolute `ru_maxrss` high-water mark at the end of the timed loop -- the real total-footprint figure to compare against the peak-RAM budget below. "inference-only RSS delta" is only the *growth* in that high-water mark caused by the timed inference loop itself, sampled after the model/session were already loaded; it is commonly 0 because load-time RSS (model weights, ORT session, allocator arenas) usually already exceeds anything a single window's inference allocates. A `0` there means "inference didn't push the high-water mark higher," not "inference used no memory."

## Original spec budgets (indicative only, not a pass/fail claim about real RPi behavior)

- latency <= 20.0 ms per 100ms frame
- INT8 model size <= 5.0 MB
- peak RAM <= 25.0 MB (compare against the "process peak RSS" column above, not the RSS delta)

Original spec budgets for real RPi 4/5 hardware -- shown here as indicative context only, NOT as a pass/fail claim about this node's numbers (which are all AMD EPYC 7742 (256-thread node) estimate).
