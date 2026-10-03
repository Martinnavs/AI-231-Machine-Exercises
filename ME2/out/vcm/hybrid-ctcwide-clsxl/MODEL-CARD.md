# Hybrid: wide CTC + xl classifier heads (policy A), both models resident

Two checkpoints decoded together by `me2_voicegen.vcm.hybrid.HybridDecoder` (`scripts/hybrid_decode_file.py` shows the call):
the grammar-constrained CTC decode of the wide model answers when it accepts (threshold -0.1, incomplete-prefix margin 4.0, beam 50);
otherwise the xl model's intent head answers when its max-softmax is >= 0.8787 (fit on validation to its own CTC false-accept rate, never tuned on test).
Per clip it returns a trace (each component's intent and confidence, agreement, which path answered) and keeps counters of which component was right.

| | `ctc-wide/` | `cls-xl/` |
|---|---|---|
| preset | `quartznet5x3-wide-heads` (channels 416, epilogue 512, head 192) | `quartznet5x3-xl-heads` (channels 648, epilogue 800, head 256) |
| parameters | 4,181,309 | 9,989,677 |
| role | CTC answer | intent and slot heads (fallback) |
| best epoch / val loss | 38 / 0.3906 | 23 / 0.3363 |
| checkpoint | 17 MB | 39 MB |
| `export/vcm_model.int8.onnx` | 4.16 MB (CTC output only) | 9.86 MB (CTC output only) |

- **Data:** public `airimonda/ai231-me2-voice-commands` v2 (train split incl. val speakers and `synthetic_negatives`) plus the public persona clips
  (`martinnavarez/ai231-fil-supplemental-data` and the capped fil50 persona rows), the "H-all" manifest. Noise and babble are the dataset's own
  clips and synthetic coloured noise; no ESC-50. Inherits the dataset source terms, several of which are non-commercial (the checkpoint `license` field says so).
- **Training:** `vcm.train --seed 0 --p-rir 0.7 --p-timestretch 0.25 --p-noise 0.5 --p-babble 0.15 --noise-source dataset --perturbation-plan table --dump-plan
  --skip-prenoised --noise-random-offset --onecycle-epochs 60 --max-epochs 70 --patience 20 --max-minutes 135` (`train.log` in each folder). One seed each.
- **Results** (`eval/hybrid-end-to-end.md`, scored from audio): ai231 test clean 98.8%, perturbed 94.9%, holdout (186, real Filipino speaker) 77.4%,
  129 real recordings 98.4%, leak-free internal held-out 97.8% / 87.3% perturbed. Versus the vanilla networks and old production: `docs/AI231-FIL50.md`.
- **Known limits:** non-command false accepts 3.4% (clean) to 6.1% (perturbed) on the ai231 negatives, mostly from the classifier fallback (CTC alone: 0.9%
  and 2.5%); the wake word sits in front in deployment. 19% of holdout commands are still rejected. Single seed; small holdout.
- **ONNX caveat:** the exported ONNX files contain the CTC output only. The classifier heads are not exported, so the xl classifier needs the PyTorch
  checkpoint (`forward_heads`) until a heads export exists. Pi latency and memory are not measured (no Pi hardware on the node).
- **Status:** best hybrid so far for streaming tests, not promoted. Ticket 04 is deferred (see `docs/AI231-FIL50.md`).
