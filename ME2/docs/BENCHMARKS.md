# Benchmarks: the hybrid (wide CTC + XL classifier heads)

Full tables behind the [README](../README.md) summary: whole-clip accuracy by split, seen vs unseen data, the unseen-noise replay, and the soak test on a Raspberry Pi 4 and a server.
How far to trust them: [`CURRENT-MODEL.md`](CURRENT-MODEL.md), "How far to trust the numbers". The soak method and tuning: [`SOAK-TEST.md`](SOAK-TEST.md).

## Headline result (test split)

Current model: the **hybrid** (wide CTC 4,181,309 params + XL classifier heads 9,989,677 params = 14.17 M parameters; **14.3 MB INT8 ONNX** = 4.16 MB + 10.10 MB; 56 MB of fp32 checkpoints),
decision rule "CTC answer, else classifier answer above 0.8787", scored whole-clip on the speaker-disjoint ai231 **test** split (INT8 ONNX; 3,823 in-scope clips with an exact variation,
76 out-of-scope clips, 250 synthetic negatives). One training run per network, so one seed. Source: `out/vcm/hybrid-ctcwide-clsxl/eval/headline-metrics.md` (`scripts/hybrid_metrics.py`).

| Metric (test) | Hybrid, INT8 | Wide CTC alone |
| --- | ---: | ---: |
| Variation balanced accuracy (93 variations + OOS) | 0.9877 | 0.9524 |
| ... on human voices only | 91.8% (214/233) | 84.5% (197/233) |
| ... on synthetic voices only | 99.3% (3,564/3,590) | 95.7% (3,436/3,590) |
| Command + slot accuracy (in scope) | 98.8% (3,778/3,823) | 95.0% (3,633/3,823) |
| Command accuracy | 98.9% | 95.0% |
| Slot accuracy (command right, slotted clips) | 99.9% (2,342/2,344) | 100.0% (2,232/2,232) |
| Out-of-scope false accept | 3.9% (3 of 76) | 0.0% (0 of 76) |
| In-scope false reject | 1.0% (40/3,823) | 4.9% (187/3,823) |
| Synthetic-negative misfire (250 clips) | 2.8% (7/250) | 1.2% (3/250) |

On the perturbed gate (room reverb + dataset noise, fixed seed): hybrid 94.8% vs CTC alone 87.8%. ONNX fp32 equals PyTorch; INT8 costs at most one clip per set. Other models for scale
(whole-clip, ai231 test): the earlier 1 MB QuartzNet trained on ai231 only (`v2s1-heads-A`, archived): CTC 87.5%, classifier 95.4%; an XL network alone: CTC 94.0%, classifier 97.7%.
**Not a size-matched comparison**: this model is about 15 times larger than the 0.95 M-parameter QuartzNet, and the XL network alone is as large as the hybrid's heads; the sizes are in [`CURRENT-MODEL.md`](CURRENT-MODEL.md).

**Results by split** (hybrid, INT8; "accuracy" = command and slot both right; exact-variation clips only, so the human-voice counts are smaller than in the full splits):

| Split | Speakers | In-scope clips | Accuracy | Command acc. | Human voices | Synthetic voices | Out of scope accepted | In scope rejected |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Val (used for early stopping and the 0.8787 threshold) | unseen | 1,001 | 99.1% | 99.5% | 100.0% (57) | 99.0% (944) | 5.6% (1 of 18) | 0.5% |
| Test | unseen | 3,823 | 98.8% | 98.9% | 91.8% (233) | 99.3% (3,590) | 3.9% (3 of 76) | 1.0% |
| Holdout (whole clip, offline) | unseen | 186 + 16 | 76.9% | 78.0% | 50.0% (86) | 100.0% (100) | 6.2% (1 of 16) | 20.4% |

Train is not scored here (the training manifest also holds persona clips; the train-to-test gap on human voices is the weakness to look at: the holdout's human voices are one real Filipino speaker, 50.0% vs
91.8% on the test split's human voices). Val chose every setting, so it is not an independent estimate, and the test split was scored many times during the project (`docs/AI231-FIL50.md`).
The wide CTC alone scores 95.8% / 95.0% / 66.7% on val / test / holdout, and 30.2% on the holdout's human voices.

**Seen vs unseen** (what the networks saw in training, relative to the ai231 **test** split and the **holdout**; checked 2026-10-03 on the training manifest `ai231-fil50-supp`, 18,300 train clips):

| | Test (4,367 ai231 in-scope clips + 2,946 persona-test clips) | Holdout (186 clips) | How checked |
| --- | --- | --- | --- |
| Audio clips | **unseen**: 0 byte-identical to a train clip | **unseen**: 0 of 186 | SHA-1 of every in-scope train, test and holdout file |
| Speakers / voices | **unseen**: 104 speaker and voice groups, none in train; the 5 persona voices are held out (10 train, 2 val, 5 test) | **unseen**: 4 groups, none in train (one real Filipino speaker, synthetic voices, 2 Fluent Speech Commands clips) | `group_id` overlap with train |
| Source corpora (SLURP, SNIPS, Fluent Speech Commands, xela, group recordings, the synthetic-voice pipeline) | seen as corpora (other speakers of the same sources are in train) | seen as corpora | source of each clip |
| Wordings | **seen**: the grammar is closed, all 93 variations are in train | **seen** | by design |
| Accent | partly seen: the Filipino-accented persona clones are in train; real Filipino speech is 5.6% of the train command clips | the real speaker is unseen, the accent is only seen through clones and a few real speakers | dataset card |
| Same person under another id | **not verified**: whether a persona reference voice or a synthetic voice is the same person as a real test or holdout speaker is unknown (dataset card caveat) | not verified | open item |
| Noise clips (perturbed gate) | **unseen**: the gate mixes the split's own noise clips | n/a | `vcm.noisy_eval` |
| Room impulse responses (perturbed gate) | **seen**: training and the gate both draw the 200 synthetic rooms from seed 0 (same generator, same seed, identical pool) | seen | `build_rir_pool` with `crc32("0:rir_pool")` |
| Soak test: ambient noise | partly seen: 131 of the 202 sessions use a noise clip from the ai231 **train** split (the training noise pool); 71 use val or test clips | | `noise_filename` per session |
| Soak test: room impulse responses | seen (same seed-0 pool) | | `soak/holdout-wake-gap-v1/continuous.json` |
| Soak test: wake-word clips | from the wake-word `test` split, not used to train the wake word | | `build_soak_audio.py --wake-split test` |
| Thresholds, early stopping | val only (never test or holdout) | | `docs/AI231-FIL50.md` |

So the accuracy numbers are speaker-disjoint and clip-disjoint on wordings the model has seen. The perturbed-gate and holdout-soak results are somewhat optimistic about rooms (seen), and the soak about noise clips (mostly seen).
**The unseen-noise analog that already exists** is the wake word + command replay on the ai231 **test** split: 784 sessions (every PAUSE, STOP and TIME clip plus the first 30 of each other intent), each a real wake-word clip from the wake-word
`test` split, a 0.1-0.4 s gap and the test command, over room tone from the test split's own noise clips (unseen), no reverb, scored through the real streaming pipeline:

| ai231 test replay, 784 sessions (settings of that run: wake-word threshold 0.9, no slot gate) | Correct first trigger | Missed (no period opened) | Latency after end of speech, median / p95 |
| --- | ---: | ---: | ---: |
| Hybrid, wake word polled every 0.05 s | 90.1% (706) | 67 (38) | 0.38 / 0.60 s |
| Wide CTC alone, same | 88.9% (697) | 77 (38) | 0.38 / 0.59 s |

(`out/vcm/hybrid-ctcwide-clsxl/eval/streaming-replay-*-poll005.json`; some of its wake-word clips are noise-mixed copies from the wake-word set.) Use this and the whole-clip tables above as the unseen-data picture of the model's performance;
the holdout soak below adds reverb and a harder real-accent set, with the caveats above.

**Soak test: the holdout behind a wake word, with reverb and noise** (186 commands + 16 out-of-scope clips, 32.6 min, wake word then a 0-1 s gap then the command, 1-8 s of ambient noise between;
tuned settings; [`SOAK-TEST.md`](SOAK-TEST.md); **Raspberry Pi 4 Model B, INT8 ONNX, one thread, 2026-10-03**, next to the same run on a server CPU core):

| | Hybrid, Raspberry Pi 4 | Hybrid, server CPU (same settings) |
| --- | ---: | ---: |
| Correct first trigger (intent and slot) | 81.2% (151/186) | 81.2% (151/186) |
| Wrong-action first triggers (wrong intent or slot) | 9 | 9 |
| Wake word never opened a period | 5 | 5 |
| Out-of-scope clips triggered | 3 of 16 | 3 of 16 |
| Triggers in the ambient gaps | 0 | 0 |
| Latency after end of speech, median / p95 (replay waits for each decode) | 0.38 / 0.65 s | 0.38 / 0.65 s |
| Decode per window, mean / p95 / max | **443 / 764 / 1,575 ms** | 70 / 126 / 184 ms |
| Wake-word gate per decoded window, mean | 38 ms | 5.6 ms |
| Real-time factor, p95 (gate + decode over the 0.25 s stride) | **3.21** | 0.53 |

The Pi gives exactly the same answers as the server (same INT8 ONNX models) but is about 6 times slower per window: at the 0.25 s stride on one core it **does not keep up live**
(p95 0.80 s of work per 0.25 s of audio; a live microphone would queue and drop windows, and the latency above would grow by the decode time). The whole 32.6-minute replay took 20 min 47 s
because only windows inside a wake-word period are decoded. Not yet tried on the Pi: `--threads 4` and a longer stride (`--stride-s 0.5`). Result: `soak/holdout-wake-gap-v1/results/rpi4.md`.

<details><summary>Archived: the server-only soak table (before the Pi run)</summary>

| | Hybrid, CPU INT8 | Hybrid, A100 | CTC only, CPU INT8 |
| --- | ---: | ---: | ---: |
| Correct first trigger (intent and slot) | 81.2% (151/186) | 81.7% (152/186) | 73.7% (137/186) |
| Wrong-action first triggers (wrong intent or slot) | 9 | 9 | 3 |
| Wake word never opened a period | 5 | 5 | 5 |
| Out-of-scope clips triggered | 3 of 16 | 4 of 16 | 3 of 16 |
| Triggers in the ambient gaps | 0 | 0 | 0 |
| Latency after end of speech, median / p95 | 0.38 / 0.65 s | 0.38 / 0.66 s | 0.37 / 0.48 s |
| Decode per window, mean / p95 | 70 / 126 ms | 64 / 102 ms | 70 / 111 ms |
| Real-time factor, p95 (gate + decode over the 0.25 s stride) | 0.53 | 0.45 | 0.47 |

</details>

The soak audio is in `soak/holdout-wake-gap-v1/` (`continuous.wav` and the truth). On a Pi, `make soak-run` fails (`uv run` tries to install the x86 CUDA torch), so call the script with the Pi's venv:
`python scripts/soak_run.py --sessions soak/holdout-wake-gap-v1 --continuous --name rpi4 --wakeword-threshold 0.8 --cls-slot-threshold 0.6 --backend onnx --threads 1`,
and compare with `soak/holdout-wake-gap-v1/results/`. Reverb, noise and the 16 out-of-scope clips come from public data only; wake-word false wakes from ordinary speech are not measured.
