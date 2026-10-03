# Decision note: per-character (D2) scoring — opt-in trial, not a production switch

Date: 2026-09-30 · Branch: `optionb-grammar-v2` · Owner decision: **ship as opt-in; try live; report back.**

## Recommendation

**Do not switch production. Run a live opt-in trial** of `score_mode=per_char` (threshold −1.204, no incomplete-prefix
margin gate) alongside the unchanged default, then decide from real use. Rollback is trivial: run the default target.

## What it changes

The VCM decoder's accept/reject confidence. Today (`mean_frame`) it is the winning phrase's beam log-mass divided by the
window's frame count, so blank frames dilute weak evidence. `per_char` divides the same mass by the phrase's character
count instead: duration-invariant, on a different scale (thresholds are about −1, not −0.1). Which intent wins, slots, and
the model/ONNX export are unchanged. Default behavior is bit-identical to before (checked against the original decoder on
111 real-logit decodes). The margin gate stays in the code but is not used in this mode: it was redundant under D2.

## Evidence (production checkpoint `option-d-fil50-ambient-rir-135m`)

Baseline "B" in every comparison = `mean_frame`, threshold −0.1, margin 4.0. Candidate "C2" = `per_char`, −1.204, no margin.

| check | B | C2 |
|---|---|---|
| Exact-correct, all intents, val+test × clean+noisy pooled (13,648 rows) | — | **+217 (+1.59 pp)**; 16 intents better, 3 worse |
| Test false accepts, clean / noisy | 2 / 4 | 0 / 0 |
| Ghost clips falsely accepted (141, never trained on) | 27 | 5 |
| VCM-only streaming over ~3.5 h ambient babble (triggers) | 303 | 17 |
| Streaming recall, INT8 CLI, 821 test clips: correct on first trigger | 720 | 737 |
| ...wrong-intent triggers | 111 | 74 |
| ...PAUSE / STOP / TIME (of 100 / 134 / 107) | 87 / 129 / 86 | 95 / 131 / 92 |

Cost: on whole clips, three intents are net worse (pooled): PAUSE −1.35 pp, MESSAGE −0.95 pp, STOP −0.75 pp. Cause is a
length bias inherent to per-char scoring (phrases ≤ 6 chars are rejected 5.2% of the time when correct, vs. 0.7% for 16+
chars); the same bias is what removes short-phrase false accepts (70% of the false accepts it removes are ≤ 6 chars).
The pre-registered strict per-cell rule failed on one cell (MESSAGE val/clean, 95→92 of 97, −3.09 pp vs. a 3.00 budget):
three clips, two from one speaker, with extra speech around the phrase that production only accepted through dilution.
Judged over all 76 cells that is one boundary crossing, not a pattern. Details: `docs/MLOPS-PROJECTS.md`, Iteration 4.

## What we do not know (why this is a trial)

- **No live-use data.** Everything is recorded/synthetic audio; the val/test sets were used repeatedly for threshold
  selection. Independent evidence is the ghosts, the streaming clips and the ambient soak.
- **The real wakeword→VCM cascade was not exercised.** The wakeword never opened a period on the ambient audio, so that
  gate is vacuous. The ambient VCM-only soak is an upper bound on compound false actions (no wakeword filter).
- **The threshold is not pinned tightly.** Two speaker-disjoint val halves gave −1.07 and −1.34 (a two-way split failed
  the pre-registered stability rule by one false accept). Treat −1.1 … −1.3 as the plausible band.
- **The app runs different settings than the ones measured.** `make app-pipeline` uses threshold −0.075 (measured: −0.1),
  margin 4.0, and `out/wakeword-sesame` (docs promote `wakeword-sesame-ambient-rir-45m`). This trial changes only the VCM
  scoring; the wakeword model is untouched. The −0.075 default is stricter than the −0.1 used in the comparisons.

## Try it

Opt-in, via make (mic → dashboard pipeline; unchanged defaults for the normal target):

```bash
make app-pipeline-perchar                              # per_char, threshold -1.204, no margin
make app-pipeline-perchar APP_PERCHAR_THRESHOLD=-1.1   # more conservative (fewer false triggers)
make app-pipeline-perchar APP_PERCHAR_THRESHOLD=-1.3   # looser (fewer missed commands)
make app-pipeline                                      # production defaults (rollback)
```

Same thing as a plain `uv run` command (JSONL events on stdout, period digest on stderr; no dashboard forwarding):

```bash
uv run python -m me2_voicegen.vcm.streaming \
  --model out/vcm/option-d-fil50-ambient-rir-135m --backend onnx --onnx-variant int8 \
  --grammar optionb --score-mode per_char --threshold=-1.204 \
  --beam-width 50 --gate wakeword --policy single_period --gate-period 3 \
  --wakeword-model out/wakeword-sesame --wakeword-backend onnx \
  --log-periods --source mic
```

The startup banner must show `score_mode: per_char`, `resolved threshold: -1.204`, `required_command_margin: None`. To keep a
session log while still seeing it: `make app-pipeline-perchar 2> >(tee perchar_session.log >&2)`.

## What to report back

For each session, roughly: (1) commands you said that **did not act** (intent, how you said it, any extra words),
(2) actions that fired when you **did not** mean a command (what was said or playing), (3) session length and the
threshold used, (4) any difference from your usual experience on PAUSE / STOP / MESSAGE specifically. The
`period: ACCEPT|REJECT … confidence` lines in the stderr log carry the per-period confidence for tuning.

## Decision rule for after the trial (set now, before seeing results)

- **Adopt** (make per_char the default via `APP_PIPELINE_*` defaults and `VCMX_SERVE_*`) if you see no more missed commands
  than usual and fewer false triggers.
- **Tune** the threshold within −1.1 … −1.3 if only one failure mode shows up.
- **Revert** (no code change needed) if PAUSE / STOP misses are noticeably worse in real use.
