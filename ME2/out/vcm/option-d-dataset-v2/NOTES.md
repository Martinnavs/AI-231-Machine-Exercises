# Run notes: option-d-dataset-v2

**What this is:** preset `optiond`, trained on the **treatment** manifest from the
`optionb-v3-vcmx` feature — refreshed Option B (upstream `b9d86ea`, live grammar)
merged with in-grammar rows from VCM Dataset B, plus ESC-50 silence re-split by
fold. See `.scratch/optionb-v3-vcmx/tickets/00-RECAP.md` for the full design and
`docs/OPTIONB-GRAMMAR-CONTRACT.md` for the current grammar.

**Not yet run, for comparison:**
- **Control** — same preset/seed/budget, manifest
  `out/conversions/v2/optionb-v3-vcmx-control/manifest.csv` (refreshed Option B
  only, no VCM Dataset B rows). Val/test are byte-identical to this run's, so
  any accuracy delta isolates the effect of adding Dataset B.
- **Old baseline** — `out/vcm/optionb-optiond` (preset `optiond`, stale grammar,
  old Option B data, 71 epochs, no time cap). **Not directly comparable**: its
  grammar has different phrasings for 7 intents (PLAY_MUSIC, PAUSE, STOP,
  LIGHT_OFF, BRIGHTNESS, COLOR, CALL — see the grammar table in the ticket), so
  its `eval_report.md` numbers are scored against a different label space than
  this run's.

## Exact commands

```
OPTIONB_MANIFEST=out/conversions/v2/optionb-v3-vcmx/manifest.csv \
OPTIONB_OUT_DIR=out/vcm/option-d-dataset-v2 \
VCM_PRESET=optiond VCM_DEVICE=cuda:7 VCM_MAX_MINUTES=60 VCM_SEED=0 \
make optionb-train

OPTIONB_MANIFEST=out/conversions/v2/optionb-v3-vcmx/manifest.csv \
OPTIONB_OUT_DIR=out/vcm/option-d-dataset-v2 \
VCM_PRESET=optiond VCM_DEVICE=cuda:7 \
make optionb-eval
```

Started: 2026-09-24, GPU 7 (idle at launch — 4 MiB used, 0% util at the time).
Time budget: 60 minutes wall-clock for training (`vcm.train`'s own deadline
check), not a fixed epoch count — see `loss_history.json` for epochs actually
completed within budget.

## Dataset this run trains on (treatment manifest)

| bucket | train | val | test |
|---|---|---|---|
| target_commands | 15,937 | 2,013 | 1,927 |
| babble | 2,394 | 409 | 255 |
| silence | 1,643 | 329 | 324 |

Includes 1,891 in-grammar VCM Dataset B commands and 2,497 Dataset B UNKNOWN
rows in train (per the ticket's Phase 2 log), plus all 1,633 ESC-50 silence
chunks split by fold (no per-recording leakage — verified). See
`out/conversions/v2/optionb-v3-vcmx/report.md` for the full breakdown and the
grammar-filter drop-reason counts.

## Results (completed 2026-09-24)

- **Training:** 90 epochs in the 3,601s (60 min) budget — `deadline_hit=True`,
  as intended. Best val_loss **0.4321** at epoch 88 (`epoch0_val_loss` 89.74).
  `nan_or_inf_seen=True` (387 skipped batches) — **same as the old baseline**
  (`out/vcm/optionb-optiond` also has `nan_or_inf_seen=True`), so this is
  normal for this training loop, not a regression caused by the new data.
  Full curve: `metadata/loss_history.json`.
- **Eval** (`metadata/eval_report.md`, beam width 50, threshold chosen on val
  by Youden's J, never on test): chosen threshold **-0.1** (val
  target_accept_rate 0.964, val false_accept_rate 0.018).
  - **Test, `target_commands`:** 1927 clips, 1889 accepted (0.980), 1887
    exact-intent-correct (**0.979 exact accuracy**).
  - **False-accept rate:** babble 3/255 (0.012), silence 5/324 (0.015) — at
    the chosen threshold only (see the val sweep table for how it degrades
    at looser thresholds).
  - **Slot accuracy:** 1057/1057 intent-correct slot-bearing clips (1.000)
    got every slot value right, out of 1068 slot-bearing clips.
  - **Speaker-group gap:** filipino_reference 0.939 exact accuracy (n=180,
    single held-out speaker `s100`) vs. foreign_reference 0.989 (n=1618) —
    gap 0.050. `n_unclassified=129`: `classify_speaker_group` only
    classifies `source_dataset == "optionb"` rows, so the VCM Dataset B
    command rows in test (`vcm_balanced`) fall into "unclassified" rather
    than either group — this is a known gap in that function, not a data
    problem (per-intent confusion counts above are unaffected).
  - **NEXT** has an unusually large test count (119, vs. ~50-70 for most
    intents) — this is the Multi-Sensor "next song" contribution from VCM
    Dataset B; see the treatment `report.md`'s drop-reason breakdown for
    context (it dominates the 1,891 kept command rows).
  - No other intent stands out — worst-case per-intent reject rate is
    TIME (7/60, 0.117) and TIMER (4/176, 0.023), both within the range the
    old baseline also showed for rejected-but-real commands.

## Still needed for the full comparison this feature was built for

1. **Control run** (`out/conversions/v2/optionb-v3-vcmx-control/manifest.csv`,
   same preset/seed/budget) — isolates whether adding Dataset B actually
   helped, since its val/test is byte-identical to this run's.
2. **In-domain eval view**: run `vcm.evaluate` against
   `out/conversions/v2/optionb-v3/manifest.csv` (Option-B-only) using this
   checkpoint, to confirm the new data didn't hurt Option-B-only behavior —
   Dataset B's val/test now heavily outweighs Option B's own probes in this
   run's numbers above.
3. Tech-lead review of the `optionb-v3-vcmx` feature branch (grammar refresh
   + merge builder) before any of this gets committed.
