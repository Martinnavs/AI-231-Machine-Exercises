# Archived checkpoints (non-production)

Two clean-ups removed the non-production models from `master`. Nothing was deleted from history.

- **2026-10-03, binaries.** The `.pt` checkpoints and `.onnx` exports of every run except the production model were untracked.
  - Each file is still in place in the working tree of the machine that trained it (`out/` is git-ignored).
  - A checksummed copy is on the training cluster at
    `/mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/archive/checkpoints-2026-10-03/` (`SHA256SUMS`, same relative paths).
  - `git show 9eb55f9:<path> > <path>` restores one; `9eb55f9` is the last commit that tracked them.
- **2026-10-10, reports and serving targets.** The remaining reports and metadata of those runs were removed from `master`, along with the Make targets that served them:
  - Reports: `eval_report`, `loss_history`, benchmarks, calibration reports and model cards.
  - Targets: `app-pipeline` (old model), `app-pipeline-perchar`, `app-pipeline-quartznet`, `quartznet-eval-report` and `app-pipeline-live`.

  `make app-pipeline` now serves the production hybrid; `app-pipeline-ctcwide` is kept as an alias. To restore the reports, use the last commit that had them, `4812047`:
  - Browse them at <https://github.com/Martinnavs/AI-231-Machine-Exercises/tree/4812047/ME2/out>.
  - Restore one run with `git checkout 4812047 -- ME2/out/vcm/<run>`.
  - The old targets are in `git show 4812047:ME2/Makefile`.
  - The same files are also on the `optionb-*` branches.

History was not rewritten, so the repository size is unchanged.

**Still tracked (production):**
- `out/vcm/hybrid-ctcwide-clsxl/`: the hybrid, see [`CURRENT-MODEL.md`](CURRENT-MODEL.md).
- `out/wakeword-sesame-ambient-rir-45m/`: the wake word it streams with.
- `out/conversions/v2/`: a few data-conversion reports and manifests. These are data, not models.

| Run | What it was | Still referenced by |
|---|---|---|
| `out/vcm/option-d-fil50-ambient-rir-135m` | the earlier production VCM (~1M-parameter MatchboxNet, 135 min with reverb); older docs call it "production" | `session_replay.py` default alignment model; `scripts/dense_pilot_*.py` |
| `out/vcm/option-d-dataset-v2` | the "treatment" VCM trained on the v2 dataset | `make vcmx-serve` default (`VCMX_TRAINED_OUT_DIR`) |
| `out/vcm/option-d-fil50`, `out/vcm/option-d-fil50-ambient-rir-135m-uservoice` | the fil50 accent-rebalance VCM and its user-voice fine-tune | `ACCENT-BALANCE-FIL50-COMMANDS.md`, `MLOPS-PROJECTS.md` |
| `out/vcm/optionb`, `out/vcm/optionb-optionc` | Option B baseline VCMs (`optionc` preset) | `make stream` default model; streaming tests (they skip when the files are absent) |
| `out/vcm/v2s1-heads-A` | the first QuartzNet CTC + heads model (ai231 only, 1 MB) | `CTC-ATTENTION.md`; `BASELINES.md` source [M] |
| `out/wakeword`, `out/wakeword-sesame`, `out/wakeword-fil50` | earlier wake-word DS-CNNs | `make stream-wakeword`, `make wakeword-train` default out-dir, wake-word export tests (they skip when absent) |

**On a fresh clone** the targets and docs in the last column will not find these paths. To run them, do one of:
- Copy the files back from the archive or from history first.
- Point the variable at the hybrid, for example `make vcmx-serve VCMX_TRAINED_OUT_DIR=out/vcm/hybrid-ctcwide-clsxl/ctc-wide`.

Tests that need these files are marked to skip when they are missing.
