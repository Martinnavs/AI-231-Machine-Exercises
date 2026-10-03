# Archived checkpoints (non-production)

On 2026-10-03 the model binaries (`.pt` checkpoints and `.onnx` exports) of every run except the current production model were
untracked from git. They are **not deleted**: each file is still in place in the working tree of the machine that trained it (`out/` is git-ignored),
and a checksummed copy is on the training cluster at
`/mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/archive/checkpoints-2026-10-03/` (`SHA256SUMS`, same relative paths).
They are also still in git history: `git show 9eb55f9:<path> > <path>` restores one (`9eb55f9` is the last commit that tracked them).
History was not rewritten, so the repository size is unchanged.

**Still tracked (production):** `out/vcm/hybrid-ctcwide-clsxl/` (the hybrid, see [`CURRENT-MODEL.md`](CURRENT-MODEL.md)) and
`out/wakeword-sesame-ambient-rir-45m/` (the wake word it streams with). Small reports and metadata of the archived runs stay tracked.

| Run | What it was | Needed by |
|---|---|---|
| `out/vcm/option-d-fil50-ambient-rir-135m` | the earlier production VCM (~1M-parameter MatchboxNet, 135 min with reverb); the README and the process docs call it "production" | `make app-pipeline` default (`APP_PIPELINE_MODEL`); `session_replay.py` default alignment model |
| `out/vcm/option-d-dataset-v2` | the "treatment" VCM trained on the v2 dataset | `make vcmx-serve` default (`VCMX_TRAINED_OUT_DIR`) |
| `out/vcm/option-d-fil50` | the fil50 accent-rebalance VCM | `ACCENT-BALANCE-FIL50-COMMANDS.md` |
| `out/vcm/optionb`, `out/vcm/optionb-optionc` | Option B baseline VCMs (`optionc` preset) | streaming tests (they skip when the files are absent) |
| `out/vcm/v2s1-heads-A` | the first QuartzNet CTC + heads model (ai231 only, 1 MB) | `CTC-ATTENTION.md` streaming commands |
| `out/wakeword`, `out/wakeword-sesame`, `out/wakeword-fil50` | earlier wake-word DS-CNNs | `make stream-wakeword`, wake-word export tests (they skip when absent) |

**On a fresh clone** the Makefile targets and docs that point at these paths will not find them. Copy the files back from the archive (or from history)
before running them, or point the variable at another model (for example `make app-pipeline APP_PIPELINE_MODEL=out/vcm/hybrid-ctcwide-clsxl/ctc-wide`).
Tests that need them are marked to skip when the files are missing.
