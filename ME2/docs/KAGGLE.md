# Running the experiment on Kaggle

Two commands on Kaggle, after a one-time upload from here.

```bash
make kaggle-setup   # env + data (idempotent)
make experiment     # train -> eval (clean + noisy gate) -> INT8 export/bench -> e2e latency -> RESULTS.md -> tarball
# or: make kaggle   # both
```

`make experiment` defaults to the QuartzNet recipe in `docs/QUARTZNET-STUDENT.md` ("Reproducing"): preset `quartznet5x3`, seed 0, 135 min,
`--p-rir 0.7`, train on `optionb-v3-vcmx-fil50-ambient`, gate on `optionb-v3-vcmx-fil50` (`optionb` grammar, beam 50, noisy seed 0).

## One-time / per-change, on this machine

```bash
make kaggle-data-size                      # sanity: ~3.8 GB, file count
make kaggle-pack-data KAGGLE_OWNER=<user>  # dist/kaggle/me2-data/me2-data.tar  (slow; only when data changes)
make kaggle-pack-code KAGGLE_OWNER=<user>  # dist/kaggle/me2-code/me2-code.tar.gz (fast; after every code change)
make kaggle-upload    KAGGLE_OWNER=<user>  # needs `pip install kaggle` + auth; or upload the two dirs by hand in the UI
```

The data tar holds only the wavs the two manifests reference, at their repo-relative paths, so the relative manifest paths resolve unchanged.

## In the Kaggle notebook

Settings: Accelerator = GPU (T4 x2 or P100), Internet = on (only needed if the image's torch has to be swapped, see below), add both datasets as inputs.

```python
!mkdir -p /kaggle/working/ME2 && tar -xzf /kaggle/input/me2-code/me2-code.tar.gz -C /kaggle/working/ME2
%cd /kaggle/working/ME2
!make kaggle-setup && make experiment
```

(Dataset mount paths can be nested, e.g. `/kaggle/input/datasets/<user>/me2-code/`; `ls /kaggle/input` and adjust the first line. The setup script finds the data dataset by itself.)

Results land in `out/vcm/<run>/` (+ `RESULTS.md`), and `/kaggle/working/<run>.tar.gz` is the notebook output you download.

## Knobs

```bash
make experiment-smoke                          # ~minutes, tiny data slice: run this first to catch setup problems
make experiment PRESET=optiond SEED=1          # other preset/seed (run name: <preset>-seed<N>-<minutes>m)
make experiment MINUTES=180                    # training wall-clock budget (alias of MAX_MINUTES); also raise MAX_EPOCHS if epochs bind first
make experiment MINUTES=60 P_RIR=0.3
make experiment STAGES="eval export e2e report package"   # re-run later stages on an existing checkpoint
make experiment FORCE=1                        # ignore the per-stage done markers
```

Others (env vars, see the header of `scripts/kaggle/run_experiment.sh`): `RUN_NAME`, `BATCH_SIZE`, `NUM_WORKERS`, `DEVICE`, `TRAIN_MANIFEST`, `EVAL_MANIFEST`, `GRAMMAR`, `BEAM`, `NOISY_SEED`.

## Caveats

- **Resume is per stage, not mid-training.** If a session dies during `train`, it restarts that stage from scratch (the trainer has no resume). Finished stages are skipped on re-run if `out/vcm/<run>` survived, which on Kaggle means saving a version of the notebook, since only `/kaggle/working` persists.
- **Training is wall-clock capped** (`MAX_MINUTES`), not epoch capped, so a slower Kaggle GPU completes fewer epochs than the reference run (61-67 epochs on the lab GPUs). Compare numbers with that in mind, or raise `MAX_MINUTES`.
- **Eval time is unmeasured on Kaggle.** The beam-search decode is CPU-bound and Kaggle has 4 vCPUs; the lab runs took 45-70 min on a big server. Budget the 12 h session accordingly.
- **Torch version.** The augmenter needs `torchaudio.prototype.functional.simulate_rir_ism`. If the image's torchaudio doesn't have it, setup installs the repo's pinned `torch==2.3.1` (cu121) from the PyTorch index, which needs Internet on.
- The data includes ESC-50 noise (CC-BY-NC-SA-4.0): keep the datasets private / non-commercial (`docs/VCM-CONTRACT.md` section 9).
