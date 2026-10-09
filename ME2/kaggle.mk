# Kaggle / one-shot experiment workflow. Included from the main Makefile.
#
#   On Kaggle (2 commands, after attaching the me2-code + me2-data datasets):
#       make kaggle-setup      # env + data, idempotent
#       make experiment        # train -> eval (noisy gate) -> export/bench -> e2e -> RESULTS.md -> tarball
#     or just `make kaggle` for both. See docs/KAGGLE.md.
#
#   Locally: `make kaggle-pack-data kaggle-pack-code` (then `make kaggle-upload`).
#   The same `make experiment` also runs locally (uv env, auto GPU/CPU), so it can be
#   tested before burning Kaggle hours: `make experiment-smoke`.

# On Kaggle there is no uv env: use the image's python with PYTHONPATH=src.
# Locally keep going through `uv run`, like every other target.
ifneq ($(wildcard /kaggle/input),)
  KAGGLE := 1
  export PYTHONPATH := src
  EXP_PY ?= python
else
  KAGGLE :=
  EXP_PY ?= $(UV) run python
endif

# Experiment knobs (forwarded to scripts/kaggle/run_experiment.sh as env; see its header).
PRESET ?= quartznet5x3
SEED ?= 0
MINUTES ?= 135
MAX_MINUTES ?= $(MINUTES)
P_RIR ?= 0.7
STAGES ?= train eval export e2e report package
export PRESET SEED MAX_MINUTES P_RIR STAGES

KAGGLE_OWNER ?= YOUR_KAGGLE_USERNAME
KAGGLE_DIST ?= dist/kaggle

.PHONY: kaggle-setup experiment kaggle experiment-smoke kaggle-pack-data kaggle-pack-code kaggle-upload kaggle-data-size

kaggle-setup: ## Init env + link/extract data (Kaggle: pip + dataset; local: uv sync). Idempotent
	PY="$(EXP_PY)" bash scripts/kaggle/setup_env.sh

experiment: ## Full run: train, eval (clean+noisy gate), INT8 export/bench, e2e latency, RESULTS.md, tarball. STAGES="eval report" to run a subset
	PY="$(EXP_PY)" bash scripts/kaggle/run_experiment.sh

kaggle: kaggle-setup ## kaggle-setup + experiment in one command
	$(MAKE) experiment

experiment-smoke: ## ~5-10 min end-to-end dry run on a tiny data slice -- do this before a long run
	SMOKE=1 PY="$(EXP_PY)" bash scripts/kaggle/run_experiment.sh

# ---- local side: build the two Kaggle datasets -------------------------------

kaggle-data-size: ## Count the files/bytes the data dataset would contain (writes nothing)
	$(UV) run python scripts/kaggle/pack_data.py --dry-run

kaggle-pack-data: ## Build dist/kaggle/me2-data/ (me2-data.tar of every manifest-referenced wav). Slow, rare
	$(UV) run python scripts/kaggle/pack_data.py --out $(KAGGLE_DIST)/me2-data --owner $(KAGGLE_OWNER)

kaggle-pack-code: ## Build dist/kaggle/me2-code/ (src, scripts, Makefile, pyproject; incl. uncommitted files). Fast, every code change
	@mkdir -p $(KAGGLE_DIST)/me2-code
	git ls-files --cached --others --exclude-standard -- src scripts Makefile kaggle.mk pyproject.toml \
	  | tar -czf $(KAGGLE_DIST)/me2-code/me2-code.tar.gz -T -
	@printf '{\n  "title": "ME2 code",\n  "id": "$(KAGGLE_OWNER)/me2-code",\n  "licenses": [{"name": "other"}]\n}\n' > $(KAGGLE_DIST)/me2-code/dataset-metadata.json
	@echo "wrote $(KAGGLE_DIST)/me2-code/me2-code.tar.gz ($$(du -h $(KAGGLE_DIST)/me2-code/me2-code.tar.gz | cut -f1))"

kaggle-upload: ## Push the packed datasets with the kaggle CLI (creates on first run, new version after). KAGGLE_OWNER=<user> required
	@test "$(KAGGLE_OWNER)" != "YOUR_KAGGLE_USERNAME" || { echo "set KAGGLE_OWNER=<your kaggle username> (and re-run the pack targets)" >&2; exit 1; }
	@command -v kaggle >/dev/null || { echo "kaggle CLI not found: pip install kaggle (auth: ~/.kaggle/kaggle.json or KAGGLE_API_TOKEN)" >&2; exit 1; }
	@for d in me2-code me2-data; do \
	  test -f $(KAGGLE_DIST)/$$d/dataset-metadata.json || { echo "$(KAGGLE_DIST)/$$d not packed -- run make kaggle-pack-$${d#me2-} first" >&2; continue; }; \
	  if kaggle datasets status $(KAGGLE_OWNER)/$$d >/dev/null 2>&1; then \
	    kaggle datasets version -p $(KAGGLE_DIST)/$$d -m "$$(git rev-parse --short HEAD)" --dir-mode tar; \
	  else \
	    kaggle datasets create -p $(KAGGLE_DIST)/$$d --dir-mode tar; \
	  fi; \
	done
