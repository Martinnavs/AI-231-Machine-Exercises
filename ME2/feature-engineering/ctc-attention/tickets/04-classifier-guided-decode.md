# 04 — Classifier-guided grammar decode (only if Gate B passes)

**Depends on:** 03 with Gate A and Gate B passed. **Test pass:** required.

## Idea

The beam search over the full grammar is about 60% of per-window time. If the classifier's top-k intents contain
the true one almost always, decode only those intents' sub-grammars: the search space and the time drop, while
CTC still decides acceptance, slots and rejection.

## Do

1. `common/grammar_core.py` / `vcm/optionb`: build a restricted grammar from an intent subset
   (`Grammar.restrict(intents) -> Grammar`, cached per subset, incomplete-prefix set re-derived for the subset;
   a test proves `restrict(all)` decodes identically to the full grammar on fixed posteriors).
2. A backend variant that returns the heads' intent probabilities with the log-posteriors (`OnnxBackend`
   gains an optional second output; ONNX export of the heads behind a flag, off by default; INT8 size still
   <= 1,048,576 B). The streaming runner decodes with `restrict(top_k)` when the backend provides them
   (`--decode-topk K`, default off).
3. Measure on the replay sessions from `vcm/streaming/session_replay.py` (val to choose K in {1, 2, 3, 5},
   test once): correct first trigger, wrong-intent triggers, latency from end of speech, and mean beam-search
   time per decode, next to the baseline mode (`make app-pipeline-live` settings).
4. Pre-registered pass: beam-search time per decode reduced by >= 2x, correct first trigger within 1 pp of the
   baseline, wrong-intent triggers not higher. If it fails, report and stop; leave the flag off.

## Acceptance criteria

- [ ] Equivalence test (full vs `restrict(all)`), unit tests for the restricted grammar and the runner path.
- [ ] Results table in `docs/CTC-ATTENTION.md` with val choice of K and one test run.
- [ ] Default behaviour and `make app-pipeline` / `app-pipeline-live` outputs unchanged.

## Non-goals
Replacing CTC acceptance with classifier acceptance; making the classifier the endpoint detector.

## Execution Log
