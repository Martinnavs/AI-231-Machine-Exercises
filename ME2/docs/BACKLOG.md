# Backlog

Deferred work items that are real, scoped, and worth tracking, but not
appropriate to build speculatively ahead of a dependency that doesn't
exist yet. Unlike `.scratch/` (gitignored, ephemeral per-feature dev-flow
tickets), this file is committed so the item survives on any machine
where this repo is cloned, not just the one it was written on.

## Mode-across-a-bounded-listening-period `AcceptancePolicy`

**Area:** `me2_voicegen.vcm.streaming`
**Status:** implemented (2026-09-21, `mode-period-gate` feature) — see
"Shipped" below; the item text under it is the original deferred
context, kept for history
**Full technical context (as shipped):** `docs/STREAMING-CONTRACT.md`
section 4 ("The mode-across-a-bounded-listening-period policy") for the
policy and section 6 (the `ListeningGate` seam) for the gate — read
those; the text below is the pre-implementation item.

### Shipped (2026-09-21)

Built as the `mode-period-gate` feature (confirmed plan
`ME2/.scratch/mode-period-gate/plan.md`; requirements source
`/home/bertie/.opencode/plan/mode-period-gate-SPEC.md`; tech-lead review
approved, `ME2/.scratch/mode-period-gate/review-round-1.md`):

- `ModePeriodPolicy` in `src/me2_voicegen/vcm/streaming/policy.py`,
  registered as `--policy mode_period` in `POLICY_REGISTRY`: collects
  every observation across a bounded listening period and flushes one
  consolidated accept/reject from the mode (most frequent
  `(intent, slots)` decode), the class's mean confidence vs. the
  operating threshold, deterministic tie-breaks.
- The component this item was blocked on, shipped as the abstract
  `ListeningGate` seam plus concrete `SpacebarGate` in the new
  `src/me2_voicegen/vcm/streaming/gate.py` (`--gate spacebar` /
  `--gate-period <s>`); a future wake-word gate satisfies the same
  protocol with zero policy/runner changes.
- Both acceptance criteria below are pinned as named tests:
  `tests/test_vcm_streaming_policy.py::test_backlog_ac1_ten_identical_decodes_collapse_to_single_accept_at_flush`
  and `::test_backlog_ac2_intermittent_noise_in_mostly_none_period_never_triggers`,
  plus the cross-cutting runner+gate+policy runs in
  `tests/test_vcm_streaming_integration.py`.
- **Approved deviation 1 supersedes this item's "zero diff to
  `runner.py`" acceptance criterion:** the shipped runner diff is the
  ~6-line `StreamingRunner._evaluate_window` change (pass `waveform` on
  the observation; read the payload's intent/slots/text/confidence from
  `decision.result` when the policy set it, else the window's own
  result) — a consolidated event must carry a *previous* window's
  decode, and the gate needs the audio, and both require payload-side
  handling. `decoder.py`/`debounce.py` stayed byte-identical and
  `StreamingRunner.__init__` is unchanged, so the seam's purpose
  (policy swappability without a runner *rewrite*) stands.
- **Post-approval extension (2026-09-21):** the `--log-periods` per-period
  stderr digest (gate open/reopened/closed lines plus the period's
  consolidated result, incl. rejected periods; stdout JSONL untouched) —
  design note `ME2/.scratch/mode-period-gate/log-periods-extension.md`.

### Problem (observed directly on real hardware, not hypothetical)

1. **Lingering duplicate triggers on a correctly-spoken phrase.** At the
   shipped defaults (`window_s=2.5`, `stride_s=0.25`, `refractory_s=1.5`),
   a spoken phrase decodes correctly across roughly ten consecutive
   windows as it moves through the sliding window, and the debounce
   window can expire while the phrase is still present — producing a
   second trigger on what is mechanically the same utterance. Observed:
   two `TIME` triggers exactly `1.5s` apart (= `refractory_s` to the
   decimal) from one lingering decode, not two separate utterances.
2. **False positives during fast speech/walkthroughs.** Consistent with
   the confidence/padding-sensitivity finding elsewhere in this repo:
   confidence is a mean per-frame log-prob over the whole fixed window,
   so a short or partially-formed phrase is diluted the same way ambient
   noise is, and can still land just inside the operating threshold.

### Fix

Once a wake-word gate exists and defines a bounded "listening period"
(see `docs/STREAMING-CONTRACT.md` section 6's product-level-integration
note — that integration is itself undecided and not this ticket's
concern), implement
a new `AcceptancePolicy` in `src/me2_voicegen/vcm/streaming/policy.py`
that consumes every `WindowObservation` across that period and emits one
consolidated `TriggerEvent` based on the **mode** (most frequent decode)
across observations, instead of accepting the first window that
individually clears the threshold.

### Why this is cheap when it's time to build it

`AcceptancePolicy.observe`/`reset` were made stateful specifically to
support this case (see contract doc) — `ThresholdPolicy` doesn't use the
history, but the protocol already does. Implementing this is:
- one new class in `policy.py`,
- one new `POLICY_REGISTRY` entry in `src/me2_voicegen/vcm/streaming/config.py`,
- **no changes to `runner.py`, `decoder.py`, or `Debouncer`** — that's the
  seam's whole purpose, and it's already proven swappable by
  `tests/test_vcm_streaming_runner.py`'s policy-swappability test.
  (**Superseded as built** — see "Shipped": approved deviation 1
  allowed the ~6-line `runner._evaluate_window` diff; `decoder.py` and
  `Debouncer` did stay untouched.)

### Acceptance criteria (for whoever picks this up)

- New policy class registered in `POLICY_REGISTRY`, selectable via
  `--policy`. (Met: `--policy mode_period`.)
- Zero diff to `runner.py`/`decoder.py`/`debounce.py` to support it.
  (**Superseded as built by approved deviation 1** — see "Shipped":
  ~6-line `runner._evaluate_window` diff; `decoder.py`/`debounce.py`
  did stay zero-diff.)
- A test proving it resolves both observed failure modes: a synthetic
  multi-window sequence with the same correct decode repeated N times
  (lingering) collapses to one event; a synthetic sequence with
  intermittent low-confidence noise mixed into mostly-blank windows does
  not falsely trigger. (Met: the two `test_backlog_ac*` tests named in
  "Shipped".)

### Blocked by (resolved)

The wake-word/gate component did not exist yet when this was deferred.
It now exists as the abstract `ListeningGate` seam plus the
`SpacebarGate` stand-in (shipped 2026-09-21;
`docs/STREAMING-CONTRACT.md` section 6) — blocker resolved. A real wake-word
detector is now a future *gate implementation*, not a prerequisite for
this policy.
