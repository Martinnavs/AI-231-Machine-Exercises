# Streaming Contract

Shared contract for the `me2_voicegen.vcm.streaming` feature (Tasks
01-04: ring buffer/debounce/policy seam, audio sources, runner loop +
inference backend, CLI). Every task in this feature reads this doc
instead of having the shapes below re-pasted into its prompt. Source of
truth for each section is the module/file named; if this doc and that
module ever disagree, the module wins and this doc is stale and needs
fixing.

Task 01 (this doc's author) owns `RingBuffer`, `Debouncer`, and the
`AcceptancePolicy` seam (`ThresholdPolicy`), plus this document. Tasks
02/03/04 own their own sections below and update them as they land --
they do not need this section re-explained to them, just pointed here.

## 1. `AudioSource` protocol (owned by Task 02)

As actually implemented in `vcm/streaming/sources.py` (this section was
drafted with a `read(n_samples)` shape before Task 02 landed; the shape
below is the real, shipped contract -- update call sites to it, not the
other way around):

```python
class AudioSource(Protocol):
    is_realtime: bool

    def blocks(self) -> Iterator[np.ndarray]:
        """Yields float32 mono blocks at common.features.SAMPLE_RATE.
        Blocks are exactly the source's configured block_samples long
        except possibly the final block (file sources only -- the
        generator simply ends at end-of-stream; a realtime source's
        generator never ends under normal operation, it just blocks
        until audio is available)."""
        ...

    def close(self) -> None: ...

    def __enter__(self) -> "AudioSource": ...
    def __exit__(self, *exc_info) -> None: ...
```

Three implementations ship: `RawPcmStreamSource(stream, block_samples)`
(wraps any `.read(n) -> bytes` object -- a subprocess stdout pipe or an
in-memory buffer -- of raw S16_LE mono 16kHz PCM), `WavFileSource(path,
block_samples, realtime=False)` (`is_realtime=False`, loads/downmixes/
resamples via `torchaudio`, same idiom as
`vcm.evaluate.evaluate_slot_eval_set`), and `MicrophoneSource`/
`open_microphone_source(command, block_samples)` (`is_realtime=True`,
spawns `command` -- always a parsed argv list, `shell=False` -- and
wraps its stdout in a `RawPcmStreamSource`; raises
`MicrophoneUnavailableError` if the process is missing or exits on its
own before/without producing audio, but never on a returncode produced
by our *own* `close()`-time `terminate()`).

`is_realtime` selects between two runner-loop modes, because a live
microphone and a pre-recorded file need fundamentally different
backpressure behavior:

- **`is_realtime=True` (mic):** capture happens on its own thread,
  continuously, regardless of whether inference keeps up. If the
  inference loop falls behind, it **skips-and-counts** -- it drains the
  capture queue/buffer to the most recent audio and records how many
  chunks it dropped, rather than blocking capture (an audio driver whose
  callback blocks on inference risks ALSA buffer underruns -- see
  `docs/raw_requirements/streaming_approach.md`'s "Audio Buffer
  Underrun" pitfall). Real time keeps moving whether or not the model
  keeps up; the buffer always reflects "now," not "whenever inference
  last got around to it."
- **`is_realtime=False` (file):** synchronous lockstep, no drops. The
  runner reads exactly as fast as it can process, and every sample in
  the file is guaranteed to reach the ring buffer and be evaluated at
  least once at some stride offset -- reproducibility for tests/eval
  requires this; a file replay that silently drops chunks under load
  would make results depend on host CPU speed.

## 2. `StreamingConfig` (owned by Task 03/04, fields listed here as they are introduced)

A single config object threading window/stride/refractory/threshold and
runner-loop knobs through the CLI, the runner loop, and the
`SlidingWindowPipeline`-equivalent streaming stack, so no task invents a
second, slightly different way to plumb the same handful of numbers.

Landed by Task 03 (`vcm.streaming.config`), frozen dataclass, precedence
`dataclass defaults < JSON config (StreamingConfig.from_json) < explicit CLI
flags (StreamingConfig.merge)`:

- `model: str = "optionc"` -- `MODEL_REGISTRY` name, run-dir path, or direct
  checkpoint/`.onnx` file path (`resolve_model`).
- `backend: str = "onnx"` (`"onnx"` | `"torch"`)
- `onnx_variant: str = "fp32"` (`"fp32"` | `"int8"`)
- `ort_threads: int = 1`
- `policy: str = "threshold"` -- `POLICY_REGISTRY` name (`resolve_policy`).
- `grammar: str = "optionb"` -- `vcm.evaluate.GRAMMAR_REGISTRY` key, reused
  rather than a second registry (`resolve_grammar`).
- `threshold: Optional[float] = None` -- resolved from the resolved model's
  run dir's `metadata/eval_report.json` (`resolve_threshold`) when `None`;
  an explicit value always wins.
- `window_s: float = 2.5`, `stride_s: float = 0.25`,
  `refractory_s: float = 1.5`, `beam_width: int = 25`, `block_s: float =
  0.064`
- `device: str = "cpu"`
- `source: str = "mic"` (`"mic"` | a wav path)
- `realtime: bool = True`
- `mic_command: Optional[str] = None`
- `listen_for: Optional[float] = None`
- `log_all_windows: bool = False`
- `gate: str = "none"` (`"none"` | `"spacebar"`) -- `GATE_REGISTRY` name
  (`resolve_gate`); which listening gate bounds the policy's listening
  periods (section 6).
- `gate_period_s: float = 5.0` -- listening-period length in seconds, the
  single source passed to both the gate's auto-close and the
  `mode_period` policy's flush boundary, so the two close boundaries
  always agree.
- `log_periods: bool = False` -- per-period digest on stderr (gate
  open/close lines plus the period's consolidated result); the JSONL on
  stdout is unchanged (section 6).
- `required_command_margin: Optional[float] = None` -- incomplete-prefix
  rejection gate margin (raw unnormalized beam log-mass units;
  docs/INCOMPLETE-GRAMMAR-REJECTION.md, Step 3). `None` (the default)
  leaves the gate disabled and window-decode acceptance is exactly the
  baseline; when set it is forwarded verbatim to every window's `decode()`
  call (semantics: VCM-CONTRACT.md section 7, "Margin gate"). The startup
  banner reports the resolved value; the JSONL schema (section 5) is
  unchanged -- a gate rejection surfaces only as `intent: null` /
  `confidence: null`.
- `score_mode: str = "mean_frame"` -- what `confidence` (and so the policy
  threshold) measures; one of `"mean_frame"` (beam log mass / window frames,
  the historical behavior) or `"per_char"` (the same raw mass / length of the
  winning phrase: duration-invariant, on a different scale -- thresholds are
  around -1, not -0.1, and are NOT interchangeable between modes). Forwarded
  verbatim to every window's `decode()`; winner selection and
  `out_of_grammar_gap` are identical in both modes. CLI: `--score-mode`;
  `make vcmx-serve` var `VCMX_SERVE_SCORE_MODE`. Opt-in and NOT the production
  setting: `per_char` needs its own calibrated threshold and is gated on the
  validation plan in `.scratch/dense-d2-loose-impl/PLAN.md`. The banner
  reports the resolved value; the JSONL schema (section 5) is unchanged.

`MODEL_REGISTRY = {"optionc": out/vcm/optionb-optionc, "default":
out/vcm/optionb}` (`optionc` is the config default -- it beats `default` on
every eval metric, see this ticket's Execution Log). `POLICY_REGISTRY =
{"threshold": ThresholdPolicy, "mode_period": ModePeriodPolicy,
"single_period": SinglePeriodPolicy}`. `single_period` requests one exact
3.0 s absolute sample interval from the runner at period close, then receives
that interval's one decode for ordinary threshold acceptance. It is separate
from `mode_period`, which decodes every stride and votes; the runner retains
period-plus-window audio so close polling/catch-up cannot contaminate its interval.
`resolve_policy(name, threshold, *, gate=None, period_s=5.0,
on_period_event=None)` is backward-compatible (resolving `threshold` is
unchanged), but `mode_period` and `single_period` require a non-`None` `gate` -- resolving
it without one raises an actionable `SystemExit`, and the CLI enforces
the same rule as a hard cross-validation error (section 6).
`on_period_event` (a `policy.PeriodEventCallback`:
`(event, samples_seen, decision) -> None`) is forwarded to
both period policies -- the CLI's `--log-periods` digest wires its stderr
printer there (section 6); `ThresholdPolicy` never receives it.

**Security note (both `--model` code paths are trust boundaries):**
`resolve_model`'s result is later fed to either `torch.load` (pickle RCE) or
ONNX Runtime's model parser -- neither is safe on an untrusted file. This is
deliberately NOT run through `fetch_dataset.resolve_under()`-style path
containment (see `config.py`'s and `backends.py`'s module docstrings for
why); only point `--model` at a checkpoint/`.onnx` file you produced
yourself or otherwise trust.

**Security note (`--config` JSON is a trust boundary too, not inert data):**
`StreamingConfig.from_json` validates field names AND, as of the R3-1
hardening pass, per-field types/choices (mirroring `__main__`'s argparse
`choices=`/`type=` constraints) -- but it does not and cannot judge whether
the *values* themselves are safe. A JSON config file can set `model` (the
trust boundary above) or `mic_command` (the argv `MicrophoneSource` spawns
via `subprocess.Popen`, `shell=False` -- see `vcm.streaming.sources`) just
as freely as the equivalent CLI flags can. Treat a `--config` file as
exactly as trusted as the command line itself, never as passive/inert
configuration data safe to copy from an untrusted source.

**Security note (`weights_only=True` is a mitigation, not a guarantee):**
`load_checkpoint`'s `weights_only=True` path defends against naive/accidental
malicious pickles, but this project's pinned `torch==2.3.1` is within the
affected range of CVE-2025-32434, a known bypass of `weights_only=True` on
torch <= 2.5.1 (fixed in 2.6.0). Upgrading torch is out of scope for this
feature (a repo-wide pinned CUDA stack) and is tracked as a separate
backlog concern. Additionally, `load_checkpoint`'s `weights_only=True`
failure path does NOT automatically retry with `weights_only=False` -- a
checkpoint failing the allow-list is itself the attack signature this
mitigation exists to catch, not a benign edge case, so it is refused
outright unless the caller also explicitly passes `allow_unsafe_load=True`.
`TorchBackend` always calls with `allow_unsafe_load=False`.

## 3. `InferenceBackend` adapter interface (owned by Task 03)

Abstracts "torch checkpoint" vs. "ONNX export" behind one call shape so
the runner loop and CLI don't care which backend produced a window's
`logp`:

```python
class InferenceBackend(Protocol):
    def logp_for_waveform(self, waveform: np.ndarray) -> np.ndarray:
        """(samples,) float32 waveform at SAMPLE_RATE -> (T, alphabet_size)
        log-posterior array -- the same contract as
        vcm.pipeline.logp_for_waveform / docs/VCM-CONTRACT.md section 7."""
        ...
```

Two implementations ship (`vcm/streaming/backends.py`):

- `OnnxBackend(model_path, ort_threads=1)`: builds an
  `onnxruntime.InferenceSession` per `benchmark._make_session`'s pattern
  (`SessionOptions`, `intra_op_num_threads=ort_threads`,
  `inter_op_num_threads=1`, `providers=["CPUExecutionProvider"]` -- CPU
  only, the installed ORT wheel offers no other real provider).
  `logp_for_waveform` runs the shared `LogMelFeatureExtractor` on the
  waveform, feeds `{"features": (1, 40, T) float32}` to `session.run`
  (`export_onnx.onnx_vs_pytorch_logits`'s call shape), and applies
  **log-softmax in numpy** to the returned `(1, T, 29)` logits -- ONNX
  emits logits, not log-probs, unlike the torch side which already does
  `log_softmax` internally. Verified within 1e-2 max-abs-diff of
  `TorchBackend` on the checked-in optionc fp32 export
  (`tests/test_vcm_streaming_backends.py::
  test_onnx_backend_matches_torch_backend_on_real_optionc_artifacts`,
  `@pytest.mark.slow`). Preset metadata for a startup banner is NOT carried
  by the `.onnx` file itself -- read it from the run dir's
  `eval_report.json`/`checkpoint_meta` instead.
- `TorchBackend(checkpoint_path, device="cpu")`: wraps
  `vcm.pipeline.load_checkpoint(..., weights_only=True)` +
  `vcm.pipeline.logp_for_waveform`. `weights_only=True` is confirmed to
  load both real checked-in checkpoints (non-tensor metadata included) as
  of torch 2.3.1; `load_checkpoint` falls back to `weights_only=False` with
  an explicit stderr warning if a future checkpoint's metadata trips the
  allow-list, rather than silently regressing.

Cold start is non-trivial for **either** backend -- see section 6, "the
ONNX backend does NOT avoid this."

**Security note:** both backends parse untrusted/arbitrary-file-format
input at construction (`torch.load` pickle deserialization / ONNX Runtime's
model parser) -- see `backends.py`'s and `config.py`'s module docstrings
for the full trust-boundary note and why path-containment is not the right
mitigation here.

## 4. The `AcceptancePolicy` seam (owned by Task 01)

### Protocol signatures

```python
# vcm/streaming/policy.py
@dataclass(frozen=True)
class WindowObservation:
    window_index: int
    samples_seen: int
    result: DecodeResult        # decoded at threshold=-inf
    waveform: Optional[np.ndarray] = None   # NEW: the window's audio
    # (ring-buffer snapshot), forwarded to the listening gate;
    # None tolerated (tests/fakes)

@dataclass(frozen=True)
class PolicyDecision:
    accept: bool
    reason: str                 # surfaced by --log-all-windows
    result: Optional[DecodeResult] = None   # NEW: authoritative override
    # for the emitted event; None = use the observed window's own
    # result (ThresholdPolicy and all pre-existing behavior unchanged)

class AcceptancePolicy(Protocol):
    def observe(self, obs: WindowObservation) -> PolicyDecision: ...
    def reset(self) -> None: ...

class ThresholdPolicy:          # shipped impl 1: per-window threshold gate
    """accept iff result.intent is not None and result.confidence >= threshold"""

class ModePeriodPolicy:         # shipped impl 2: one decision per listening period
    def __init__(self, threshold: float, *, gate: ListeningGate,
                 period_s: float) -> None: ...
```

`WindowObservation`/`PolicyDecision` are frozen dataclasses. `observe`/
`reset` are stateful methods, not a pure function of one `DecodeResult`,
because named future cases (multi-window smoothing, N-consecutive
voting, active-region scoring) all need rolling history across
observations; this costs `ThresholdPolicy` nothing since it ignores
history entirely (it's a pure function of the one `WindowObservation`
it's given, it just happens to satisfy the stateful protocol). Likewise
`PolicyDecision` is a dataclass, not a bare `bool`, so a future policy
that needs to influence emission (e.g. a per-intent refractory override)
can gain a field without changing the protocol signature -- which is
exactly what the additive `result` field below did.

Two fields landed additively with `ModePeriodPolicy` (the
`mode-period-gate` feature, 2026-09-21), both defaulted, both dataclasses
still frozen:

- `WindowObservation.waveform` carries the window's audio (the ring-
  buffer snapshot the runner already had for the decode) so a listening
  gate can see the same audio the model decoded: `SpacebarGate` ignores
  it, a future wake-word gate consumes it (section 6).
- `PolicyDecision.result` is an authoritative result override: the
  runner emits `decision.result`'s intent/slots/text/confidence when it
  is not `None`, else the window's own result. That is how a period-end
  flush carries a *previous* window's decode into the event it authorizes
  (section 5 documents the payload side).

`ThresholdPolicy` is behaviorally unchanged by both: it never sets
`result` and never reads `waveform`.

### Fixed pipeline order

```
logp -> decode(threshold=-inf) -> DecodeResult -> AcceptancePolicy.observe(...) -> PolicyDecision.accept -> Debouncer (refractory gate) -> TriggerEvent
```

Every window is decoded once, at the most permissive threshold; the
`AcceptancePolicy` decides whether that decode is real evidence; only
*then* does the `Debouncer` decide whether enough time has passed since
the last emission to actually fire a `TriggerEvent`. This order is
fixed -- a future policy or backend must not reorder it (e.g. gating the
decoder call itself on debounce state), because the two questions below
are deliberately independent. `ModePeriodPolicy` upholds it: it polls
its `ListeningGate` (section 6) inside `observe`, at the same stride
boundary the runner already evaluates -- the gate sits between the
observation stream and the policy's decision, never between the decoder
and the policy -- so the chain is unchanged for both shipped policies.

### Evidence vs. emission-rate: why these are two separate seams

`AcceptancePolicy` answers "is this window real evidence?" (a function
of the model's output for this window, possibly with rolling history).
`Debouncer` answers "has enough time passed since the last emission?" (a
function of elapsed samples only, independent of *why* the previous
emission fired). These are different questions, and keeping them as two
separate, independently swappable seams means a future history-based
policy (e.g. N-consecutive-window voting) composes cleanly with
debounce-after: the policy can look back across windows to decide
"real", and the debouncer still independently rate-limits how often a
"real" verdict is allowed to become a `TriggerEvent`, without either one
needing to know the other's internals.

### The `-inf` decode rationale

Every window is decoded via `infer_waveform(..., threshold=NEG_INF)` (or
equivalent) *before* the `AcceptancePolicy` ever sees it -- the policy,
not the decoder, is what applies the operating threshold. This directly
reuses an established precedent, not a new invention:
`vcm.evaluate.decode_split` already decodes every dataset row at
`vcm.evaluate.NEG_INF_THRESHOLD` to capture each row's best-reachable
intent/confidence independent of any operating threshold, so a later
threshold sweep is pure arithmetic over cached decodes rather than
re-running the model per candidate threshold
(`vcm.evaluate.sweep_thresholds`). The streaming seam applies the same
idea for the same reason: decode once, threshold-gate via the swappable
`AcceptancePolicy` afterward, so a different policy (or the
`--log-all-windows` diagnostic below) can see what the decoder actually
found on every window, not just the windows that happened to already
clear one hardcoded threshold. `ThresholdPolicy.observe` then reduces to
exactly `vcm.evaluate._accepted`'s body: `intent is not None and
confidence >= threshold`.

The motivating case for not hardcoding the threshold into the decode
call itself: `vcm.evaluate`'s own confidence/padding-sensitivity finding
(the model's confidence score is measurably sensitive to how much
silence padding surrounds a spoken command in a fixed-length clip) means
a single held-out val-split-chosen threshold is already a compromise,
not a universal cutoff -- keeping the raw decode separate from the
accept/reject decision is what lets that decision evolve (a different
threshold, a smarter policy, a `--log-all-windows` audit trail) without
touching the decoder or re-running inference.

### The mode-across-a-bounded-listening-period policy (`ModePeriodPolicy`)

Two related, real (not hypothetical) failure modes were observed testing
this feature against the live microphone on real hardware, both
consistent with the confidence/padding-sensitivity finding above:

1. **Lingering duplicate triggers on a correctly-spoken phrase.** At the
   shipped defaults (`window_s=2.5`, `stride_s=0.25`, `refractory_s=1.5`),
   a spoken phrase stays inside the sliding window for close to its full
   duration as the window advances, so it decodes correctly across
   roughly ten consecutive windows, not one. `Debouncer` only blocks a
   *second* trigger for `refractory_s` after the first; since the phrase
   can still be sitting in the window once that cooldown expires, it
   fires again on what is mechanically the same utterance. Observed
   directly in manual testing: two `TIME` triggers at `t=2.0s` and
   `t=3.5s` -- a gap of exactly `1.5s`, matching `refractory_s` to the
   decimal -- from a single lingering decode, not two independent
   utterances.
2. **False positives during fast speech/walkthroughs.** Consistent with
   the padding-sensitivity mechanism: confidence is a mean per-frame
   log-prob over the *whole* fixed window, so a short or partially-formed
   phrase is diluted by the surrounding non-speech frames in the same way
   ambient noise is, and can still land just inside the threshold.

Both are fixed by the `mode-period-gate` feature (shipped 2026-09-21):
the bounded "listening period" is now defined by the `ListeningGate`
seam (section 6, `SpacebarGate` today, a wake-word detector later), and
`ModePeriodPolicy` is the policy that consumes every `WindowObservation`
across one such period and emits at most one consolidated decision for
it. This is exactly the shape `AcceptancePolicy.observe`/`reset`'s
statefulness was reserved for (see above) -- rolling history across a
bounded window of observations, which `ThresholdPolicy` doesn't use but
the protocol already supports. It landed as one new class in `policy.py`
plus one `POLICY_REGISTRY` line (`vcm/streaming/config.py`), as the seam
was designed to allow; the only runner change is the ~6-line
`_evaluate_window` diff documented in section 5 (the consolidated event
must carry a *previous* window's decode, and the gate needs the audio).

**What it does.** `ModePeriodPolicy(threshold, *, gate, period_s)`
implements `observe`/`reset`. Each observation first polls the gate
(`gate.poll(obs.samples_seen, obs.waveform)`). While the gate is closed
the observation is rejected outright (`gate closed (not in a listening
period)`) and nothing is collected. While a period is open, every
observation is appended to the in-flight collection and `observe`
returns the non-accepting `collecting (...)` reason (section 5). When
the first observation with
`samples_seen >= open_at + int(period_s * SAMPLE_RATE)` arrives, the
period is **flushed**: the collected observations are grouped by decode
class `(intent, tuple(sorted(slots.items())))`, the winning class is
picked by (count desc, confidence-sum desc, earliest-first-seen asc) --
a deterministic total order, no randomness -- and, when the winning
class is a real intent, the class's **mean** confidence is compared to
the operating threshold: accept iff `mean >= threshold`. Whenever a
real decode class wins, the decision's `result` is set on **both**
accept and reject to the latest winning observation's decode with its
confidence replaced by the mean (`dataclasses.replace`), so the runner
emits the mode's intent/slots/text at the mean confidence even when the
period-end window itself decoded something else (or nothing); when the
winning class is the `None` class the flush rejects with the
silence-dominated reason instead and sets no `result`. In every case
the collection is cleared after the flush, so a
period yields at most one accept, and a period that is still open when
the run ends (Ctrl-C / `--listen-for` / EOF) is simply discarded --
there is no flush-on-exit. Two further rules are part of the algorithm:
a **re-press** mid-period (the gate reporting a *new*
`open_at_samples` while collecting) discards all in-flight observations
and restarts the period at the new position; and a press landing on the
*exact* stride a period ends returns the flush decision for that
observation **and** seeds the new period's collection with the same
observation (settled at plan confirmation: identical situations are
handled identically, rather than the new period starting at the next
observation). `reset()` clears the in-flight period; the runner calls it
at the start of every run.

**Why both failure modes die.** Failure mode 1 (lingering duplicates):
one period yields at most one accept, so "the same phrase decoded ten
times in a row" collapses into the single mode decision instead of
riding out the debouncer's refractory and firing twice; the
`Debouncer` still independently rate-limits consecutive *periods* as
before (a second period's flush landing inside the refractory is
suppressed and counted in `suppressed` like any other suppression).
Failure mode 2 (ambient false positives): with the gate closed, decodes
are rejected before any threshold arithmetic, so a confident ambient
decode while no period is open cannot trigger at all; inside a period,
the noise must additionally be the *most frequent* decode class across
the whole period, and its *mean* confidence must still clear the
threshold, which intermittent noise can't do.

**The wake-word seam.** The policy depends only on the `ListeningGate`
protocol (section 6), never on `SpacebarGate` itself: a future
wake-word gate that detects the wake word in the audio `poll` is handed
(the `window` argument, carried on `WindowObservation.waveform`) drops
in with zero policy or runner changes -- the operator presses SPACE
today, the model detects the wake word later, and nothing between
`observe` and the emitted event changes.

**Proven by** `tests/test_vcm_streaming_policy.py` (both original
backlog acceptance criteria are named tests there: ten identical correct
decodes across one open period collapse to exactly one accept at the
flush with intent preserved; intermittent low-confidence noise in a
mostly-`None` period never triggers) and
`tests/test_vcm_streaming_integration.py` (the same rules, incl. the
flush/re-press edges, through the real runner).

## 5. JSONL event schema (owned by Task 04, shape fixed here)

One JSON object per line, one line per emitted `TriggerEvent` (or, under
`--log-all-windows`, one line per *evaluated window* regardless of
accept/reject/suppress -- the diagnostic mode this schema exists to
support):

```json
{
  "event": "trigger" | "window",
  "t_seconds": 12.34,
  "window_index": 42,
  "intent": "CALL",
  "slots": {},
  "text": "call mom",
  "confidence": -0.87,
  "policy_reason": "intent='CALL' confidence=-0.87 >= threshold -1.0"
}
```

- `event`: `"trigger"` for an actually-emitted `TriggerEvent`; `"window"`
  for a non-emitted window only ever logged under `--log-all-windows`
  (rejected by policy, or accepted by policy but suppressed by the
  debouncer -- `policy_reason` and/or a suppression note distinguish
  the two).
- `t_seconds`: wall-clock-equivalent time, i.e. `samples_seen /
  common.features.SAMPLE_RATE`.
- `intent`/`slots`/`text`/`confidence`: from the **authoritative**
  result -- `PolicyDecision.result` when the policy set it, else the
  window's own `DecodeResult` (`None`/`{}`/`""`/`-inf`-equivalent when
  no grammar terminal was reached at all). A `ModePeriodPolicy` flush
  sets it whenever a real decode class wins (on both accept and
  reject): the period's winning decode class with its confidence
  replaced by the class mean; the silence-dominated flush sets no
  `result`. So a gated run's emitted event carries the *mode's*
  intent/slots/text while `t_seconds`/`window_index` stay the
  period-end window's, which may have decoded something else.
  `ThresholdPolicy` never sets it, so threshold-mode records are
  exactly as before.
- `policy_reason`: `PolicyDecision.reason` verbatim.

### Decision reason formats (per policy)

`policy_reason` is policy-specific; the two shipped policies produce:

`ThresholdPolicy` (one decision per window):

- accept: `intent='CALL' confidence=-0.87 >= threshold -1.0`
- reject, no decode: `no_match: intent is None`
- reject, below threshold: `confidence -1.2 below threshold -1.0`

`ModePeriodPolicy` (one decision per listening period, section 4):

- gate closed, no period open: `gate closed (not in a listening period)`
- period open, still collecting: `collecting (3 obs, running mode
  'TIME' 2/3)` -- the running mode is the current winning class; `2/3`
  is its count over the total observations collected so far (the
  running mode is `None` while no real decode has been seen yet)
- flush, silence-dominated: `mode_period: silence dominated the period
  (None 8/10)` -- the `None` class won, with its count over the period's
  total
- flush, mode clears the threshold: `mode_period: intent='TIME' mean
  confidence -0.45 >= threshold -1.0 over 9 obs`
- flush, mode below the threshold: `mode_period: intent='TIME' mean
  confidence -1.3 below threshold -1.0 over 9 obs`

Under `--log-all-windows`, a gated run's non-emitted `window` records
carry the `gate closed` / `collecting (...)` reasons as they accrue and
the flush records with its flush reason -- no new schema fields either
way.

## 6. The `ListeningGate` seam (owned by the `mode-period-gate` feature; `vcm/streaming/gate.py`)

The component that bounds a `ModePeriodPolicy`'s listening periods
(section 4) shipped as an abstract seam with one concrete stand-in,
`SpacebarGate`. Pipeline position: the gate sits between the observation
stream and the acceptance policy -- `ModePeriodPolicy.observe` polls it
at each stride (`gate.poll(samples_seen, waveform)`) and the fixed order
of section 4 is preserved; a gate sees only the sample position and the
same ring-buffer snapshot the model decoded, never logp/decode
internals.

### Protocol signatures

```python
# vcm/streaming/gate.py
@dataclass(frozen=True)
class GateState:
    is_open: bool
    open_at_samples: Optional[int]   # None when closed

class ListeningGate(Protocol):
    """The seam the future wake-word gate will satisfy. `window` is the
    current ring-buffer snapshot (the same audio the model decoded); a
    WakeWordGate consumes it, SpacebarGate ignores it."""
    def poll(self, samples_seen: int, window: Optional[np.ndarray] = None) -> GateState: ...
    def close(self) -> None: ...

class GateUnavailableError(Exception): ...

class SpacebarGate:
    def __init__(self, stdin, *, key: bytes = b" ",
                 period_s: float = 5.0) -> None: ...
    def poll(self, samples_seen: int, window=None) -> GateState: ...
    def close(self) -> None: ...

GATE_REGISTRY: dict[str, type[ListeningGate]] = {"spacebar": SpacebarGate}
def resolve_gate(name: str, *, period_s: float, stdin=sys.stdin) -> ListeningGate: ...
```

### `SpacebarGate` -- raw TTY, one key, one period

`SpacebarGate` enters minimal raw mode on the operator's own `stdin`
(clears `ICANON|ECHO`, keeps `ISIG` -- so Ctrl-C still delivers SIGINT
and is never a gate key), and per `poll` reads pending keypresses with
`select(timeout=0)` + `os.read(fd, 1)`, draining everything pending at
that stride (a double-press within one stride yields one
`open_at_samples`, i.e. one period, not two). A press of `key` (default
SPACE) opens a period at the current `samples_seen` -- a press while
already open restarts the period there (discard-and-restart; the policy
side turns the new `open_at_samples` into an in-flight collection
reset, section 4) -- every other byte is read and discarded with no
state change, and the period auto-closes at
`open_at + int(period_s * SAMPLE_RATE)`. That auto-close expression is
the same one the policy's flush boundary uses, from the same
`gate_period_s` source, so the two close boundaries always agree.
`close()` is idempotent, restores the exact saved termios attributes,
and is additionally registered with `atexit` as a backstop for failure
paths between construction and the runner's `try/finally`. Press
latency is at most one stride (250 ms at the default `stride_s`): the
gate is polled at stride boundaries, not on a faster tick, so the
runner keeps its shape.

**Non-TTY failure mode:** if `stdin` is not a usable TTY (piped/
redirected input, CI, no file descriptor at all), construction raises
`GateUnavailableError` with an actionable message naming the three ways
out (run in an interactive shell, pass `--gate none`, or replay a file
with `--source <wav>`); the CLI catches it and exits 1 with that
message, never a traceback -- the same fail-fast pattern as
`MicrophoneUnavailableError` (section 1).

### `GATE_REGISTRY` / `resolve_gate`

`GATE_REGISTRY` maps CLI names to gate classes (today: `"spacebar"`);
`resolve_gate(name, *, period_s, stdin=sys.stdin)` constructs the named
gate, and an unknown name raises an actionable `SystemExit` naming
`sorted(GATE_REGISTRY)` -- the same loud-error style as
`resolve_policy`/`resolve_model` (section 2).

### CLI wiring (`--gate` / `--gate-period`)

`--gate none|spacebar` (default `none`) and `--gate-period <seconds>`
(argparse dest `gate_period_s`, default `5.0`) -- or the
`gate`/`gate_period_s` fields in a `--config` JSON, validated the same
way (section 2). The CLI cross-validates `--gate` against `--policy` in
**both directions as a hard error before any side effect** (no model
load, no TTY raw mode, no microphone): `--policy mode_period` with
`--gate none` is rejected (a mode-period policy with no bounded period
is meaningless), and `--gate spacebar` with `--policy threshold` is
rejected (the threshold policy consumes no periods, so the gate would
be silently ignored). When a gate is configured, it is constructed
after config resolution and **before** the model load and microphone
open (piped stdin therefore fails before any device is touched),
passed to `resolve_policy(..., gate=gate, period_s=cfg.gate_period_s)`,
and closed in a `try/finally` around `runner.run()` (the `atexit`
registration inside `SpacebarGate` is only the backstop for failure
paths before the `try`). The startup banner reports the gate:
`listening gate: none`, or, with a gate,
`listening gate: spacebar (press SPACE to open a 5.0 s listening
period)` -- the as-shipped wording prints the float followed by a space
and `s`, so `--gate-period 10` prints `10.0 s`.

### The `--log-periods` digest (stderr; stdout JSONL untouched)

The middle verbosity between "triggers only" and `--log-all-windows`:
one line per *period lifecycle event*, human-readable on **stderr**
(stdout stays pure JSONL for machine consumers), with
`--log-all-windows` unchanged as the full-verbosity option. `--log-
periods` (argparse dest `log_periods`, `store_true`, default `False`
off the `log_periods` config field) requires no new cross-validation:
the existing gate/policy pair rules mean a gate exists exactly when a
period lifecycle exists, and the flag is a silent no-op on
`--policy threshold --gate none`.

When enabled, `main()` wraps the constructed gate in a pass-through
(`__main__._RunEndGateLogger`) and passes its printer
(`__main__._print_period_event`) to the policy via
`resolve_policy(..., on_period_event=...)`. The policy emits the
lifecycle events -- it observes the same gate states the gate produces
internally, at the same `samples_seen`, so the digest loses no
transitions:

- `open` at a press, `reopened` at a mid-period re-press (the
  discard-and-restart, section 4), `closed` at the flush carrying the
  period's consolidated `PolicyDecision` (the accept/reject reason,
  section 5's formats, with the `mode_period: ` prefix stripped).
- A press landing on the exact flush stride (section 4's edge) prints
  the `closed` line for the period that ended and then the `open` line
  for the one that starts, at the same `t`.

As-shipped line formats (`__main__._print_period_event`):

```
gate: open      t=12.30s
gate: reopened  t=15.10s
gate: closed    t=17.30s  period: ACCEPT  intent='TIME' mean confidence 0.87 >= threshold 0.7 over 8 obs
gate: closed    t=27.10s  period: REJECT  silence dominated the period (None 9/10)
gate: closed    (run ended)
```

The last line is the wrapper's: the run ended (Ctrl-C/EOF) with a
period still open -- the in-flight period is discarded, never flushed,
so the digest closes the open line instead of ending unbalanced. A run
whose polls never observed an open period prints nothing (no `open` was
printed, so nothing to balance). An accepted flush also still emits its
`"event": "trigger"` JSONL line on stdout exactly as before; the digest
line is a human echo, and a debouncer-suppressed accept shows up as the
digest line without a following trigger line (plus the run summary's
`suppressed` count).

### `WakeWordGate` -- the trained DS-CNN, in place of the keypress

Feature `wakeword-gate` (`feature-engineering/wakeword-gate/SPEC.md`)
implements the wake-word gate this section originally only promised was
possible: `WakeWordGate` (`vcm/streaming/wakeword_gate.py`) satisfies
`ListeningGate` by running the trained wakeword DS-CNN (feature
`wakeword-dscnn`) against the same ring-buffer `window` `SpacebarGate`
ignores, with **zero changes to `AcceptancePolicy`/`StreamingRunner`**.

`poll(samples_seen, window)` takes the **trailing** `WAKEWORD_WINDOW_SECONDS`
(1.5s) slice of `window` -- never a centered crop of the whole buffer, which
can be several seconds long (`period_s + window_s`) and would otherwise let
the classifier score stale audio from earlier in the period.
`wakeword.augment.center_window` is reused only for its zero-pad branch (the
buffer hasn't filled 1.5s yet); it's a no-op once the input is already
exactly `WAKEWORD_WINDOW_SAMPLES` long. The cropped waveform goes through
the same `LogMelFeatureExtractor` front-end and `DSCNN` forward pass the
wakeword training/eval pipeline uses, then a softmax over the 3-way logits;
crossing `threshold` (default `DEFAULT_WAKEWORD_THRESHOLD = 0.9`) opens/
restarts the period at `samples_seen` -- identical discard-and-restart
semantics to a `SpacebarGate` re-press, so `ModePeriodPolicy`/
`SinglePeriodPolicy` consume it unmodified.

Two inference backends, mirroring `InferenceBackend`'s `OnnxBackend`/
`TorchBackend` split but classifier-shaped (`wakeword_prob(waveform) ->
float`, not `logp_for_waveform`'s `(T, alphabet)`):
`WakewordOnnxBackend`/`WakewordTorchBackend`. `WakewordTorchBackend` loads
its checkpoint via `torch.load(..., weights_only=True)`, refusing rather
than silently falling back to unsafe pickle deserialization -- the same
convention `TorchBackend` uses (`backends.py`'s module docstring), not
`vcm.export_onnx.load_checkpoint`'s plain `torch.load`.

`GATE_REGISTRY["wakeword"]` is registered as a module-level side effect in
`config.py` (not in `gate.py` itself) to avoid a `gate.py` <->
`wakeword_gate.py` import cycle -- the same place `MODEL_REGISTRY`/
`POLICY_REGISTRY` already live. `resolve_gate` gained optional
`wakeword_backend`/`wakeword_threshold` kwargs and branches on
`name == "wakeword"` to use them instead of `stdin`.

CLI: `--gate none|spacebar|wakeword`, `--wakeword-model` (registry name /
run dir / file, independent of `--model`, which is the VCM decode model),
`--wakeword-backend {onnx,torch}` (default `torch`), `--wakeword-onnx-variant
{fp32,int8}`, `--wakeword-threshold` (default `0.9`). The `--gate`/`--policy`
cross-validation in `main()` is generalized from `cfg.gate == "spacebar"` to
`cfg.gate != "none"` -- a wakeword gate is exactly as bound-to-a-period as
spacebar. `make stream-wakeword` is the Makefile convenience target
(mirrors `stream-single-period`).

No detection-threshold calibration pipeline (unlike VCM's per-grammar
`chosen_operating_threshold` resolved from `eval_report.json`) -- a fixed
default, tunable by hand. Deferred as future work per the feature's SPEC.

### Product-level integration (still not decided): cold start

> Cold start is non-trivial (`torch`/`torchaudio` imported at module
> level via `common/features.py`'s `LogMelFeatureExtractor` wrapping
> `torchaudio.transforms.MelSpectrogram` -- the ONNX backend does NOT
> avoid this; plus ORT session construction, `compile_grammar`'s
> 129-phrase trie build, ~0.2s `arecord` device-open). A long-running
> process's ring buffer already holds pre-gate audio vs. a freshly-spawned
> one starting empty -- BUT the planned future product UX (wake-word ->
> deliberate delay -> "Listening..." indicator + audio chime -> user cued
> to speak) is expected to mitigate cold start at the integration layer
> AND substantially neutralizes the pre-gate-audio advantage too (a user
> cued to wait won't have spoken into the pre-gate buffer). Both
> spawn-per-trigger and signal-a-running-process therefore remain viable;
> this is genuinely not decided here -- record it so a future integration
> decision has both facts, not just the risk in isolation.

## 7. Symbols this feature reuses (source of truth: the named module)

- `vcm.pipeline.WINDOW_S` / `STRIDE_S` / `DEFAULT_REFRACTORY_S`,
  `infer_waveform`, `logp_for_waveform`, `load_checkpoint`.
- `common.features.SAMPLE_RATE`, `LogMelFeatureExtractor`.
- `vcm.evaluate.decode_split`, `NEG_INF_THRESHOLD`, `_accepted`.
- `vcm.decoder.DecodeResult`, `decode`.
- `vcm.streaming.buffer.RingBuffer`, `vcm.streaming.debounce.Debouncer`,
  `vcm.streaming.policy.{WindowObservation,PolicyDecision,
  AcceptancePolicy,ThresholdPolicy,ModePeriodPolicy}` (this feature's
  own new symbols, Task 01; `ModePeriodPolicy` landed later, with the
  `mode-period-gate` feature, 2026-09-21).
- `vcm.streaming.sources.{AudioSource,RawPcmStreamSource,WavFileSource,
  MicrophoneSource,open_microphone_source,MicrophoneUnavailableError,
  DEFAULT_MIC_COMMAND}` (this feature's own new symbols, Task 02).
- `vcm.streaming.gate.{GateState,ListeningGate,GateUnavailableError,
  SpacebarGate,GATE_REGISTRY,resolve_gate}` (new module,
  `mode-period-gate` feature, 2026-09-21; source of truth
  `vcm/streaming/gate.py`, section 6).
- `vcm.streaming.config.{StreamingConfig,POLICY_REGISTRY,
  resolve_policy}` now also carry the `gate`/`gate_period_s` fields and
  the `mode_period` registry entry (section 2).
