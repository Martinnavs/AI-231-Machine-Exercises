# Adding a new TTS backend

As of this spike, `generate_sample.py` is genuinely backend-agnostic — `--backend` dispatches
through `me2_voicegen.synthesis.factory`, and no backend-specific string is hardcoded anywhere in
the CLI except the `DEFAULT_BACKEND` constant's value. This is proven, not just asserted:
`tests/test_generate_sample_cli.py::FakeSynthesizer` has a deliberately unrelated constructor
signature (`voice_id`/`speed`, vs. the real backend's `model_dir`/`device`/`fp16`) and is driven
end-to-end through `generate_sample.main()`. If you're reading this because you're about to add a
second backend (e.g. Piper, XTTS-v2), read this doc — don't rediscover the same design.

This is the ME2 analogue of the reference `simple-audio-transcriber` repo's own
`design/adding-a-backend.md`; the shape below mirrors it deliberately.

## Steps

1. **Implement `Synthesizer`** (`src/me2_voicegen/synthesis/base.py`) in a new module under
   `synthesis/`, e.g. `synthesis/my_backend.py`:

   ```python
   from .base import Synthesizer, SynthesisResult, VoicePrompt

   class MyBackendSynthesizer(Synthesizer):
       def __init__(self, model_dir: str | None = None, device: str = "auto", **kwargs) -> None:
           # heavy imports go here, not at module top level - see "Gotcha" below
           ...

       def synthesize(self, text: str, prompt: VoicePrompt | None = None) -> SynthesisResult:
           if prompt is None:
               raise ValueError("MyBackendSynthesizer requires a voice prompt")
           ...
           return SynthesisResult(audio=audio_np_float32, sample_rate=sr)
   ```

   It's an `ABC` (`class Synthesizer(ABC)`), not a `Protocol` — instantiating a subclass that
   hasn't implemented `synthesize()` fails immediately with `TypeError`, which is deliberate
   (same rationale the reference repo's `Transcriber` ABC recorded for its own design).

   The contract: `synthesize(text: str, prompt: VoicePrompt | None = None) -> SynthesisResult`.
   `VoicePrompt` is `(wav_path: Path, text: str | None = None)`; `SynthesisResult` is
   `(audio: np.ndarray, sample_rate: int)` where `audio` is **float32, shape `(channels,
   samples)`, never a `torch.Tensor`** — even a torch-based backend must convert
   (`.detach().cpu().numpy().astype(np.float32)`) before returning, so a future fully torch-free
   backend (e.g. Piper) can plug into the same contract with zero torch dependency of its own.
   `prompt` is `Optional` because a non-cloning backend has no reference clip to work from — if
   your backend requires one, raise a clear error when `prompt is None` rather than silently
   degrading (mirrors `CosyVoice2Synthesizer`'s behavior). Backend configuration (model dir,
   device, precision, ...) belongs entirely in your concrete `__init__`, never as a `synthesize()`
   parameter, so callers stay decoupled from any one backend's config shape.

   If your backend has a natural default reference asset (like CosyVoice2's vendored
   `zero_shot_prompt.wav`), declare it as optional class attributes rather than hardcoding it in
   the CLI:

   ```python
   class MyBackendSynthesizer(Synthesizer):
       DEFAULT_PROMPT_WAV: Path | None = Path("path/to/my/default/prompt.wav")
       DEFAULT_PROMPT_TEXT: str | None = "transcript of that clip"
   ```

   `generate_sample.py`'s `_build_prompt()` discovers these generically via `getattr(backend_cls,
   "DEFAULT_PROMPT_WAV", None)` — this is exactly how `CosyVoice2Synthesizer` supplies its own
   default prompt without the CLI file ever mentioning "cosyvoice" by name.

2. **Register it** in `synthesis/factory.py`:

   ```python
   _BACKENDS: dict[str, type[Synthesizer]] = {
       "cosyvoice2": CosyVoice2Synthesizer,
       "my-backend": MyBackendSynthesizer,
   }
   ```

   That's it — `--backend my-backend` now works, `--help` lists it in `--backend`'s choices
   automatically (`list_backends()` reads `_BACKENDS` live and is what builds `argparse`'s
   `choices=`), and nothing else in the CLI needs touching. This is a flat eager dict, not a
   plugin/entry-point system — keep it that way unless a real need for dynamic registration shows
   up. Registering here also gets your backend `generate_personas.py`'s batch-over-personas CLI for
   free — it dispatches through the same factory and `cli_common` helpers as `generate_sample.py`,
   so there's no separate wiring step for the persona-batch path.

3. **Common CLI flags → constructor kwargs.** `generate_sample.py` builds a config dict from the
   flags below using these exact constructor parameter names, and (via `inspect.signature`) only
   forwards the ones your `__init__` actually declares — anything you don't accept is silently
   skipped, not an error:

   | CLI flag | forwarded as constructor kwarg | use it if your backend has... |
   |---|---|---|
   | `--device` | `device` | a local-compute device choice (`auto`/`cuda`/`cpu`) |

   (Only `--device` exists as a common flag today — `cosyvoice2`'s `model_dir`/`fp16` are reached
   entirely through `--opt`, on purpose, because there's only one backend so far and adding a
   common flag for a config knob only one backend has isn't justified yet. If a second backend
   also wants `model_dir` or `fp16` as a first-class flag, that's the point to promote it into
   this table — don't add a common flag speculatively for a backend you haven't built.)

   Anything not in that table is reachable via `--opt KEY=VALUE` (repeatable), passed straight
   through to your constructor **without** filtering — an unknown `--opt` key is a real,
   immediate error (fail loud), unlike the common flags above. `--opt` values are auto-coerced:
   `true`/`false` → `bool`, `none` → `None`, else `int`/`float`/`str`. Example:
   `--opt model_dir=/path --opt fp16=true --opt trt_concurrent=2`.

4. **Test it.** Two tiers, same pattern the reference repo and this spike both already use:

   - **Fast/mocked** (`tests/test_cosyvoice2_backend.py` is the template): monkeypatch whatever
     underlying model/SDK your backend wraps so tests run in milliseconds with no real weights,
     no GPU, no network. Cover at minimum: successful synthesis mapping to `SynthesisResult`
     (including any chunk-concatenation or tensor→numpy conversion your backend does), the
     missing-prompt error path if your backend requires one, and any fallback/device-handling
     logic you add. You do **not** need a new `@pytest.mark.slow` test per backend unless there's
     a real reason (e.g. your backend's failure mode is something only observable against real
     weights/hardware that a monkeypatched test structurally can't catch) — one real end-to-end
     slow test already exists (`tests/test_end_to_end_slow.py`) proving the factory-to-disk path
     works for real against `cosyvoice2`; a second backend usually only needs the fast tier plus
     confidence that the same factory/CLI plumbing (already covered generically by
     `tests/test_synthesis_factory.py` and `tests/test_generate_sample_cli.py`) still applies to
     it.
   - You do not need to re-test the `--backend`/`--opt`/common-flag-filtering machinery itself —
     `tests/test_generate_sample_cli.py` and `tests/test_synthesis_factory.py` already cover that
     generically against fake backends (including the criterion that no backend-specific string
     ever leaks into `generate_sample.py`). Just test your own backend's logic.

## Gotcha: heavy imports must stay inside the backend class, never at module top level

`CosyVoice2Synthesizer`'s `add_cosyvoice_to_syspath()` call and
`from cosyvoice.cli.cosyvoice import AutoModel` import both live **inside `__init__`**, not at
`cosyvoice2_backend.py`'s module top level. Do the same for your backend: any import of your
backend's own SDK, `torch`, or anything that requires the vendor clone / model weights / a GPU to
exist must be deferred into `__init__` (or the methods that need it), never executed just by
`import`ing the module.

This is not a style preference — it's enforced by a regression test,
`tests/test_smoke_fast_suite.py`, which asserts the real `cosyvoice` package never lands in
`sys.modules` as a side effect of importing `generate_sample`/`synthesis.factory`/
`synthesis.cosyvoice2_backend`, or of calling `list_backends()`/`get_backend_class()`, and that
`--help` exits 0 without touching it either. `list_backends()`, `--help`, and the entire fast test
suite must work with **zero GPU, zero vendor clone, and zero model weights present** — that
property was originally verified only manually (by physically renaming `vendor/` away and
re-running `--help`/the test suite), and `test_smoke_fast_suite.py` turns that one-time check into
an always-on guard against exactly the regression that would silently break it: someone hoisting a
deferred import up to module scope. If you add a backend whose import breaks this guard, the fast
suite will fail — that's the signal telling you to move the import back inside your class.

## If you extend `SynthesisResult` or `VoicePrompt`

Both contracts are deliberately minimal today (`SynthesisResult`: `audio`, `sample_rate` only —
no alignment/prosody metadata; `VoicePrompt`: `wav_path`, optional `text` only). If a new
backend's natural output includes something richer and you want to surface it, that's a real
interface change affecting `cosyvoice2_backend.py` too (it would need to either populate the new
field or leave it at a sensible default) and anything downstream that consumes `SynthesisResult`.
Don't add fields "just in case" for a backend you haven't built yet — YAGNI applies per-field,
not just per-backend.
