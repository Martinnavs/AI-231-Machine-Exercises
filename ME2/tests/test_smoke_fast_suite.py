"""Cross-cutting smoke test for whole-plan criterion 7: the fast suite (no
`slow` marker) must never require the vendor clone, model weights, or a GPU.

Ticket 03's developer verified this once manually by physically renaming
vendor/ away and re-running --help + the full suite. That's a point-in-time
check, not a regression guard - nothing stops a later change from
reintroducing a module-level `import cosyvoice` or an eager AutoModel/torch.cuda
call. This test asserts the same property in a way that runs on every fast
suite invocation: the real `cosyvoice` vendor package must never end up in
sys.modules as a side effect of importing/using the CLI and factory surface
that a fast, no-vendor, no-GPU environment is supposed to exercise.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


def test_cosyvoice_vendor_package_never_imported_by_fast_surface() -> None:
    assert "cosyvoice" not in sys.modules, (
        "the real vendor `cosyvoice` package got imported somewhere in the fast "
        "suite - criterion 7 requires the fast suite to run with no vendor clone"
    )

    from me2_voicegen.generation import generate_sample
    from me2_voicegen.synthesis import factory

    factory.list_backends()
    factory.get_backend_class("cosyvoice2")

    assert "cosyvoice" not in sys.modules


def test_generate_sample_help_exits_cleanly_without_vendor_or_gpu() -> None:
    from me2_voicegen.generation import generate_sample

    with pytest.raises(SystemExit) as exc_info:
        generate_sample.parse_args(["--help"])

    assert exc_info.value.code == 0
    assert "cosyvoice" not in sys.modules


def test_generate_personas_help_exits_cleanly_without_vendor_or_gpu() -> None:
    """Same regression guard as test_generate_sample_help_exits_cleanly_..., for
    the new persona-batch CLI (Ticket 01 of persona-batch-generation) - nothing
    stops a later change from reintroducing a module-level vendor import here
    either."""
    from me2_voicegen.generation import generate_personas

    with pytest.raises(SystemExit) as exc_info:
        generate_personas.parse_args(["--help"])

    assert exc_info.value.code == 0
    assert "cosyvoice" not in sys.modules


def test_cosyvoice2_backend_module_import_alone_does_not_touch_vendor() -> None:
    """Importing the backend module (as opposed to instantiating the class)
    must not eagerly import the vendor package - only __init__ may do that."""
    from me2_voicegen.synthesis import cosyvoice2_backend  # noqa: F401

    assert "cosyvoice" not in sys.modules


# ---------------------------------------------------------------------------
# vcm-toy feature (ticket 08's own cross-cutting regression test): every
# `vcm.*` module must import cleanly with no vendor CosyVoice import, so
# adding this whole feature never drags GPU/TTS-vendor weight requirements
# into the fast suite by accident. `vcm.slot_eval_set` is the one module in
# this subpackage that *does* import from `me2_voicegen.synthesis`/
# `me2_voicegen.generation.personas` (it reuses the existing TTS pipeline per ticket
# 07) - that's exactly the case this guard exists to catch if it ever stops
# being import-time-lazy.
# ---------------------------------------------------------------------------

_VCM_MODULES = [
    "me2_voicegen.vcm.alphabet",
    "me2_voicegen.vcm.text",
    "me2_voicegen.common.features",
    "me2_voicegen.common.augment",
    "me2_voicegen.vcm.dataset",
    "me2_voicegen.vcm.optiona.grammar",
    "me2_voicegen.vcm.decoder",
    "me2_voicegen.vcm.model",
    "me2_voicegen.vcm.train",
    "me2_voicegen.vcm.pipeline",
    "me2_voicegen.vcm.evaluate",
    "me2_voicegen.vcm.export_onnx",
    "me2_voicegen.vcm.benchmark",
    "me2_voicegen.vcm.slot_eval_set",
    "me2_voicegen.vcm.streaming.buffer",
    "me2_voicegen.vcm.streaming.debounce",
    "me2_voicegen.vcm.streaming.gate",
    "me2_voicegen.vcm.streaming.policy",
    "me2_voicegen.vcm.streaming.sources",
    "me2_voicegen.vcm.streaming.config",
    "me2_voicegen.vcm.streaming.backends",
    "me2_voicegen.vcm.streaming.runner",
    "me2_voicegen.vcm.streaming.__main__",
]

# Option B grammar feature (ticket 02): pure-stdlib string-matching modules
# with no reason to ever import the vendor package, but nothing stops a
# later change from adding one - same regression guard as _VCM_MODULES.
_OPTIONB_MODULES = [
    "me2_voicegen.common.grammar_core",
    "me2_voicegen.vcm.optionb.text",
    "me2_voicegen.vcm.optionb.numbers",
    "me2_voicegen.vcm.optionb.grammar",
    "me2_voicegen.vcm.optionb.incomplete_prefix_grammar",
]


def test_vcm_subpackage_modules_never_import_cosyvoice_vendor_package() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n" + "\n".join(f"import {m}" for m in _VCM_MODULES) + "\n"
            "print('cosyvoice' in sys.modules)",
        ],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False", (
        "importing a vcm.* module pulled the real vendor `cosyvoice` package "
        f"into sys.modules - this would break the fast suite's GPU/vendor-free "
        f"guarantee. stdout={result.stdout!r} stderr={result.stderr!r}"
    )


def test_optionb_subpackage_modules_never_import_cosyvoice_vendor_package() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n" + "\n".join(f"import {m}" for m in _OPTIONB_MODULES) + "\n"
            "print('cosyvoice' in sys.modules)",
        ],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False", (
        "importing an optionb.* module pulled the real vendor `cosyvoice` "
        f"package into sys.modules - this would break the fast suite's "
        f"GPU/vendor-free guarantee. stdout={result.stdout!r} stderr={result.stderr!r}"
    )
