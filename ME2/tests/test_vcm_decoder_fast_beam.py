"""The fast `prefix_beam_search` must be bit-identical to the original loop
(`_prefix_beam_search_reference`): same prefixes, same order, same `pb`/`pnb` floats.
See docs/BEAM-SEARCH.md section 6 and `.scratch/fast-beam-search/`."""

from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from me2_voicegen.common import grammar_core as gc
from me2_voicegen.vcm import alphabet as A
from me2_voicegen.vcm import decoder as dec
from me2_voicegen.vcm.optiona import grammar as og
from me2_voicegen.vcm.optionb.grammar import OPTIONB_GRAMMAR

BACKENDS = ["python", "numba"]  # numba is a declared dependency: no silent skip
FIXTURES = Path(__file__).parent / "fixtures" / "dense_pilot"
GRAMMARS = {"optionb": OPTIONB_GRAMMAR, "spec": og.SPEC_GRAMMAR, "toy": og.TOY_GRAMMAR}
BEAMS = [50, 10, 1]


def signature(beams):
    return [(p, id(e.node), e.pb.hex(), e.pnb.hex()) for p, e in beams.items()]


def run(backend, logp, root, beam):
    return dec._prefix_beam_search_with(logp, root, beam, backend)


def ref(logp, root, beam):
    return dec._prefix_beam_search_reference(logp, root, beam)


def _clips():
    out = []
    for name in ("val_clean", "val_noisy_s0"):
        z = np.load(FIXTURES / f"{name}.npz")
        lp, off = z["logp"], z["offsets"]
        out += [lp[a:b] for a, b in zip(off[:-1], off[1:]) if b > a]
    return out


CLIPS = _clips()


def _log_softmax(x):
    m = x.max(axis=-1, keepdims=True)
    return x - (m + np.log(np.exp(x - m).sum(axis=-1, keepdims=True)))


def _synthetic():
    rng = np.random.default_rng(7)
    uniform = np.full((30, A.ALPHABET_SIZE), -np.log(A.ALPHABET_SIZE))
    peaky = _log_softmax(rng.normal(scale=6.0, size=(60, A.ALPHABET_SIZE)))
    flat = _log_softmax(rng.normal(scale=0.5, size=(40, A.ALPHABET_SIZE)))
    with_ninf = flat.copy()
    with_ninf[:, 3:12] = -np.inf
    with_ninf[::5, A.BLANK_ID] = -np.inf
    return {"uniform": uniform, "peaky": peaky, "flat": flat, "neg_inf": with_ninf, "empty": np.zeros((0, A.ALPHABET_SIZE))}


SYNTHETIC = _synthetic()


def test_fixture_is_present_and_real():
    assert len(CLIPS) >= 30


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("gname", GRAMMARS)
def test_fixture_clips_identical(backend, gname, beam):
    root = GRAMMARS[gname].root
    for clip in CLIPS:
        assert signature(run(backend, clip, root, beam)) == signature(ref(clip, root, beam))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("gname", GRAMMARS)
@pytest.mark.parametrize("name", list(SYNTHETIC))
def test_synthetic_identical(backend, name, gname, beam):
    root, lp = GRAMMARS[gname].root, SYNTHETIC[name]
    assert signature(run(backend, lp, root, beam)) == signature(ref(lp, root, beam))


@pytest.mark.parametrize("backend", BACKENDS)
def test_float32_input_identical(backend):
    root = OPTIONB_GRAMMAR.root
    lp = SYNTHETIC["peaky"].astype(np.float32)
    assert signature(run(backend, lp, root, 50)) == signature(ref(lp, root, 50))


@pytest.mark.parametrize("backend", BACKENDS)
def test_empty_grammar_and_wide_beam(backend):
    empty = gc.compile_grammar("E", {})
    lp = SYNTHETIC["flat"]
    assert signature(run(backend, lp, empty.root, 50)) == signature(ref(lp, empty.root, 50))
    tiny = gc.compile_grammar("T", {"r": [(("h", "i"), {}, "A"), (("h", "e", "y"), {}, "B")]})
    assert signature(run(backend, lp, tiny.root, 1000)) == signature(ref(lp, tiny.root, 1000))


@pytest.mark.parametrize("backend", BACKENDS)
def test_empty_input_returns_start_beam(backend):
    out = run(backend, SYNTHETIC["empty"], OPTIONB_GRAMMAR.root, 50)
    assert list(out) == [""] and out[""].node is OPTIONB_GRAMMAR.root
    assert (out[""].pb, out[""].pnb) == (0.0, dec.NEG_INF)


def test_nan_and_posinf_route_to_reference():
    root = OPTIONB_GRAMMAR.root
    for bad in (np.nan, np.inf):
        lp = SYNTHETIC["flat"].copy()
        lp[3, 5] = bad
        for backend in BACKENDS:
            assert signature(run(backend, lp, root, 10)) == signature(ref(lp, root, 10))


def test_bad_inputs_behave_like_reference():
    root = OPTIONB_GRAMMAR.root
    lp = SYNTHETIC["flat"]
    for args in ((lp[:, :28], root, 10), (lp, root, 0)):
        try:
            expected = ref(*args)
        except Exception as exc:  # noqa: BLE001
            with pytest.raises(type(exc)):
                run("numba", *args)
        else:
            assert signature(run("numba", *args)) == signature(expected)
    with pytest.raises(ValueError):
        dec._prefix_beam_search_with(lp, root, 10, "nope")


def test_non_tree_trie_routes_to_reference():
    a, b, shared = gc.TrieNode(), gc.TrieNode(), gc.TrieNode()
    shared.terminal = [("X", {})]
    root = gc.TrieNode()
    root.children = {"a": a, "b": b}
    a.children = {"c": shared}
    b.children = {"c": shared}  # DAG: `shared` has two parents
    assert dec._compile_trie(root) is None
    lp = SYNTHETIC["peaky"]
    for backend in BACKENDS:
        assert signature(run(backend, lp, root, 10)) == signature(ref(lp, root, 10))


def test_compiled_trie_drops_non_alphabet_edges():
    compiled = dec._compile_trie(OPTIONB_GRAMMAR.root)
    assert not any(ch.isdigit() for p in compiled.prefixes for ch in p)
    raw, stack = 0, [OPTIONB_GRAMMAR.root]
    while stack:
        n = stack.pop()
        raw += 1
        stack.extend(n.children.values())
    assert len(compiled.nodes) < raw
    assert all(list(c) == sorted(c) for c in compiled.children)


def test_cache_hits_and_is_bounded():
    dec._COMPILED_CACHE.clear()
    first = dec._get_compiled(OPTIONB_GRAMMAR.root)
    assert dec._get_compiled(OPTIONB_GRAMMAR.root) is first
    assert dec._get_compiled(dataclasses.replace(OPTIONB_GRAMMAR, name="copy").root) is first
    for i in range(20):
        dec._get_compiled(gc.compile_grammar(f"G{i}", {"r": [((f"w{i}",), {}, "I")]}).root)
        assert len(dec._COMPILED_CACHE) <= dec._COMPILED_CACHE_MAX


def test_decode_utterance_unchanged(monkeypatch):
    root_grammar = OPTIONB_GRAMMAR
    got = [dec.decode_utterance(c, root_grammar, -0.1, beam_width=50, required_command_margin=4.0) for c in CLIPS]
    monkeypatch.setattr(dec, "prefix_beam_search", dec._prefix_beam_search_reference)
    want = [dec.decode_utterance(c, root_grammar, -0.1, beam_width=50, required_command_margin=4.0) for c in CLIPS]
    assert got == want


# ---- backend selection -------------------------------------------------------------------------------------------


@pytest.fixture
def fresh_backend(monkeypatch):
    dec._reset_backend()
    yield monkeypatch
    dec._reset_backend()


@pytest.mark.parametrize("want", ["python", "reference", "numba"])
def test_env_selects_backend(fresh_backend, want):
    fresh_backend.setenv("ME2_BEAM_BACKEND", want)
    assert dec.beam_backend() == want
    lp = SYNTHETIC["peaky"]
    assert signature(dec.prefix_beam_search(lp, OPTIONB_GRAMMAR.root, 50)) == signature(ref(lp, OPTIONB_GRAMMAR.root, 50))


def test_auto_prefers_numba(fresh_backend):
    fresh_backend.delenv("ME2_BEAM_BACKEND", raising=False)
    assert dec.beam_backend() == "numba" and dec.numba_import_error() is None


def test_invalid_env_raises(fresh_backend):
    fresh_backend.setenv("ME2_BEAM_BACKEND", "fast")
    with pytest.raises(ValueError):
        dec.beam_backend()


def _no_numba(name):
    raise ImportError("no numba here")


def test_auto_falls_back_to_python_and_reports_why(fresh_backend):
    fresh_backend.delenv("ME2_BEAM_BACKEND", raising=False)
    fresh_backend.setattr(dec, "_load_numba", lambda: _no_numba("x"))
    assert dec.beam_backend() == "python"
    assert "no numba here" in dec.numba_import_error()


def test_explicit_numba_unavailable_is_an_error(fresh_backend):
    fresh_backend.setenv("ME2_BEAM_BACKEND", "numba")
    fresh_backend.setattr(dec, "_load_numba", lambda: _no_numba("x"))
    with pytest.raises(RuntimeError):
        dec.beam_backend()


def test_warm_up_returns_backend_and_time(fresh_backend):
    fresh_backend.setenv("ME2_BEAM_BACKEND", "python")
    name, ms = dec.warm_up(OPTIONB_GRAMMAR.root, 50)
    assert name == "python" and ms >= 0


def test_importing_decoder_does_not_import_numba():
    code = "import sys, me2_voicegen.vcm.decoder; sys.exit('numba' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
