"""Where the per-window decode time goes: encoder vs the grammar beam search, and what the beam search actually does.

Slides a 2.5 s window over a wav at the live stride (0.25 s), times `OnnxBackend.logp_for_waveform` (log-mel features + INT8 ONNX + log-softmax)
and the beam search at each beam width for each backend (the original `reference` loop, the exact `python` fast path, and `numba`; all three
must return bit-identical beams on every window or the script stops), and reports how blank-dominated the frames are, how many characters the grammar allows per
step, and how full the beam is. Single thread, no profiler, so the times are the real ones (they include windows outside a wake-word period;
the soak only decodes windows inside one). `docs/BEAM-SEARCH.md` quotes the output.

    PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python scripts/profile_beam_search.py --wav soak/holdout-wake-gap-v1/continuous.wav --seconds 120
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf

import me2_voicegen.vcm.decoder as D
from me2_voicegen.vcm import alphabet as A
from me2_voicegen.vcm.streaming.backends import OnnxBackend
from me2_voicegen.vcm.streaming.config import resolve_grammar

SR, WINDOW_S, STRIDE_S = 16000, 2.5, 0.25
ROOT = Path(__file__).resolve().parents[1]


def counted_beam_search(logp: np.ndarray, root, beam_width: int) -> tuple[dict, list[int]]:
    """`decoder.prefix_beam_search` with the same steps, returning the live beam count at the start of each frame (the real
    function is the one under test: `main` asserts both return the same prefixes)."""
    beams = {"": D.BeamEntry(node=root, pb=0.0, pnb=D.NEG_INF)}
    live: list[int] = []
    for t in range(logp.shape[0]):
        live.append(len(beams))
        nxt: dict[str, D.BeamEntry] = {}

        def add(prefix, node, d_pb=D.NEG_INF, d_pnb=D.NEG_INF):
            e = nxt.get(prefix)
            if e is None:
                e = nxt[prefix] = D.BeamEntry(node=node)
            e.pb, e.pnb = D._logsumexp(e.pb, d_pb), D._logsumexp(e.pnb, d_pnb)

        lt = logp[t]
        for prefix, e in beams.items():
            total, last = e.total(), (prefix[-1] if prefix else None)
            add(prefix, e.node, d_pb=total + float(lt[A.BLANK_ID]))
            for cid, ch in A.ID_TO_CHAR.items():
                if cid == A.BLANK_ID:
                    continue
                c = float(lt[cid])
                if ch == last:
                    add(prefix, e.node, d_pnb=e.pnb + c)
                    child = e.node.children.get(ch)
                    if child is not None:
                        add(prefix + ch, child, d_pnb=e.pb + c)
                else:
                    child = e.node.children.get(ch)
                    if child is not None:
                        add(prefix + ch, child, d_pnb=total + c)
        if len(nxt) > beam_width:
            nxt = dict(sorted(nxt.items(), key=lambda kv: kv[1].total(), reverse=True)[:beam_width])
        beams = nxt
    return beams, live


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wav", type=Path, required=True)
    ap.add_argument("--seconds", type=float, default=120.0, help="use the first N seconds of the wav")
    ap.add_argument("--model", type=Path, default=ROOT / "out/vcm/hybrid-ctcwide-clsxl/ctc-wide/export/vcm_model.int8.onnx")
    ap.add_argument("--beams", type=int, nargs="+", default=[50, 25, 10])
    a = ap.parse_args()

    grammar, _ = resolve_grammar("optionb")
    backend = OnnxBackend(a.model, ort_threads=1)
    x, sr = sf.read(a.wav, dtype="float32", frames=int(a.seconds * SR))
    assert sr == SR
    win, stride = int(WINDOW_S * SR), int(STRIDE_S * SR)
    windows = [x[s:s + win] for s in range(0, len(x) - win, stride)]
    backend.logp_for_waveform(windows[0])  # warm-up

    t0 = time.perf_counter()
    logps = [backend.logp_for_waveform(w) for w in windows]
    enc_ms = 1000 * (time.perf_counter() - t0) / len(windows)
    T = logps[0].shape[0]
    print(f"{len(windows)} windows of {WINDOW_S} s at stride {STRIDE_S} s; T = {T} frames per window ({1000 * WINDOW_S / T:.0f} ms per frame)")
    print(f"encoder (features + ONNX + log-softmax): {enc_ms:.1f} ms per window")
    backends = ["reference", "python"]
    try:
        D._load_numba()
        backends.append("numba")
        D.warm_up(grammar.root, 50)  # JIT compile outside the timings
    except (ImportError, OSError) as exc:
        print(f"numba unavailable ({exc}); skipping that backend")

    def signature(beams):
        return [(p, id(e.node), e.pb.hex(), e.pnb.hex()) for p, e in beams.items()]

    for b in a.beams:
        ref_sigs, ref_ms = None, None
        for backend in backends:
            t0 = time.perf_counter()
            results = [D._prefix_beam_search_with(lp, grammar.root, b, backend) for lp in logps]
            ms = 1000 * (time.perf_counter() - t0) / len(logps)
            sigs = [signature(r) for r in results]
            if backend == "reference":
                ref_sigs, ref_ms = sigs, ms
            else:
                assert sigs == ref_sigs, f"{backend} diverged from the reference at beam {b}"
            note = "" if backend == "reference" else f", {ref_ms / ms:.1f}x, identical on all {len(logps)} windows"
            print(f"beam search, beam {b:>2}, {backend:>9}: {ms:7.2f} ms per window = {100 * ms / (ms + enc_ms):.0f}% of encoder + beam search{note}")

    blank = np.concatenate([np.exp(lp[:, A.BLANK_ID]) for lp in logps])
    print(f"frames with P(blank) > 0.99: {100 * (blank > 0.99).mean():.1f}%, > 0.999: {100 * (blank > 0.999).mean():.1f}%")
    children, stack = [], [grammar.root]
    while stack:
        node = stack.pop()
        children.append(len(node.children))
        stack.extend(node.children.values())
    n_chars = len(A.ID_TO_CHAR) - 1
    print(f"grammar trie: {len(children)} nodes, children per node mean {np.mean(children):.2f}, max {max(children)}; "
          f"the loop tests {n_chars} characters per beam per frame")
    sample = logps[:30]
    live = []
    for lp in sample:
        beams, per_frame = counted_beam_search(lp, grammar.root, max(a.beams))
        assert set(beams) == set(D._prefix_beam_search_reference(lp, grammar.root, beam_width=max(a.beams))), "instrumented copy diverged from decoder.py"
        live += per_frame
    live = np.array(live)
    print(f"beam {max(a.beams)}: mean live beams per frame {live.mean():.1f}, full on {100 * (live >= max(a.beams)).mean():.0f}% of frames; "
          f"~{live.mean() * n_chars * T:,.0f} inner-loop iterations per window (first {len(sample)} windows)")


if __name__ == "__main__":
    main()
