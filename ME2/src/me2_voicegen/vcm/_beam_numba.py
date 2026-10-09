"""numba kernel for `vcm.decoder`'s grammar prefix beam search.

Same arithmetic, in the same order, as `decoder._prefix_beam_search_python`
(float64, no fastmath, no parallel), so the result is bit-identical to the
reference. Imported lazily by the decoder (importing numba costs ~1 s);
nothing else should import this module.
"""

from __future__ import annotations

import math

import numba
import numpy as np


@numba.njit(cache=False, fastmath=False)
def _lse(a, b):
    if a == -np.inf:
        return b
    if b == -np.inf:
        return a
    m = a if a > b else b
    return m + math.log(math.exp(a - m) + math.exp(b - m))


@numba.njit(cache=False, fastmath=False)
def search(logp, beam_width, ch_cid, ch_nid, last_cid):
    """logp (T, >=29) float64 C-contiguous; ch_cid/ch_nid (N, maxc) int64, -1 padded; last_cid (N,) int64.
    Returns (node ids in beam order, pb[N], pnb[N]); pb/pnb are only meaningful at the returned ids."""
    T = logp.shape[0]
    N = last_cid.shape[0]
    pb = np.full(N, -np.inf)
    pnb = np.full(N, -np.inf)
    npb = np.full(N, -np.inf)
    npnb = np.full(N, -np.inf)
    seen = np.zeros(N, np.bool_)
    cur = np.empty(N, np.int64)
    order = np.empty(N, np.int64)
    ncur = 1
    cur[0] = 0
    pb[0] = 0.0
    for t in range(T):
        row = logp[t]
        no = 0
        for k in range(ncur):
            nid = cur[k]
            a = pb[nid]
            b = pnb[nid]
            tot = _lse(a, b)
            if not seen[nid]:
                seen[nid] = True
                order[no] = nid
                no += 1
            npb[nid] = _lse(npb[nid], tot + row[0])
            lc = last_cid[nid]
            if lc > 0:
                npnb[nid] = _lse(npnb[nid], b + row[lc])
            for j in range(ch_cid.shape[1]):
                cid = ch_cid[nid, j]
                if cid < 0:
                    break
                cn = ch_nid[nid, j]
                d = (a if cid == lc else tot) + row[cid]
                if not seen[cn]:
                    seen[cn] = True
                    order[no] = cn
                    no += 1
                npnb[cn] = _lse(npnb[cn], d)
        for k in range(ncur):
            nid = cur[k]
            pb[nid] = -np.inf
            pnb[nid] = -np.inf
        if no > beam_width:
            tots = np.empty(no)
            for k in range(no):
                tots[k] = _lse(npb[order[k]], npnb[order[k]])
            # stable, descending: same survivors and order as sorted(..., reverse=True)
            keep = np.argsort(-tots, kind="mergesort")[:beam_width]
            ncur = beam_width
            for k in range(beam_width):
                cur[k] = order[keep[k]]
        else:
            ncur = no
            for k in range(no):
                cur[k] = order[k]
        for k in range(ncur):
            nid = cur[k]
            pb[nid] = npb[nid]
            pnb[nid] = npnb[nid]
        for k in range(no):
            nid = order[k]
            npb[nid] = -np.inf
            npnb[nid] = -np.inf
            seen[nid] = False
    return cur[:ncur].copy(), pb, pnb
