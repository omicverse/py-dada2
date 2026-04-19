"""Pairwise alignment — Needleman-Wunsch ends-free, Numba-accelerated.

Mirrors the C++ implementations in dada2/src/nwalign_endsfree.cpp (and
the vectorized variant). Default parameters (match=5, mismatch=-4,
gap=-8, band=16) match `setDadaOpt`.

The R DADA2 package exposes:
    nwalign(s1, s2, match=5, mismatch=-4, gap=-8, band=16, endsfree=True)
We reproduce the same byte-for-byte alignment outputs.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*a, **kw):  # noqa: D401
        def deco(f):
            return f
        return deco


# Encode A=1, C=2, G=3, T=4 (DADA2 convention; gap -> 0)
_ENC = {"A": 1, "C": 2, "G": 3, "T": 4, "N": 5, "-": 0}
_DEC = {1: "A", 2: "C", 3: "G", 4: "T", 5: "N", 0: "-"}


def _enc(s: str) -> np.ndarray:
    out = np.zeros(len(s), dtype=np.int8)
    for i, c in enumerate(s):
        out[i] = _ENC.get(c.upper(), 5)
    return out


def _dec(arr: np.ndarray) -> str:
    return "".join(_DEC.get(int(x), "N") for x in arr)


@njit(cache=True)
def _nw_endsfree_core(s1: np.ndarray, s2: np.ndarray,
                      match: int, mismatch: int,
                      gap_p: int, end_gap_p: int,
                      band: int) -> Tuple[np.ndarray, np.ndarray]:
    """Banded Needleman-Wunsch with ends-free gaps.

    Mirrors nwalign_endsfree.cpp. ``band`` is the half-band; if band <= 0
    the full N-W matrix is computed.

    Returns two int8 arrays = aligned s1, aligned s2 (same length).
    """
    n1 = s1.shape[0]
    n2 = s2.shape[0]

    # Score matrix (n1+1) x (n2+1). Use full matrix; banding pruning logic
    # mirrors the C++ inner-loop bounds. NEG = very negative sentinel.
    NEG = -32000
    H = np.full((n1 + 1, n2 + 1), NEG, dtype=np.int32)
    # Traceback: 0=stop,1=diag,2=up(gap in s2),3=left(gap in s1)
    T = np.zeros((n1 + 1, n2 + 1), dtype=np.int8)
    H[0, 0] = 0
    # ends-free: row 0 and col 0 get end_gap_p (= 0 for the typical
    # endsfree call) instead of gap_p.
    for j in range(1, n2 + 1):
        H[0, j] = j * end_gap_p
        T[0, j] = 3
    for i in range(1, n1 + 1):
        H[i, 0] = i * end_gap_p
        T[i, 0] = 2

    if band <= 0:
        band = max(n1, n2) + 1

    for i in range(1, n1 + 1):
        # banded inner-loop bounds
        jmin = max(1, i - band)
        jmax = min(n2, i + band)
        for j in range(jmin, jmax + 1):
            sij = match if s1[i - 1] == s2[j - 1] else mismatch
            d = H[i - 1, j - 1] + sij
            # vertical (insertion in s2 / deletion in s1) — penalty depends on edge
            if j == n2:
                u = H[i - 1, j] + end_gap_p
            else:
                u = H[i - 1, j] + gap_p
            if i == n1:
                left = H[i, j - 1] + end_gap_p
            else:
                left = H[i, j - 1] + gap_p
            best = d
            t = 1
            if u > best:
                best = u
                t = 2
            if left > best:
                best = left
                t = 3
            H[i, j] = best
            T[i, j] = t

    # Traceback from bottom-right.
    i, j = n1, n2
    a1 = []
    a2 = []
    while i > 0 or j > 0:
        t = T[i, j]
        if t == 1:
            a1.append(s1[i - 1])
            a2.append(s2[j - 1])
            i -= 1
            j -= 1
        elif t == 2:
            a1.append(s1[i - 1])
            a2.append(np.int8(0))
            i -= 1
        elif t == 3:
            a1.append(np.int8(0))
            a2.append(s2[j - 1])
            j -= 1
        else:
            # boundary fallback — should not normally happen given init
            if i > 0:
                a1.append(s1[i - 1])
                a2.append(np.int8(0))
                i -= 1
            else:
                a1.append(np.int8(0))
                a2.append(s2[j - 1])
                j -= 1
    a1_arr = np.empty(len(a1), dtype=np.int8)
    a2_arr = np.empty(len(a2), dtype=np.int8)
    L = len(a1)
    for k in range(L):
        a1_arr[k] = a1[L - 1 - k]
        a2_arr[k] = a2[L - 1 - k]
    return a1_arr, a2_arr


def nwalign(s1: str, s2: str,
            match: int = 5, mismatch: int = -4,
            gap_p: int = -8, homo_gap_p: int = 0,
            band: int = 16, endsfree: bool = True) -> Tuple[str, str]:
    """Banded Needleman-Wunsch alignment.

    Returns (aligned_s1, aligned_s2) as strings with '-' gaps. Mirrors
    R DADA2's `nwalign(s1, s2, match=5, mismatch=-4, gap=-8, band=16, endsfree=TRUE)`.
    """
    end_gap = 0 if endsfree else gap_p
    a1, a2 = _nw_endsfree_core(_enc(s1), _enc(s2), match, mismatch,
                               gap_p, end_gap, band)
    return _dec(a1), _dec(a2)


def nwhamming(s1: str, s2: str, **kwargs) -> int:
    """Pairwise hamming distance after NW alignment, ignoring end gaps."""
    a1, a2 = nwalign(s1, s2, **kwargs)
    # Strip leading/trailing positions where either side is gap (ends-free).
    n = len(a1)
    i = 0
    while i < n and (a1[i] == "-" or a2[i] == "-"):
        i += 1
    j = n - 1
    while j > i and (a1[j] == "-" or a2[j] == "-"):
        j -= 1
    h = 0
    for k in range(i, j + 1):
        if a1[k] != a2[k]:
            h += 1
    return h


def rc(seq: str) -> str:
    """Reverse-complement (mirrors R's rc())."""
    table = str.maketrans("ACGTNRYSWKMBDHVacgtnryswkmbdhv-",
                          "TGCANYRSWMKVHDBtgcanyrswmkvhdb-")
    return seq.translate(table)[::-1]
