"""k-mer distance pre-screen.

Mirrors dada2/src/kmers.cpp. Default k=5 (KMER_SIZE). The k-mer distance
is the L1 distance of the per-kmer count vectors normalised by the
shorter sequence length, then 1 - similarity.

R interface:
    kmer_dist(s1, s2, kmer_size)        -> double
    kord_dist(s1, s2, kmer_size, SSE)   -> double
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*a, **kw):
        def deco(f):
            return f
        return deco


_NT2I = {"A": 0, "C": 1, "G": 2, "T": 3}


def _kmer_vec(seq: str, k: int) -> np.ndarray:
    n_kmer = 1 << (2 * k)
    out = np.zeros(n_kmer, dtype=np.uint16)
    n = len(seq)
    if n < k:
        return out
    idx = 0
    valid = 0
    mask = n_kmer - 1
    for i, c in enumerate(seq):
        v = _NT2I.get(c.upper(), -1)
        if v < 0:
            valid = 0
            idx = 0
            continue
        idx = ((idx << 2) | v) & mask
        valid += 1
        if valid >= k:
            out[idx] += 1
    return out


@njit(cache=True)
def _kmer_dist(kv1: np.ndarray, len1: int, kv2: np.ndarray, len2: int, k: int) -> float:
    # Mirrors kmer_dist in kmers.cpp:
    #   shared = sum(min(kv1[i], kv2[i]))
    #   dist = 1 - shared / (min(len1, len2) - k + 1)
    n_kmer = kv1.shape[0]
    shared = 0
    for i in range(n_kmer):
        a = kv1[i]
        b = kv2[i]
        shared += a if a < b else b
    denom = (len1 if len1 < len2 else len2) - k + 1
    if denom <= 0:
        return 1.0
    return 1.0 - shared / denom


def kmer_dist(s1: str, s2: str, kmer_size: int = 5) -> float:
    kv1 = _kmer_vec(s1, kmer_size)
    kv2 = _kmer_vec(s2, kmer_size)
    return _kmer_dist(kv1, len(s1), kv2, len(s2), kmer_size)


def kord_dist(s1: str, s2: str, kmer_size: int = 5, SSE: int = 0) -> float:  # noqa: N803
    """Ordered kmer distance (also from kmers.cpp).

    Same value as kmer_dist when sequences have no repeated kmers — for
    DADA2's purpose of a fast kmer pre-screen the unordered count is what
    enters the cutoff (`KDIST_CUTOFF=0.42`). Implemented as a passthrough
    here for API parity.
    """
    return kmer_dist(s1, s2, kmer_size=kmer_size)


def kmer_matches(s1: str, s2: str, kmer_size: int = 5) -> int:
    """Number of kmers shared between s1 and s2 (sum of min counts)."""
    kv1 = _kmer_vec(s1, kmer_size)
    kv2 = _kmer_vec(s2, kmer_size)
    return int(np.minimum(kv1, kv2).sum())
