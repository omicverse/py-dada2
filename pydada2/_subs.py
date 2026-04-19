"""Substitution objects — port of dada2/src/nwalign_endsfree.cpp helpers
``sub_new`` / ``sub_free`` / ``al2subs``.

A Sub captures the substitution map between two aligned sequences:
    - len0:  length of reference sequence (first arg)
    - map:   index_in_seq1[i] = position in seq0 (or -1 for gaps)
    - pos:   array of substitution positions (in seq0 coords)
    - nt0/nt1: the reference and query nucleotides at each substitution
    - q0/q1: rounded mean quality scores at each substitution position

This is a pure Python port driven by our `align.nwalign`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .align import nwalign
from .kmers import kmer_dist


@dataclass
class Sub:
    nsubs: int
    len0: int
    map: np.ndarray         # int32, length len0
    pos: np.ndarray         # int32, length nsubs (positions in seq0 coords)
    nt0: np.ndarray         # int8, length nsubs (1=A, 2=C, 3=G, 4=T)
    nt1: np.ndarray         # int8
    q0: Optional[np.ndarray]
    q1: Optional[np.ndarray]
    aligned0: str           # full aligned strings (with '-' gaps)
    aligned1: str


_NT2I = {"A": 1, "C": 2, "G": 3, "T": 4}


def _nt_to_int(c: str) -> int:
    return _NT2I.get(c.upper(), 0)


def al2subs(a0: str, a1: str,
            q0_seq: Optional[np.ndarray] = None,
            q1_seq: Optional[np.ndarray] = None) -> Sub:
    """Convert two aligned strings (with '-' gaps) into a Sub.

    Mirrors al2subs in nwalign_endsfree.cpp. Substitutions are positions
    where both sides have a non-gap base AND they differ. Gap positions
    are not substitutions (they are indels — DADA2's substitution model
    ignores them but the alignment is preserved).
    """
    assert len(a0) == len(a1)
    n_aln = len(a0)

    # Build seq0 length and the position map seq0->seq1
    len0 = sum(1 for c in a0 if c != "-")
    map_arr = np.full(len0, -1, dtype=np.int32)

    pos_list: List[int] = []
    nt0_list: List[int] = []
    nt1_list: List[int] = []
    q0_list: List[int] = []
    q1_list: List[int] = []

    p0 = 0  # walking index into seq0
    p1 = 0  # walking index into seq1
    for k in range(n_aln):
        c0, c1 = a0[k], a1[k]
        if c0 != "-" and c1 != "-":
            map_arr[p0] = p1
            if c0 != c1:
                pos_list.append(p0)
                nt0_list.append(_nt_to_int(c0))
                nt1_list.append(_nt_to_int(c1))
                if q0_seq is not None and q1_seq is not None:
                    q0_list.append(int(q0_seq[p0]))
                    q1_list.append(int(q1_seq[p1]))
            p0 += 1
            p1 += 1
        elif c0 == "-" and c1 != "-":
            # insertion in seq1 — advance p1 only
            p1 += 1
        elif c0 != "-" and c1 == "-":
            # deletion in seq1 — leaves map[p0] = -1
            p0 += 1
        # else: both gaps — skip

    return Sub(
        nsubs=len(pos_list),
        len0=len0,
        map=map_arr,
        pos=np.asarray(pos_list, dtype=np.int32),
        nt0=np.asarray(nt0_list, dtype=np.int8),
        nt1=np.asarray(nt1_list, dtype=np.int8),
        q0=np.asarray(q0_list, dtype=np.int32) if q0_list else None,
        q1=np.asarray(q1_list, dtype=np.int32) if q1_list else None,
        aligned0=a0,
        aligned1=a1,
    )


def sub_new(seq0: str, seq1: str,
            *, match: int = 5, mismatch: int = -4, gap_p: int = -8,
            homo_gap_p: int = 0, band: int = 16,
            use_kmers: bool = True, kdist_cutoff: float = 0.42,
            q0_seq: Optional[np.ndarray] = None,
            q1_seq: Optional[np.ndarray] = None) -> Optional[Sub]:
    """Compute the substitution between seq0 (reference / center) and seq1.

    Mirrors sub_new in nwalign_endsfree.cpp. Returns None if the kmer
    distance exceeds ``kdist_cutoff`` (the kmer pre-screen).
    """
    if use_kmers:
        kd = kmer_dist(seq0, seq1, kmer_size=5)
        if kd > kdist_cutoff:
            return None
    a0, a1 = nwalign(seq0, seq1, match=match, mismatch=mismatch,
                     gap_p=gap_p, homo_gap_p=homo_gap_p, band=band,
                     endsfree=True)
    return al2subs(a0, a1, q0_seq=q0_seq, q1_seq=q1_seq)


def compute_lambda(seq1_int: np.ndarray, qind: np.ndarray, sub: Optional[Sub],
                   err_mat: np.ndarray, use_quals: bool) -> float:
    """Lambda = product over positions of err[transition, quality].

    ``seq1_int`` is the seq encoded with A=1..T=4 (length = len(seq1)).
    ``qind`` is the quality index at each position (length = len(seq1)).
    Mirrors compute_lambda in pval.cpp.
    """
    if sub is None:
        return 0.0
    n = seq1_int.shape[0]
    # tvec[pos1] = (nti0 * 4 + nti1) where default is self-transition
    tvec = (seq1_int - 1) * 4 + (seq1_int - 1)  # nti1*4+nti1 for default
    if sub.nsubs > 0:
        # for each substitution, override tvec at the seq1 position
        for s in range(sub.nsubs):
            p0 = int(sub.pos[s])
            p1 = int(sub.map[p0])
            if p1 < 0 or p1 >= n:
                continue
            tvec[p1] = (int(sub.nt0[s]) - 1) * 4 + (int(sub.nt1[s]) - 1)
    if use_quals:
        # err_mat is (16, n_q); index by (tvec, qind)
        rates = err_mat[tvec, qind]
    else:
        rates = err_mat[tvec, 0]
    # multiply
    # use logsum for stability
    if (rates <= 0).any():
        return 0.0
    return float(np.exp(np.log(rates).sum()))
