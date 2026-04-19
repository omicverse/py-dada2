"""Bimera (two-parent chimera) detection.

Mirrors R DADA2's `isBimera`, `isBimeraDenovo`, `isBimeraDenovoTable`,
`removeBimeraDenovo`, and the C++ kernels in `src/chimera.cpp`.

A sequence is "bimeric" if its left half can be matched to one parent and
its right half to another parent, with the union of the two
ends-free-aligned coverages spanning the full sequence.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .align import nwalign
from .opts import get_dada_opt
from .io import get_uniques, get_sequences


def _get_lr(a0: str, a1: str, allow_one_off: bool, max_shift: int):
    """Mirror chimera.cpp::get_lr.

    a0 = aligned query, a1 = aligned parent. Returns (left, right,
    left_oo, right_oo) — number of bases at the left/right of the query
    that match the parent (with end-shift slack ≤ max_shift).
    """
    n = len(a0)
    pos = 0; left = 0
    while pos < n and a0[pos] == "-":
        pos += 1
    while pos < min(n, max_shift) and a1[pos] == "-":
        pos += 1
        left += 1
    while pos < n and a0[pos] == a1[pos]:
        pos += 1
        left += 1
    left_oo = left
    if allow_one_off:
        if pos < n and a0[pos] != "-":
            left_oo += 1
        pos += 1
        while pos < n and a0[pos] == a1[pos]:
            pos += 1
            left_oo += 1

    pos = n - 1; right = 0
    while pos >= 0 and a0[pos] == "-":
        pos -= 1
    while pos > (n - max_shift - 1) and pos >= 0 and a1[pos] == "-":
        pos -= 1
        right += 1
    while pos >= 0 and a0[pos] == a1[pos]:
        pos -= 1
        right += 1
    right_oo = right
    if allow_one_off:
        if pos >= 0 and a0[pos] != "-":
            right_oo += 1
        pos -= 1
        while pos >= 0 and a0[pos] == a1[pos]:
            pos -= 1
            right_oo += 1
    return left, right, left_oo, right_oo


def _ham_endsfree(a0: str, a1: str) -> int:
    n = len(a0)
    i = 0
    while i < n and (a0[i] == "-" or a1[i] == "-"):
        i += 1
    j = n - 1
    while j > i and (a0[j] == "-" or a1[j] == "-"):
        j -= 1
    return sum(1 for k in range(i, j + 1) if a0[k] != a1[k])


def is_bimera(sq: str, parents: Sequence[str], allow_one_off: bool = False,
              min_one_off_par_dist: int = 4, max_shift: int = 16) -> bool:
    """Mirror C_is_bimera (chimera.cpp::C_is_bimera).

    Returns True if `sq` can be modelled as a two-parent chimera of the
    given parents. Walks parents in order; for each one, accumulates the
    best left/right coverage seen so far. As soon as a (left+right) ≥
    len(sq) split exists, returns True.
    """
    opts = get_dada_opt()
    match = opts["MATCH"]
    mismatch = opts["MISMATCH"]
    gap_p = opts["GAP_PENALTY"]
    n = len(sq)
    max_left = max_right = 0
    oo_max_left = oo_max_right = oo_max_left_oo = oo_max_right_oo = 0

    for par in parents:
        a0, a1 = nwalign(sq, par, match=match, mismatch=mismatch, gap_p=gap_p,
                         band=max_shift, endsfree=True)
        left, right, left_oo, right_oo = _get_lr(a0, a1, allow_one_off, max_shift)
        # Toss id / pure-shift / internal-indel parents
        if (left + right) >= n:
            continue
        if left > max_left:
            max_left = left
        if right > max_right:
            max_right = right
        if allow_one_off and _ham_endsfree(a0, a1) >= min_one_off_par_dist:
            if left > oo_max_left:
                oo_max_left = left
            if right > oo_max_right:
                oo_max_right = right
            if left_oo > oo_max_left_oo:
                oo_max_left_oo = left_oo
            if right_oo > oo_max_right_oo:
                oo_max_right_oo = right_oo
        # check
        if (max_right + max_left) >= n:
            return True
        if allow_one_off:
            if (oo_max_left + oo_max_right_oo) >= n or (oo_max_left_oo + oo_max_right) >= n:
                return True
    return False


def is_bimera_denovo(unqs, min_fold_parent: float = 2, min_parent_abundance: int = 8,
                     allow_one_off: bool = False, min_one_off_par_dist: int = 4,
                     max_shift: int = 16, verbose: bool = False) -> Dict[str, bool]:
    """Mirror isBimeraDenovo: vector form of is_bimera over a uniques."""
    uni = get_uniques(unqs)
    seqs = list(uni.keys())
    abunds = np.array(list(uni.values()), dtype=np.int64)
    out = {}
    for i, s in enumerate(seqs):
        a = abunds[i]
        # parents = sequences > min_fold * a AND > min_parent_abundance
        mask = (abunds > min_fold_parent * a) & (abunds > min_parent_abundance)
        pars = [seqs[j] for j in np.where(mask)[0]]
        if len(pars) < 2:
            out[s] = False
        else:
            out[s] = is_bimera(s, pars, allow_one_off=allow_one_off,
                               min_one_off_par_dist=min_one_off_par_dist,
                               max_shift=max_shift)
    if verbose:
        n_bim = sum(1 for v in out.values() if v)
        print(f"Identified {n_bim} bimeras out of {len(out)} input sequences.")
    return out


def is_bimera_denovo_table(seqtab: pd.DataFrame, min_sample_fraction: float = 0.9,
                            ignore_n_negatives: int = 1, min_fold_parent: float = 1.5,
                            min_parent_abundance: int = 2, allow_one_off: bool = False,
                            min_one_off_par_dist: int = 4, max_shift: int = 16,
                            verbose: bool = False) -> Dict[str, bool]:
    """Mirror isBimeraDenovoTable: per-sample bimera vote then consensus."""
    sqs = list(seqtab.columns)
    nflag = np.zeros(len(sqs), dtype=np.int64)
    nsam = np.zeros(len(sqs), dtype=np.int64)

    mat = seqtab.values
    n_rows, n_cols = mat.shape
    for i in range(n_rows):
        row = mat[i]
        present = row > 0
        nsam[present] += 1
        # for each present seq j, find parents within this sample
        for j in np.where(present)[0]:
            mask = (row > min_fold_parent * row[j]) & (row >= min_parent_abundance)
            pars_idx = [k for k in np.where(mask)[0] if k != j]
            if len(pars_idx) < 2:
                continue
            pars = [sqs[k] for k in pars_idx]
            if is_bimera(sqs[j], pars, allow_one_off=allow_one_off,
                         min_one_off_par_dist=min_one_off_par_dist,
                         max_shift=max_shift):
                nflag[j] += 1

    bims = {}
    for j, s in enumerate(sqs):
        cond = (nflag[j] >= nsam[j]) or (nflag[j] > 0 and nflag[j] >= (nsam[j] - ignore_n_negatives) * min_sample_fraction)
        bims[s] = bool(cond)
    if verbose:
        print(f"Identified {sum(bims.values())} bimeras out of {len(bims)} input sequences.")
    return bims


def remove_bimera_denovo(unqs, method: str = "consensus", verbose: bool = False, **kwargs):
    """Mirror removeBimeraDenovo. Returns the input with bimeras dropped."""
    if isinstance(unqs, pd.DataFrame):
        if method == "pooled":
            # treat as a uniques summed across samples
            uni = {c: int(unqs[c].sum()) for c in unqs.columns}
            bim = is_bimera_denovo(uni, verbose=verbose, **kwargs)
            keep = [c for c in unqs.columns if not bim[c]]
            return unqs[keep]
        elif method == "consensus":
            bim = is_bimera_denovo_table(unqs, verbose=verbose, **kwargs)
            keep = [c for c in unqs.columns if not bim[c]]
            return unqs[keep]
        elif method == "per-sample":
            out = unqs.copy()
            for i in range(out.shape[0]):
                row_uni = {c: int(out.iloc[i][c]) for c in out.columns if out.iloc[i][c] > 0}
                bim = is_bimera_denovo(row_uni, verbose=False, **kwargs)
                for c, v in bim.items():
                    if v:
                        out.iloc[i, out.columns.get_loc(c)] = 0
            keep = [c for c in out.columns if out[c].sum() > 0]
            return out[keep]
        else:
            raise ValueError("method must be one of: pooled, consensus, per-sample")
    if isinstance(unqs, dict):
        bim = is_bimera_denovo(unqs, verbose=verbose, **kwargs)
        return {k: v for k, v in unqs.items() if not bim[k]}
    raise TypeError(f"Unsupported input type: {type(unqs)}")
