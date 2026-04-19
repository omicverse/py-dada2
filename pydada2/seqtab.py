"""ASV sequence table assembly.

Mirrors R DADA2's `makeSequenceTable`, `mergeSequenceTables`, and
`collapseNoMismatch`.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Union

import numpy as np
import pandas as pd

from .align import nwalign
from .io import get_uniques


def make_sequence_table(samples: Union[Mapping, Sequence],
                        orderBy: str = "abundance") -> pd.DataFrame:  # noqa: N803
    """Mirror makeSequenceTable.

    Build a (n_samples × n_ASVs) integer DataFrame whose column names are
    the ASV sequences and whose row names are sample names.
    """
    # normalise input → dict of name -> uniques
    if isinstance(samples, pd.DataFrame):
        return samples
    if isinstance(samples, dict) and "sequence" in samples and "abundance" in samples:
        # single dada result-like
        samples = {"sample": dict(zip(samples["sequence"], samples["abundance"]))}
    elif isinstance(samples, list):
        samples = {f"sample_{i}": get_uniques(s) for i, s in enumerate(samples)}
    elif isinstance(samples, dict):
        samples = {k: get_uniques(v) for k, v in samples.items()}

    all_seqs: List[str] = []
    seen = set()
    for u in samples.values():
        for s in u:
            if s not in seen:
                all_seqs.append(s)
                seen.add(s)

    mat = np.zeros((len(samples), len(all_seqs)), dtype=np.int64)
    col_idx = {s: i for i, s in enumerate(all_seqs)}
    for r, (_, u) in enumerate(samples.items()):
        for s, c in u.items():
            mat[r, col_idx[s]] = c
    df = pd.DataFrame(mat, index=list(samples.keys()), columns=all_seqs)
    if orderBy == "abundance":
        col_order = df.sum(axis=0).sort_values(ascending=False).index
        df = df[col_order]
    return df


def merge_sequence_tables(*tabs: pd.DataFrame, repeats: str = "error") -> pd.DataFrame:
    """Mirror mergeSequenceTables: combine tables with potentially overlapping
    columns (sequences) and rows (samples)."""
    out = pd.concat(tabs, axis=0, sort=False).fillna(0).astype(np.int64)
    if out.index.has_duplicates:
        if repeats == "error":
            raise ValueError("Duplicated sample names across tables.")
        elif repeats == "sum":
            out = out.groupby(out.index).sum()
    return out


def collapse_no_mismatch(seqtab: pd.DataFrame, minOverlap: int = 20,  # noqa: N803
                         orderBy: str = "abundance",                  # noqa: N803
                         identicalOnly: bool = False,                 # noqa: N803
                         verbose: bool = False) -> pd.DataFrame:
    """Mirror collapseNoMismatch.

    Collapse sequences that are identical up to overall shifts (i.e., one
    is contained in another with no mismatches over an overlap of at
    least ``minOverlap``).
    """
    seqs = list(seqtab.columns)
    abunds = seqtab.sum(axis=0).values
    order = np.argsort(-abunds)
    seqs_ordered = [seqs[i] for i in order]
    parent_of: Dict[str, str] = {}

    for i, s in enumerate(seqs_ordered):
        if identicalOnly:
            for p in seqs_ordered[:i]:
                if s == p:
                    parent_of[s] = p
                    break
            continue
        # check if s is a no-mismatch shift of any earlier (more-abundant) seq
        for p in seqs_ordered[:i]:
            a, b = nwalign(s, p, band=-1, endsfree=True)
            mm = ind = 0
            n = len(a); k0 = 0; k1 = n - 1
            while k0 < n and (a[k0] == "-" or b[k0] == "-"):
                k0 += 1
            while k1 > k0 and (a[k1] == "-" or b[k1] == "-"):
                k1 -= 1
            match = 0
            for k in range(k0, k1 + 1):
                if a[k] == "-" or b[k] == "-":
                    ind += 1
                elif a[k] == b[k]:
                    match += 1
                else:
                    mm += 1
            if mm == 0 and ind == 0 and match >= minOverlap:
                parent_of[s] = p
                break

    # Aggregate by parent
    seq_groups: Dict[str, List[str]] = {}
    for s in seqs:
        p = parent_of.get(s, s)
        seq_groups.setdefault(p, []).append(s)

    cols = list(seq_groups.keys())
    new_mat = np.zeros((seqtab.shape[0], len(cols)), dtype=np.int64)
    for j, p in enumerate(cols):
        for s in seq_groups[p]:
            new_mat[:, j] += seqtab[s].values
    out = pd.DataFrame(new_mat, index=seqtab.index, columns=cols)
    if orderBy == "abundance":
        co = out.sum(axis=0).sort_values(ascending=False).index
        out = out[co]
    if verbose:
        print(f"Collapsed {seqtab.shape[1]} -> {out.shape[1]} unique sequences.")
    return out
