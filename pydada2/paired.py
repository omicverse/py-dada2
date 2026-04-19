"""Paired-end merging.

Mirrors R DADA2's mergePairs / C_eval_pair / C_pair_consensus.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .align import nwalign, rc
from .io import DerepObject, derep_fastq


def eval_pair(s1: str, s2: str) -> Tuple[int, int, int]:
    """Mirror C_eval_pair: count (matches, mismatches, indels) over the
    *internal* (non-end-gap) alignment region of two aligned strings.
    """
    n = len(s1)
    assert n == len(s2)
    # find internal start
    i = 0
    while i < n and (s1[i] == "-" or s2[i] == "-"):
        i += 1
    j = n - 1
    while j > i and (s1[j] == "-" or s2[j] == "-"):
        j -= 1
    matches = mismatches = indels = 0
    for k in range(i, j + 1):
        if s1[k] == "-" or s2[k] == "-":
            indels += 1
        elif s1[k] == s2[k]:
            matches += 1
        else:
            mismatches += 1
    return matches, mismatches, indels


def pair_consensus(s1: str, s2: str, prefer: int = 1, trim_overhang: bool = False) -> str:
    """Mirror C_pair_consensus: build a merged sequence from two aligned strings.

    For positions inside the overlap, prefer the side indicated by `prefer`
    (1=forward, 2=reverse). For positions where one side is gap and the
    other isn't (overhang), use whichever has a base — unless trim_overhang.
    """
    n = len(s1)
    assert n == len(s2)
    # Determine internal region
    i = 0
    while i < n and (s1[i] == "-" or s2[i] == "-"):
        i += 1
    j = n - 1
    while j > i and (s1[j] == "-" or s2[j] == "-"):
        j -= 1

    out_chars: List[str] = []
    for k in range(n):
        c1, c2 = s1[k], s2[k]
        in_overlap = (i <= k <= j)
        if in_overlap:
            if c1 == "-":
                out_chars.append(c2)
            elif c2 == "-":
                out_chars.append(c1)
            else:
                out_chars.append(c1 if prefer == 1 else c2)
        else:
            if trim_overhang:
                continue
            if c1 != "-":
                out_chars.append(c1)
            elif c2 != "-":
                out_chars.append(c2)
    return "".join(c for c in out_chars if c != "-")


def merge_pairs(
    dadaF, derepF, dadaR, derepR,
    minOverlap: int = 12,                 # noqa: N803
    maxMismatch: int = 0,                 # noqa: N803
    returnRejects: bool = False,          # noqa: N803
    propagateCol: Sequence[str] = (),     # noqa: N803
    justConcatenate: bool = False,        # noqa: N803
    trimOverhang: bool = False,           # noqa: N803
    verbose: bool = False,
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """Merge denoised forward + reverse reads. See R mergePairs."""
    # normalise to lists
    if isinstance(dadaF, dict):
        dadaF = [dadaF]
    if isinstance(dadaR, dict):
        dadaR = [dadaR]
    if isinstance(derepF, (str, DerepObject)):
        derepF = [derepF]
    if isinstance(derepR, (str, DerepObject)):
        derepR = [derepR]
    n = len(dadaF)
    assert len(derepF) == len(dadaR) == len(derepR) == n, "F/R lists must match length"

    out = []
    for i in range(n):
        dF = dadaF[i]; dR = dadaR[i]
        drpF = derepF[i] if isinstance(derepF[i], DerepObject) else derep_fastq(derepF[i])
        drpR = derepR[i] if isinstance(derepR[i], DerepObject) else derep_fastq(derepR[i])

        mapF = drpF.map  # read -> unique idx
        mapR = drpR.map
        rF = dF["map"][mapF]  # read -> cluster idx (forward)
        rR = dR["map"][mapR]
        # form pairs
        df = pd.DataFrame({"forward": rF, "reverse": rR})
        keep = (df["forward"] >= 0) & (df["reverse"] >= 0)
        df = df[keep]
        if df.empty:
            cols = ["sequence", "abundance", "forward", "reverse",
                    "nmatch", "nmismatch", "nindel", "prefer", "accept"]
            out.append(pd.DataFrame(columns=cols))
            continue

        # unique forward/reverse pairs
        ups = df.groupby(["forward", "reverse"]).size().reset_index(name="abundance")

        Funq = [dF["clustering"][int(f)]["sequence"] for f in ups["forward"]]
        Runq = [rc(dR["clustering"][int(r)]["sequence"]) for r in ups["reverse"]]

        nmatch = []; nmm = []; nind = []; pref = []; acc = []; seqs = []
        for f, r, abu in zip(Funq, Runq, ups["abundance"]):
            if justConcatenate:
                seqs.append(f + "NNNNNNNNNN" + r)
                nmatch.append(0); nmm.append(0); nind.append(0)
                pref.append(0); acc.append(True)
                continue
            # heavy mismatch/gap penalty for tighter overlap
            if maxMismatch == 0:
                aF, aR = nwalign(f, r, match=1, mismatch=-64, gap_p=-64,
                                 band=-1, endsfree=True)
            else:
                aF, aR = nwalign(f, r, match=1, mismatch=-8, gap_p=-8,
                                 band=-1, endsfree=True)
            m, mm, ind = eval_pair(aF, aR)
            nmatch.append(m); nmm.append(mm); nind.append(ind)
            # prefer 2 if R cluster is more abundant (n0)
            f_idx = int(ups["forward"].iloc[len(seqs)])
            r_idx = int(ups["reverse"].iloc[len(seqs)])
            f_n0 = dF["clustering"][f_idx]["n0"]
            r_n0 = dR["clustering"][r_idx]["n0"]
            prefer = 2 if r_n0 > f_n0 else 1
            pref.append(prefer)
            accept = (m >= minOverlap) and ((mm + ind) <= maxMismatch)
            acc.append(accept)
            seqs.append(pair_consensus(aF, aR, prefer=prefer, trim_overhang=trimOverhang) if accept else "")

        ups["sequence"] = seqs
        ups["nmatch"] = nmatch
        ups["nmismatch"] = nmm
        ups["nindel"] = nind
        ups["prefer"] = pref
        ups["accept"] = acc

        for col in propagateCol:
            ups[f"F.{col}"] = [dF["clustering"][int(f)].get(col) for f in ups["forward"]]
            ups[f"R.{col}"] = [dR["clustering"][int(r)].get(col) for r in ups["reverse"]]

        ups = ups.sort_values("abundance", ascending=False).reset_index(drop=True)
        if not returnRejects:
            ups = ups[ups["accept"]].reset_index(drop=True)
        out.append(ups)
        if verbose:
            print(f"{ups[ups['accept']]['abundance'].sum()} paired-reads "
                  f"(in {ups['accept'].sum()} unique pairings) successfully merged.")

    if len(out) == 1:
        return out[0]
    return out
