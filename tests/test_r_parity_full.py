"""Full R-parity over the four canonical fixtures (sam1F/R, sam2F/R) plus
paired merging.

Run R reference first:
    /scratch/users/steorra/env/CMAP/bin/Rscript tests/r_reference_full.R
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pydada2
from pydada2 import derep_fastq, dada, merge_pairs


HERE = Path(__file__).parent
DATA = Path("/scratch/users/steorra/analysis/omicverse_dev/cache/dada2_src/inst/extdata")
R_OUT = HERE / "r_out"


def _flat_err():
    err = np.full((16, 41), 1e-3)
    for i in range(4):
        err[i * 4 + i] = 1 - 3 * 1e-3
    return err


def _read_ref(stem: str):
    f = R_OUT / f"{stem}.tsv"
    if not f.exists():
        pytest.skip(f"{f} not generated")
    return pd.read_csv(f, sep="\t")


@pytest.mark.parametrize("sample,side", [
    ("sam1", "F"), ("sam1", "R"),
    ("sam2", "F"), ("sam2", "R"),
])
def test_dada_per_sample_matches_r(sample, side):
    rdf = _read_ref(f"dada_{sample}{side}")
    pdr = derep_fastq(str(DATA / f"{sample}{side}.fastq.gz"))
    res = dada(pdr, err=_flat_err(), verbose=False)
    py_map = {c["sequence"]: c["abundance"] for c in res["clustering"]}
    r_map = dict(zip(rdf["sequence"], rdf["abundance"]))
    assert set(py_map) == set(r_map), \
        f"{sample}{side}: ASV set differs (py={len(py_map)} R={len(r_map)})"
    diffs = {s: (r_map[s], py_map[s]) for s in r_map if py_map[s] != r_map[s]}
    assert not diffs, f"{sample}{side}: abundance diffs: {list(diffs.items())[:3]}"


@pytest.mark.parametrize("sample", ["sam1", "sam2"])
def test_merge_pairs_matches_r(sample):
    rdf = _read_ref(f"merge_{sample}")
    pdrF = derep_fastq(str(DATA / f"{sample}F.fastq.gz"))
    pdrR = derep_fastq(str(DATA / f"{sample}R.fastq.gz"))
    err = _flat_err()
    ddF = dada(pdrF, err=err, verbose=False)
    ddR = dada(pdrR, err=err, verbose=False)
    pmerged = merge_pairs(ddF, pdrF, ddR, pdrR,
                          minOverlap=12, maxMismatch=0, verbose=False)
    # Compare the (sequence, abundance) pairs by content
    r_pairs = sorted(zip(rdf["sequence"], rdf["abundance"]))
    p_pairs = sorted(zip(pmerged["sequence"], pmerged["abundance"]))
    assert len(r_pairs) == len(p_pairs), \
        f"{sample}: merged-pair count: py={len(p_pairs)} R={len(r_pairs)}"
    # Sequences must match exactly
    assert {s for s, _ in r_pairs} == {s for s, _ in p_pairs}, \
        f"{sample}: merged sequences differ"
    # Abundances per sequence
    rmap = dict(r_pairs); pmap = dict(p_pairs)
    diffs = [(s[:20] + "…", rmap[s], pmap[s]) for s in rmap if rmap[s] != pmap[s]]
    assert not diffs, f"{sample}: merged abundance diffs: {diffs[:3]}"
