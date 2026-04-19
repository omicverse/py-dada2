"""R-parity tests for py-dada2.

These run R DADA2 (CMAP env) and py-dada2 on the same input fastq and
compare ASV identity, abundances, dereplication, etc.

Skip if the R reference TSV is missing — generate it via:

    /scratch/users/steorra/env/CMAP/bin/Rscript tests/r_reference.R
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pydada2
from pydada2 import derep_fastq, dada


HERE = Path(__file__).parent
DATA = Path("/scratch/users/steorra/analysis/omicverse_dev/cache/dada2_src/inst/extdata")
R_OUT = HERE / "r_out"


def _flat_err():
    err = np.full((16, 41), 1e-3)
    for i in range(4):
        err[i * 4 + i] = 1 - 3 * 1e-3
    return err


@pytest.fixture(scope="module")
def r_ref():
    f = R_OUT / "dada_sam1F_flat_err.tsv"
    if not f.exists():
        pytest.skip(f"R reference not generated yet at {f}")
    return pd.read_csv(f, sep="\t")


def test_derep_matches_r():
    f = R_OUT / "derep_sam1F.tsv"
    if not f.exists():
        pytest.skip("R derep reference not generated yet.")
    rdf = pd.read_csv(f, sep="\t")
    pdr = derep_fastq(str(DATA / "sam1F.fastq.gz"))
    pdf = pd.DataFrame({"sequence": list(pdr.uniques.keys()),
                        "abundance": list(pdr.uniques.values())})
    # Compare as dict keyed by sequence
    rd = dict(zip(rdf["sequence"], rdf["abundance"]))
    pd_ = dict(zip(pdf["sequence"], pdf["abundance"]))
    assert rd == pd_, "derep mismatch between R and py"


def test_dada_cluster_count_matches_r(r_ref):
    pdr = derep_fastq(str(DATA / "sam1F.fastq.gz"))
    res = dada(pdr, err=_flat_err(), verbose=False)
    assert len(res["clustering"]) == len(r_ref), \
        f"cluster count: py={len(res['clustering'])} R={len(r_ref)}"


def test_dada_top_asv_sequence_matches_r(r_ref):
    """The top ASV (most-abundant cluster) should match exactly."""
    pdr = derep_fastq(str(DATA / "sam1F.fastq.gz"))
    res = dada(pdr, err=_flat_err(), verbose=False)
    py_top = max(res["clustering"], key=lambda c: c["abundance"])["sequence"]
    r_top = r_ref.loc[r_ref["abundance"].idxmax(), "sequence"]
    assert py_top == r_top, f"top ASV differs: py={py_top[:30]} R={r_top[:30]}"


def test_dada_all_asv_sequences_present(r_ref):
    """Every R ASV should appear in py-dada2 output (and vice-versa)."""
    pdr = derep_fastq(str(DATA / "sam1F.fastq.gz"))
    res = dada(pdr, err=_flat_err(), verbose=False)
    py_seqs = {c["sequence"] for c in res["clustering"]}
    r_seqs = set(r_ref["sequence"])
    only_r = r_seqs - py_seqs
    only_py = py_seqs - r_seqs
    assert not only_r and not only_py, \
        f"ASV set differs. Only-in-R: {len(only_r)}, only-in-py: {len(only_py)}"


def test_dada_abundances_match_r(r_ref):
    pdr = derep_fastq(str(DATA / "sam1F.fastq.gz"))
    res = dada(pdr, err=_flat_err(), verbose=False)
    py_map = {c["sequence"]: c["abundance"] for c in res["clustering"]}
    r_map = dict(zip(r_ref["sequence"], r_ref["abundance"]))
    diffs = []
    for s, ra in r_map.items():
        pa = py_map.get(s)
        if pa is None or pa != ra:
            diffs.append((s[:20] + "…", ra, pa))
    assert not diffs, f"{len(diffs)} ASV abundance mismatches: {diffs[:5]}"
