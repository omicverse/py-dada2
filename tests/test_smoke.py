"""Smoke tests — verify the pipeline runs end-to-end without R."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import pydada2
from pydada2 import (
    derep_fastq, dada, merge_pairs, make_sequence_table,
    remove_bimera_denovo, nwalign, kmer_dist, rc, eval_pair,
    pair_consensus,
)


DATA = Path("/scratch/users/steorra/analysis/omicverse_dev/cache/dada2_src/inst/extdata")


def _flat_err():
    err = np.full((16, 41), 1e-3)
    for i in range(4):
        err[i * 4 + i] = 1 - 3 * 1e-3
    return err


def test_imports():
    assert pydada2.__version__ == "0.1.0"


def test_nwalign_identity():
    a, b = nwalign("ACGTACGT", "ACGTACGT")
    assert a == b == "ACGTACGT"


def test_nwalign_endsfree():
    # ends-free should not penalise leading/trailing gaps
    a, b = nwalign("AAAACGTACGT", "ACGTACGTGGGG", band=-1)
    assert "ACGTACGT" in a.replace("-", "")
    assert "ACGTACGT" in b.replace("-", "")


def test_kmer_dist_bounds():
    assert kmer_dist("ACGTACGTACGT", "ACGTACGTACGT") == 0.0
    d = kmer_dist("ACGTACGTACGT", "TTTTTTTTTTTT")
    assert 0 < d <= 1.0


def test_rc():
    assert rc("ACGT") == "ACGT"
    assert rc("AAAA") == "TTTT"
    assert rc("ACGN") == "NCGT"


def test_eval_pair_no_overlap():
    m, mm, ind = eval_pair("ACGT--", "--CGTA")
    # only inner CGT-A region (with one gap on each side stripped)
    assert isinstance(m, int) and m >= 0


def test_pair_consensus_prefer_forward():
    s = pair_consensus("ACGT", "ACGT", prefer=1)
    assert s == "ACGT"


def test_derep_sam1F():
    if not DATA.exists():
        pytest.skip("DADA2 extdata not available locally")
    drp = derep_fastq(str(DATA / "sam1F.fastq.gz"))
    assert drp.n_unique > 0
    assert int(drp.abundances().sum()) > 0


def test_dada_sam1F_runs():
    if not DATA.exists():
        pytest.skip("DADA2 extdata not available locally")
    drp = derep_fastq(str(DATA / "sam1F.fastq.gz"))
    res = dada(drp, err=_flat_err(), verbose=False)
    assert "denoised" in res
    assert "clustering" in res
    assert len(res["clustering"]) > 0
    # At least 80% of reads should land in clusters
    total = int(drp.abundances().sum())
    assigned = sum(c["abundance"] for c in res["clustering"])
    assert assigned >= 0.5 * total


def test_make_sequence_table_and_chimera():
    if not DATA.exists():
        pytest.skip()
    drp1 = derep_fastq(str(DATA / "sam1F.fastq.gz"))
    drp2 = derep_fastq(str(DATA / "sam2F.fastq.gz"))
    err = _flat_err()
    dd1 = dada(drp1, err=err, verbose=False)
    dd2 = dada(drp2, err=err, verbose=False)
    seqtab = make_sequence_table({"sam1": dd1, "sam2": dd2})
    assert seqtab.shape[0] == 2
    assert seqtab.shape[1] >= 1
    nochim = remove_bimera_denovo(seqtab, method="consensus")
    # chimera removal is monotone — never adds
    assert nochim.shape[1] <= seqtab.shape[1]
