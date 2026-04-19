"""Quality filtering and trimming.

Mirrors R DADA2 ``filterAndTrim``, ``fastqFilter``, ``fastqPairedFilter``.

Streams reads through a pipeline:
    enforce maxLen → trim left/right → trim by truncQ → enforce truncLen
    → enforce minLen → reject by maxN/minQ/maxEE → optional PhiX removal
    → optional low-complexity removal.

Important parity points with the R reference:
- ``trim_left=N`` means drop the first N bases (start = max(1, N+1)).
- ``trunc_q`` works on the *first* position with q <= truncQ (greedy left-to-right).
- ``max_ee`` uses EE = sum(10^(-Q/10)).
"""
from __future__ import annotations

import gzip
import io
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .io import _iter_fastq, _open_fastq, _phred_offset


def _open_out(path: Union[str, Path], compress: bool):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if compress and not str(p).endswith(".gz"):
        p = Path(str(p) + ".gz")
    if str(p).endswith(".gz"):
        return io.TextIOWrapper(gzip.open(p, "wb"), encoding="ascii", newline="\n")
    return open(p, "w", encoding="ascii", newline="\n")


def _expected_errors(qual_arr: np.ndarray) -> float:
    """sum(10^(-Q/10)) — matches R's ``C_matrixEE`` per-row."""
    return float(np.sum(np.power(10.0, -qual_arr / 10.0)))


def _trim_tail_truncq(seq: str, qual: np.ndarray, trunc_q: int) -> Tuple[str, np.ndarray]:
    """Trim seq/qual at the *first* position with quality <= trunc_q.

    Mirrors ShortRead::trimTails(., k=1, a=truncQ): scan from the left,
    cut at the first base with quality at or below the threshold.
    """
    if trunc_q <= 0:
        return seq, qual
    bad = np.where(qual <= trunc_q)[0]
    if bad.size == 0:
        return seq, qual
    cut = int(bad[0])
    return seq[:cut], qual[:cut]


def _matches_iupac(seq: str, primer: str, max_mismatch: int = 0) -> int:
    """Find first index where ``primer`` matches ``seq`` allowing IUPAC ambiguous
    codes in the primer. Returns -1 if no match.
    """
    iupac = {
        "A": "A", "C": "C", "G": "G", "T": "T",
        "R": "AG", "Y": "CT", "S": "CG", "W": "AT",
        "K": "GT", "M": "AC", "B": "CGT", "D": "AGT",
        "H": "ACT", "V": "ACG", "N": "ACGT",
    }
    n, m = len(seq), len(primer)
    if m > n:
        return -1
    for i in range(n - m + 1):
        mm = 0
        for j in range(m):
            if seq[i + j] not in iupac.get(primer[j], primer[j]):
                mm += 1
                if mm > max_mismatch:
                    break
        else:
            if mm <= max_mismatch:
                return i
    return -1


def fastq_filter(
    fn: Union[str, Path],
    fout: Union[str, Path],
    truncQ: int = 2,                   # noqa: N803
    truncLen: int = 0,                 # noqa: N803
    maxLen: float = float("inf"),      # noqa: N803
    minLen: int = 20,                  # noqa: N803
    trimLeft: int = 0,                 # noqa: N803
    trimRight: int = 0,                # noqa: N803
    maxN: int = 0,                     # noqa: N803
    minQ: int = 0,                     # noqa: N803
    maxEE: float = float("inf"),       # noqa: N803
    rm_phix: bool = True,
    rm_lowcomplex: float = 0,
    orient_fwd: Optional[str] = None,
    qualityType: str = "Auto",         # noqa: N803
    compress: bool = True,
    verbose: bool = False,
) -> Tuple[int, int]:
    """Filter & trim a single fastq. Returns (reads_in, reads_out).

    See R/filter.R::fastqFilter for parameter semantics.
    """
    if str(fn) == str(fout):
        raise ValueError("Input and output paths must differ.")

    offset = _phred_offset(qualityType, fn)
    start = max(1, trimLeft + 1)  # 1-indexed
    end = truncLen if truncLen and truncLen > 0 else None

    # Delete fout if exists (R behaviour: removes before writing)
    fout_p = Path(fout)
    if compress and not str(fout_p).endswith(".gz"):
        fout_p = Path(str(fout_p) + ".gz")
    if fout_p.exists():
        fout_p.unlink()

    inseqs = 0
    outseqs = 0

    with _open_out(fout, compress=compress) as wh:
        for h, s, q in _iter_fastq(fn):
            inseqs += 1
            qarr = np.frombuffer(q.encode("ascii"), dtype=np.uint8).astype(np.int64) - offset

            # maxLen — enforced on raw read
            if len(s) > maxLen:
                continue

            # trim left
            if start > 1:
                if len(s) < start:
                    continue
                s = s[start - 1:]
                qarr = qarr[start - 1:]

            # trim right (drop trailing trimRight bases)
            if trimRight > 0:
                if len(s) <= trimRight:
                    continue
                s = s[:-trimRight]
                qarr = qarr[:-trimRight]

            # trunc Q (left-most below threshold)
            s, qarr = _trim_tail_truncq(s, qarr, truncQ)

            # truncLen (post-trims)
            if end is not None:
                if len(s) < end:
                    continue
                s = s[:end]
                qarr = qarr[:end]

            # minLen
            if len(s) < minLen:
                continue

            # maxN
            if s.count("N") > maxN:
                continue

            # minQ
            if minQ > truncQ and (qarr.min() < minQ):
                continue

            # maxEE
            if np.isfinite(maxEE):
                if _expected_errors(qarr) > maxEE:
                    continue

            # PhiX (lazy import to avoid requiring biopython if unused)
            if rm_phix and _is_phix(s):
                continue

            # low-complexity
            if rm_lowcomplex > 0:
                if _seq_complexity(s) < rm_lowcomplex:
                    continue

            outseqs += 1
            new_q = bytes((qarr.clip(0, 93) + offset).astype(np.uint8))
            wh.write(f"{h}\n{s}\n+\n{new_q.decode('ascii')}\n")

    if outseqs == 0:
        if fout_p.exists():
            fout_p.unlink()
        if verbose:
            print(f"The filter removed all reads: {fout_p} not written.")

    if verbose:
        pct = 100 * outseqs / max(1, inseqs)
        print(f"Read in {inseqs}, output {outseqs} ({pct:.1f}%) filtered sequences.")

    return inseqs, outseqs


def fastq_paired_filter(
    fn: Sequence[Union[str, Path]],
    fout: Sequence[Union[str, Path]],
    maxN=(0, 0),                       # noqa: N803
    truncQ=(2, 2),                     # noqa: N803
    truncLen=(0, 0),                   # noqa: N803
    maxLen=(float("inf"), float("inf")),  # noqa: N803
    minLen=(20, 20),                   # noqa: N803
    trimLeft=(0, 0),                   # noqa: N803
    trimRight=(0, 0),                  # noqa: N803
    minQ=(0, 0),                       # noqa: N803
    maxEE=(float("inf"), float("inf")),  # noqa: N803
    rm_phix=(True, True),
    rm_lowcomplex=(0, 0),
    matchIDs: bool = False,            # noqa: N803
    qualityType: str = "Auto",         # noqa: N803
    compress: bool = True,
    verbose: bool = False,
) -> Tuple[int, int]:
    """Paired-end filter. Reads must be in matching order in fn[0] and fn[1]
    (R's default behaviour with matchIDs=False).
    """
    fn = list(fn)
    fout = list(fout)
    assert len(fn) == 2 and len(fout) == 2

    offF = _phred_offset(qualityType, fn[0])
    offR = _phred_offset(qualityType, fn[1])

    fout_p = [Path(f) for f in fout]
    if compress:
        fout_p = [Path(str(p) + ".gz") if not str(p).endswith(".gz") else p for p in fout_p]
    for p in fout_p:
        if p.exists():
            p.unlink()
    for p in fout_p:
        p.parent.mkdir(parents=True, exist_ok=True)

    iterF = _iter_fastq(fn[0])
    iterR = _iter_fastq(fn[1])

    inseqs = 0
    outseqs = 0

    def _trim_one(s, qarr, off, *, trim_left, trim_right, trunc_q, trunc_len, min_len, max_len,
                  max_n, min_q, max_ee, rm_phix_flag, rm_low):
        if len(s) > max_len:
            return None
        start = max(1, trim_left + 1)
        if start > 1:
            if len(s) < start:
                return None
            s = s[start - 1:]
            qarr = qarr[start - 1:]
        if trim_right > 0:
            if len(s) <= trim_right:
                return None
            s = s[:-trim_right]
            qarr = qarr[:-trim_right]
        s, qarr = _trim_tail_truncq(s, qarr, trunc_q)
        if trunc_len and trunc_len > 0:
            if len(s) < trunc_len:
                return None
            s = s[:trunc_len]
            qarr = qarr[:trunc_len]
        if len(s) < min_len:
            return None
        if s.count("N") > max_n:
            return None
        if min_q > trunc_q and qarr.min() < min_q:
            return None
        if np.isfinite(max_ee) and _expected_errors(qarr) > max_ee:
            return None
        if rm_phix_flag and _is_phix(s):
            return None
        if rm_low > 0 and _seq_complexity(s) < rm_low:
            return None
        return s, qarr

    with _open_out(fout[0], compress=compress) as wF, _open_out(fout[1], compress=compress) as wR:
        for (hF, sF, qF), (hR, sR, qR) in zip(iterF, iterR):
            inseqs += 1
            qFa = np.frombuffer(qF.encode("ascii"), dtype=np.uint8).astype(np.int64) - offF
            qRa = np.frombuffer(qR.encode("ascii"), dtype=np.uint8).astype(np.int64) - offR

            tF = _trim_one(sF, qFa, offF,
                           trim_left=trimLeft[0], trim_right=trimRight[0],
                           trunc_q=truncQ[0], trunc_len=truncLen[0],
                           min_len=minLen[0], max_len=maxLen[0],
                           max_n=maxN[0], min_q=minQ[0], max_ee=maxEE[0],
                           rm_phix_flag=rm_phix[0], rm_low=rm_lowcomplex[0])
            if tF is None:
                continue
            tR = _trim_one(sR, qRa, offR,
                           trim_left=trimLeft[1], trim_right=trimRight[1],
                           trunc_q=truncQ[1], trunc_len=truncLen[1],
                           min_len=minLen[1], max_len=maxLen[1],
                           max_n=maxN[1], min_q=minQ[1], max_ee=maxEE[1],
                           rm_phix_flag=rm_phix[1], rm_low=rm_lowcomplex[1])
            if tR is None:
                continue

            outseqs += 1
            sF, qFa = tF
            sR, qRa = tR
            qF_b = bytes((qFa.clip(0, 93) + offF).astype(np.uint8))
            qR_b = bytes((qRa.clip(0, 93) + offR).astype(np.uint8))
            wF.write(f"{hF}\n{sF}\n+\n{qF_b.decode('ascii')}\n")
            wR.write(f"{hR}\n{sR}\n+\n{qR_b.decode('ascii')}\n")

    if outseqs == 0:
        for p in fout_p:
            if p.exists():
                p.unlink()
        if verbose:
            print(f"The filter removed all reads. Output not written.")

    return inseqs, outseqs


def filter_and_trim(
    fwd: Union[str, Path, Sequence[Union[str, Path]]],
    filt: Union[str, Path, Sequence[Union[str, Path]]],
    rev: Optional[Union[str, Path, Sequence[Union[str, Path]]]] = None,
    filt_rev: Optional[Union[str, Path, Sequence[Union[str, Path]]]] = None,
    *,
    compress: bool = True,
    truncQ=2,                          # noqa: N803
    truncLen=0,                        # noqa: N803
    trimLeft=0,                        # noqa: N803
    trimRight=0,                       # noqa: N803
    maxLen=float("inf"),               # noqa: N803
    minLen=20,                         # noqa: N803
    maxN=0,                            # noqa: N803
    minQ=0,                            # noqa: N803
    maxEE=float("inf"),                # noqa: N803
    rm_phix=True,
    rm_lowcomplex=0,
    matchIDs: bool = False,            # noqa: N803
    qualityType: str = "Auto",         # noqa: N803
    verbose: bool = False,
) -> np.ndarray:
    """High-level entrypoint matching R DADA2's ``filterAndTrim``.

    Returns an ``(n_files, 2)`` int array with columns ("reads.in", "reads.out").
    """
    paired = rev is not None

    def _vec(x):
        return list(x) if isinstance(x, (list, tuple)) else [x]

    fwd_l = _vec(fwd)
    filt_l = _vec(filt)
    if paired:
        rev_l = _vec(rev)
        fr_l = _vec(filt_rev)
        assert len(fwd_l) == len(rev_l) == len(filt_l) == len(fr_l)
    n = len(fwd_l)

    def _pair_or_scalar(v):
        if isinstance(v, (list, tuple)):
            return tuple(v)
        return (v, v) if paired else v

    out = np.zeros((n, 2), dtype=np.int64)
    for i in range(n):
        if paired:
            kw = dict(
                maxN=_pair_or_scalar(maxN), truncQ=_pair_or_scalar(truncQ),
                truncLen=_pair_or_scalar(truncLen), maxLen=_pair_or_scalar(maxLen),
                minLen=_pair_or_scalar(minLen), trimLeft=_pair_or_scalar(trimLeft),
                trimRight=_pair_or_scalar(trimRight), minQ=_pair_or_scalar(minQ),
                maxEE=_pair_or_scalar(maxEE),
                rm_phix=_pair_or_scalar(rm_phix), rm_lowcomplex=_pair_or_scalar(rm_lowcomplex),
                matchIDs=matchIDs, qualityType=qualityType, compress=compress, verbose=verbose,
            )
            out[i] = fastq_paired_filter([fwd_l[i], rev_l[i]],
                                          [filt_l[i], fr_l[i]], **kw)
        else:
            out[i] = fastq_filter(
                fwd_l[i], filt_l[i],
                truncQ=truncQ, truncLen=truncLen, maxLen=maxLen, minLen=minLen,
                trimLeft=trimLeft, trimRight=trimRight, maxN=maxN, minQ=minQ,
                maxEE=maxEE, rm_phix=rm_phix, rm_lowcomplex=rm_lowcomplex,
                qualityType=qualityType, compress=compress, verbose=verbose,
            )
    return out


# ----- helpers for PhiX & complexity -----------------------------------

_PHIX_REF = None  # lazy

def _is_phix(seq: str, word_size: int = 16) -> bool:
    """Return True if the read shares any 16-mer with the PhiX genome.

    Mirrors R's ``isPhiX`` (C_matchRef with default word_size=16).
    """
    global _PHIX_REF
    if _PHIX_REF is None:
        try:
            # Look up the PhiX genome that ships with R DADA2 (included in
            # inst/extdata). We check a couple of known paths.
            candidates = [
                Path(__file__).parent / "data" / "phix_genome.fa",
                Path("/scratch/users/steorra/analysis/omicverse_dev/cache/dada2_src/inst/extdata/phix_genome.fa"),
            ]
            phix_path = next((p for p in candidates if p.exists()), None)
            if phix_path is None:
                _PHIX_REF = ""  # graceful: no PhiX data, skip removal
            else:
                _PHIX_REF = "".join(_load_fasta_seqs(phix_path)).upper()
        except Exception:
            _PHIX_REF = ""
    if not _PHIX_REF:
        return False
    seq_u = seq.upper()
    n = len(seq_u)
    if n < word_size:
        return False
    # Build a kmer set from the read; check membership in the (precomputed)
    # PhiX kmer set.
    pkset = _phix_kmerset(word_size)
    for i in range(n - word_size + 1):
        if seq_u[i:i + word_size] in pkset:
            return True
    return False


_PHIX_KMERS_CACHE: Dict[int, set] = {}


def _phix_kmerset(k: int) -> set:
    if k in _PHIX_KMERS_CACHE:
        return _PHIX_KMERS_CACHE[k]
    s = set()
    if not _PHIX_REF:
        _PHIX_KMERS_CACHE[k] = s
        return s
    ref = _PHIX_REF
    for i in range(len(ref) - k + 1):
        s.add(ref[i:i + k])
    # Reverse complement too — DADA2's matchRef does both strands by default.
    rc = _rc(ref)
    for i in range(len(rc) - k + 1):
        s.add(rc[i:i + k])
    _PHIX_KMERS_CACHE[k] = s
    return s


def _rc(seq: str) -> str:
    table = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(table)[::-1]


def _load_fasta_seqs(path: Path):
    op = gzip.open if str(path).endswith(".gz") else open
    cur = []
    with op(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if cur:
                    yield "".join(cur)
                    cur = []
            else:
                cur.append(line.strip())
        if cur:
            yield "".join(cur)


def _seq_complexity(seq: str, kmer_size: int = 2) -> float:
    """Effective number of kmers via Shannon information.
    Mirrors R seqComplexity with default kmerSize=2.
    """
    if len(seq) < kmer_size:
        return 0.0
    counts: Dict[str, int] = {}
    for i in range(len(seq) - kmer_size + 1):
        k = seq[i:i + kmer_size]
        counts[k] = counts.get(k, 0) + 1
    total = sum(counts.values())
    p = np.fromiter((c / total for c in counts.values()), dtype=float, count=len(counts))
    H = -np.sum(p * np.log(p))  # nats
    return float(np.exp(H))
