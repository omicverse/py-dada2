"""FASTQ I/O + dereplication.

Mirrors R DADA2's ``derepFastq`` / ``getUniques`` / ``getSequences`` /
``uniquesToFasta``.

Dereplication groups identical reads, keeping abundances (read counts) and
the average integer quality at each position (round to nearest, matching
the R reference's ``round(mean(qual))`` per-position consensus).
"""
from __future__ import annotations

import gzip
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Union

import numpy as np


_FastqLike = Union[str, Path, "DerepObject", Mapping[str, int]]


def _open_fastq(path: Union[str, Path]):
    """Open a fastq file, transparently handling .gz."""
    p = Path(path)
    if p.suffix == ".gz":
        return io.TextIOWrapper(gzip.open(p, "rb"), encoding="ascii", newline="\n")
    return open(p, "r", encoding="ascii", newline="\n")


def _iter_fastq(path: Union[str, Path]):
    """Yield (header, seq, qual_str) for each read."""
    with _open_fastq(path) as fh:
        while True:
            h = fh.readline()
            if not h:
                return
            s = fh.readline()
            plus = fh.readline()
            q = fh.readline()
            if not q:
                return
            yield h.rstrip("\n"), s.rstrip("\n"), q.rstrip("\n")


@dataclass
class DerepObject:
    """Mirror of R's `derep-class`.

    Attributes
    ----------
    uniques : dict
        sequence -> read count, ordered by descending abundance
    quals : np.ndarray
        (n_unique, max_seqlen) array of mean per-position Phred quality
        scores (rounded to nearest int as in the R source). NaN for
        positions past the end of a sequence (variable lengths).
    map : np.ndarray
        For each input read i, the index in ``uniques`` of its unique seq.
    """
    uniques: Dict[str, int] = field(default_factory=dict)
    quals: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=float))
    map: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    @property
    def n_unique(self) -> int:
        return len(self.uniques)

    def sequences(self) -> List[str]:
        return list(self.uniques.keys())

    def abundances(self) -> np.ndarray:
        return np.fromiter(self.uniques.values(), dtype=np.int64, count=len(self.uniques))


def derep_fastq(
    path: Union[str, Path, Sequence[Union[str, Path]]],
    qualityType: str = "Auto",  # noqa: N803 — match R API
    verbose: bool = False,
) -> Union[DerepObject, List[DerepObject]]:
    """Dereplicate a fastq file (or list of files).

    Parameters
    ----------
    path : str | Path | sequence of those
        FASTQ file path(s). Compressed (`.gz`) is supported.
    qualityType : {"Auto", "Phred33", "Phred64"}
        Quality offset (auto-detected from min observed score by default).
    verbose : bool
        Print per-file dereplication summary.

    Returns
    -------
    DerepObject (single input) or list[DerepObject].
    """
    if isinstance(path, (list, tuple)):
        return [derep_fastq(p, qualityType=qualityType, verbose=verbose) for p in path]

    # First pass: collect sequences and per-position raw quality sums + counts.
    # Do it streaming to avoid loading the whole fastq.
    seqs: Dict[str, int] = {}
    # We accumulate quality as int sums; later divide and round.
    # To mirror R behaviour we keep variable lengths — store per-sequence list.
    qsum: Dict[str, np.ndarray] = {}
    qcnt: Dict[str, int] = {}
    read_to_idx: List[int] = []
    seq_to_idx: Dict[str, int] = {}

    offset = _phred_offset(qualityType, path)

    for _, s, q in _iter_fastq(path):
        if s not in seqs:
            seqs[s] = 0
            qsum[s] = np.zeros(len(s), dtype=np.int64)
            qcnt[s] = 0
            seq_to_idx[s] = len(seq_to_idx)
        seqs[s] += 1
        # decode quality
        qarr = np.frombuffer(q.encode("ascii"), dtype=np.uint8).astype(np.int64) - offset
        if qarr.shape[0] != qsum[s].shape[0]:
            # length mismatch shouldn't happen for an identical-seq dedup, but
            # be defensive: pad/trim.
            n = min(qarr.shape[0], qsum[s].shape[0])
            qsum[s][:n] += qarr[:n]
        else:
            qsum[s] += qarr
        qcnt[s] += 1
        read_to_idx.append(seq_to_idx[s])

    # Sort by descending abundance, then by sequence (stable, mirrors R)
    items = sorted(seqs.items(), key=lambda kv: (-kv[1], kv[0]))
    ordered: Dict[str, int] = dict(items)
    new_idx = {s: i for i, s in enumerate(ordered)}

    n_unique = len(ordered)
    max_len = max((len(s) for s in ordered), default=0)
    quals = np.full((n_unique, max_len), np.nan, dtype=float)
    for s, i in new_idx.items():
        n = qsum[s].shape[0]
        # R: `round(colMeans(qual))` — nearest int, banker's-rounding compatible
        quals[i, :n] = np.round(qsum[s] / qcnt[s])

    rmap = np.array([new_idx[list(seq_to_idx.keys())[old_i]] for old_i in read_to_idx],
                    dtype=np.int64)
    # NB: faster reverse-mapping: build old_i -> seq up front
    inv_seq = {v: k for k, v in seq_to_idx.items()}
    rmap = np.fromiter((new_idx[inv_seq[i]] for i in read_to_idx),
                       dtype=np.int64, count=len(read_to_idx))

    if verbose:
        print(f"Dereplicating sequence entries in {path}…")
        print(f"  Encountered {sum(seqs.values())} total sequences in {n_unique} unique sequences.")

    return DerepObject(uniques=ordered, quals=quals, map=rmap)


def _phred_offset(qualityType: str, path) -> int:
    """Detect Phred offset (33 or 64). Mirrors ShortRead's auto behaviour."""
    qt = qualityType.lower() if qualityType else "auto"
    if qt == "phred33":
        return 33
    if qt == "phred64":
        return 64
    # Auto: peek at the first chunk and pick.
    min_q = 255
    for i, (_, _, q) in enumerate(_iter_fastq(path)):
        for c in q.encode("ascii"):
            if c < min_q:
                min_q = c
        if i >= 1000:
            break
    # Phred33 minimum is '!' (33). If we ever see anything < 64, must be Phred33.
    return 33 if min_q < 64 else 64


def get_uniques(obj) -> Dict[str, int]:
    """Mirror R's getUniques: extract the seq->abundance dict.

    Accepts a DerepObject, a dict, a `dada` result dict with a
    "denoised" entry, or a pandas DataFrame with `sequence` +
    `abundance` columns (e.g. from `merge_pairs`).
    """
    if isinstance(obj, DerepObject):
        return dict(obj.uniques)
    if isinstance(obj, dict):
        if "denoised" in obj and isinstance(obj["denoised"], dict):
            return dict(obj["denoised"])
        return dict(obj)
    # pandas DataFrame from merge_pairs / chimeras / etc.
    if hasattr(obj, "columns") and "sequence" in obj.columns and "abundance" in obj.columns:
        out: Dict[str, int] = {}
        for s, a in zip(obj["sequence"], obj["abundance"]):
            if not s:
                continue
            out[s] = out.get(s, 0) + int(a)
        return out
    raise TypeError(f"Don't know how to extract uniques from {type(obj)}")


def get_sequences(obj) -> List[str]:
    """Mirror R's getSequences."""
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, (list, tuple)):
        if all(isinstance(x, str) for x in obj):
            return list(obj)
    if isinstance(obj, dict):
        return list(obj.keys())
    if isinstance(obj, DerepObject):
        return obj.sequences()
    if hasattr(obj, "columns"):  # pandas DataFrame
        return list(obj.columns)
    raise TypeError(f"Cannot extract sequences from {type(obj)}")


def uniques_to_fasta(unqs, fout: Union[str, Path], ids=None, mode: str = "w") -> None:
    """Write uniques (dict or DerepObject) to FASTA, one record per unique."""
    seqs = get_uniques(unqs)
    p = Path(fout)
    if ids is None:
        ids = [f"sq{i}" for i in range(1, len(seqs) + 1)]
    with open(p, mode) as fh:
        for i, (seq, count) in enumerate(seqs.items()):
            fh.write(f">{ids[i]};size={count}\n{seq}\n")
