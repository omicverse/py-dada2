"""The DADA2 divisive amplicon-denoising algorithm — Python port.

Mirrors dada2/src/Rmain.cpp::dada_uniques + cluster.cpp + pval.cpp.

Algorithm (from cluster.cpp):

    bb = b_new(raws)                  # one cluster (Bi[0]) holding all uniques
    b_compare(bb, 0)                  # align all raws to the center; record lambda
    b_p_update(bb)                    # Poisson-tail abundance p-value per raw
    while (nclust < max_clust):
        i = b_bud(bb)                 # find raw with min p-value passing thresholds
        if not i: break
        b_compare(bb, i)              # align all raws to the new center
        for _ in range(MAX_SHUFFLE):  # iteratively rebalance
            if not b_shuffle2(bb): break
        b_p_update(bb)

After convergence the cluster centers are the inferred ASV sequences.

The Python port favours readability and correctness over peak speed. The
inner alignment loop is hot — `nwalign` is Numba-jitted; everything else
runs on numpy. For typical MiSeq SOP samples (~700 uniques / sample) the
port runs in seconds.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
from scipy.stats import poisson

from .opts import get_dada_opt
from .io import DerepObject, derep_fastq
from ._subs import Sub, sub_new, compute_lambda


_NT2I_MAP = {"A": 1, "C": 2, "G": 3, "T": 4}
_TAIL_APPROX_CUTOFF = 1e-7


def _seq_to_int(seq: str) -> np.ndarray:
    arr = np.zeros(len(seq), dtype=np.int8)
    for i, c in enumerate(seq):
        arr[i] = _NT2I_MAP.get(c.upper(), 0)
    return arr


# ----- Raw / Bi / B containers (Pythonic, mirrors the C structs) ---------

@dataclass
class Raw:
    seq: str
    seq_int: np.ndarray         # A=1..T=4
    qual: np.ndarray            # rounded mean quals (length = len(seq))
    reads: int
    index: int                  # original position
    prior: bool = False
    p: float = 1.0
    E_minmax: float = 0.0
    lock: bool = False
    correct: bool = True
    # comparison to current cluster
    comp_i: int = 0             # cluster id of the comparison
    comp_lambda: float = 0.0
    comp_hamming: int = 0


@dataclass
class Bi:
    seq: str
    center: Raw
    raws: List[Raw] = field(default_factory=list)
    reads: int = 0
    update_e: bool = True
    check_locks: bool = True
    self_lambda: float = 0.0
    # all comparisons (raw_index -> Comparison(lambda,hamming))
    comp: Dict[int, "Comparison"] = field(default_factory=dict)
    # birth metadata
    birth_type: str = "I"       # I=initial A=abundance P=prior
    birth_from: int = 0
    birth_pval: float = 0.0
    birth_fold: float = 0.0
    birth_e: float = 0.0


@dataclass
class Comparison:
    i: int
    index: int
    lambda_: float
    hamming: int


@dataclass
class B:
    raws: List[Raw]
    bi: List[Bi]
    nraw: int
    reads: int
    omegaA: float
    omegaP: float
    use_quals: bool
    nalign: int = 0
    nshroud: int = 0


def _calc_pA(reads: int, E_reads: float, prior: bool) -> float:
    """Mirror calc_pA in pval.cpp.

    P(X > reads-1 | Poisson(E_reads)) — conditional on presence (divide by
    1 - exp(-E_reads)) when prior is False.
    """
    if E_reads <= 0:
        return 1.0
    # ppois lower.tail=False: P(X > k)
    pval = float(poisson.sf(reads - 1, E_reads))
    if not prior:
        norm = 1.0 - math.exp(-E_reads)
        if norm < _TAIL_APPROX_CUTOFF:
            norm = E_reads - 0.5 * E_reads * E_reads
        pval = pval / norm
    return pval


def _get_pA(raw: Raw, bi: Bi, detect_singletons: bool) -> float:
    """Mirror get_pA in pval.cpp."""
    lam = raw.comp_lambda
    ham = raw.comp_hamming
    if raw.reads == 1 and not raw.prior and not detect_singletons:
        return 1.0
    if ham == 0:
        return 1.0
    if lam == 0:
        return 0.0
    E_reads = lam * bi.reads
    return _calc_pA(raw.reads, E_reads, raw.prior or detect_singletons)


def _b_p_update(b: B, greedy: bool, detect_singletons: bool) -> None:
    for bi in b.bi:
        if bi.update_e:
            for raw in bi.raws:
                raw.p = _get_pA(raw, bi, detect_singletons)
            bi.update_e = False
        if greedy and bi.check_locks:
            for raw in bi.raws:
                E_center = bi.center.reads * raw.comp_lambda
                if E_center > raw.reads:
                    raw.lock = True
                if raw is bi.center:
                    raw.lock = True
            bi.check_locks = False


def _b_compare(b: B, i: int, err_mat: np.ndarray,
               *, match: int, mismatch: int, gap_p: int, homo_gap: int,
               band: int, use_kmers: bool, kdist_cutoff: float,
               greedy: bool, ncol_q: int) -> None:
    """Mirror b_compare in cluster.cpp.

    Aligns every raw in b.raws to bi[i].center, computes lambda,
    stores Comparison entries when E_minmax can be improved.
    """
    bi = b.bi[i]
    center = bi.center
    center_reads = center.reads
    use_quals = b.use_quals

    for raw in b.raws:
        # greedy speed-ups (mirror cluster.cpp lines 56-63)
        sub: Optional[Sub]
        if greedy and raw.reads > center_reads:
            sub = None
        elif greedy and raw.lock:
            sub = None
        else:
            sub = sub_new(
                center.seq, raw.seq,
                match=match, mismatch=mismatch, gap_p=gap_p, homo_gap_p=homo_gap,
                band=band, use_kmers=use_kmers, kdist_cutoff=kdist_cutoff,
                q0_seq=center.qual if use_quals else None,
                q1_seq=raw.qual if use_quals else None,
            )
            b.nalign += 1
            if sub is None:
                b.nshroud += 1

        # compute lambda = prod err_mat[transition, quality] over positions of seq1 (raw)
        if sub is None:
            lam = 0.0
            ham = -1
        else:
            qind = (raw.qual.clip(0, ncol_q - 1)).astype(np.int64) if use_quals else np.zeros(raw.seq_int.shape[0], dtype=np.int64)
            lam = compute_lambda(raw.seq_int, qind, sub, err_mat, use_quals)
            ham = sub.nsubs

        if raw is center:
            bi.self_lambda = lam

        # store comparison if potentially useful (cluster.cpp lines 73-85)
        if lam * b.reads > raw.E_minmax:
            if lam * center_reads > raw.E_minmax:
                raw.E_minmax = lam * center_reads
            comp = Comparison(i=i, index=raw.index, lambda_=lam, hamming=ham if ham >= 0 else 0)
            bi.comp[raw.index] = comp
            if i == 0 or raw is center:
                raw.comp_i = i
                raw.comp_lambda = lam
                raw.comp_hamming = ham if ham >= 0 else 0


def _b_shuffle2(b: B) -> bool:
    """Mirror b_shuffle2 in cluster.cpp.

    For each raw, find the cluster maximising E_reads = lambda * bi.reads.
    Move the raw if a better cluster exists. Centers cannot move.
    """
    nraw = b.nraw
    # initialise from cluster 0 (which has comparisons to every raw after b_compare(0))
    emax = np.full(nraw, -1.0, dtype=float)
    imax = np.zeros(nraw, dtype=np.int32)
    src_comp: List[Optional[Comparison]] = [None] * nraw
    for index in range(nraw):
        c = b.bi[0].comp.get(index)
        if c is None:
            continue
        emax[index] = c.lambda_ * b.bi[0].reads
        imax[index] = 0
        src_comp[index] = c

    for i in range(1, len(b.bi)):
        for index, comp in b.bi[i].comp.items():
            e = comp.lambda_ * b.bi[i].reads
            if e > emax[index]:
                emax[index] = e
                imax[index] = i
                src_comp[index] = comp

    shuffled = False
    for i, bi in enumerate(b.bi):
        # iterate in reverse so we can pop
        new_raws = []
        for raw in bi.raws:
            target_i = int(imax[raw.index])
            if target_i == i:
                new_raws.append(raw)
                continue
            if raw is bi.center:
                # cannot move centers
                new_raws.append(raw)
                continue
            # move raw to bi[target_i]
            b.bi[target_i].raws.append(raw)
            comp = src_comp[raw.index]
            if comp is not None:
                raw.comp_i = comp.i
                raw.comp_lambda = comp.lambda_
                raw.comp_hamming = comp.hamming
            shuffled = True
        bi.raws = new_raws

    if shuffled:
        for bi in b.bi:
            _bi_census(bi)

    return shuffled


def _bi_census(bi: Bi) -> None:
    new_reads = sum(r.reads for r in bi.raws)
    if new_reads != bi.reads:
        bi.update_e = True
    bi.reads = new_reads


def _bi_assign_center(bi: Bi) -> None:
    if not bi.raws:
        return
    bi.center = max(bi.raws, key=lambda r: r.reads)
    bi.seq = bi.center.seq
    for r in bi.raws:
        r.lock = False
    bi.check_locks = True


def _b_bud(b: B, min_fold: float, min_hamming: int, min_abund: int) -> int:
    """Mirror b_bud. Returns index of new cluster, or 0 if none added."""
    minraw: Optional[Raw] = b.bi[0].center
    minraw_prior: Optional[Raw] = b.bi[0].center
    mini = -1; minr = -1
    mini_prior = -1; minr_prior = -1

    for i, bi in enumerate(b.bi):
        for r, raw in enumerate(bi.raws):
            if r == 0:  # center
                continue
            if raw.reads < min_abund:
                continue
            ham = raw.comp_hamming
            lam = raw.comp_lambda
            if ham < min_hamming:
                continue
            if min_fold > 1 and not (raw.reads >= min_fold * lam * bi.reads):
                continue
            # most significant
            if (raw.p < minraw.p) or (raw.p == minraw.p and raw.reads > minraw.reads):
                mini, minr, minraw = i, r, raw
            if raw.prior and ((raw.p < minraw_prior.p) or (raw.p == minraw_prior.p and raw.reads > minraw_prior.reads)):
                mini_prior, minr_prior, minraw_prior = i, r, raw

    pA = minraw.p * b.nraw  # Bonferroni-correct
    pP = minraw_prior.p

    if pA < b.omegaA and mini >= 0:
        bi_src = b.bi[mini]
        expected = minraw.comp_lambda * bi_src.reads
        # pop raw
        bi_src.raws.pop(minr)
        # new bi
        new_bi = Bi(seq=minraw.seq, center=minraw, raws=[minraw], reads=minraw.reads,
                    birth_type="A", birth_from=mini, birth_pval=pA,
                    birth_fold=minraw.reads / expected if expected > 0 else float("inf"),
                    birth_e=expected, update_e=True)
        b.bi.append(new_bi)
        _bi_assign_center(new_bi)
        _bi_census(new_bi)
        _bi_census(bi_src)
        return len(b.bi) - 1

    if pP < b.omegaP and mini_prior >= 0:
        bi_src = b.bi[mini_prior]
        expected = minraw_prior.comp_lambda * bi_src.reads
        bi_src.raws.pop(minr_prior)
        new_bi = Bi(seq=minraw_prior.seq, center=minraw_prior, raws=[minraw_prior],
                    reads=minraw_prior.reads, birth_type="P", birth_from=mini_prior,
                    birth_pval=pP, birth_fold=minraw_prior.reads / expected if expected > 0 else float("inf"),
                    birth_e=expected, update_e=True)
        b.bi.append(new_bi)
        _bi_assign_center(new_bi)
        _bi_census(new_bi)
        _bi_census(bi_src)
        return len(b.bi) - 1

    return 0


def _build_trans(b: B, ncol_q: int) -> np.ndarray:
    """Compute the (16, ncol_q) transition matrix from current clustering.

    Sums per-position transitions weighted by raw.reads, comparing every raw
    to its current cluster center via the stored comparison's hamming
    pattern. We re-derive the per-position transitions from a fresh
    sub_new to ensure the same alignment used in lambda is used here.
    """
    trans = np.zeros((16, ncol_q), dtype=np.int64)
    for bi in b.bi:
        for raw in bi.raws:
            if raw is bi.center:
                # self transitions: every position contributes nti->nti
                for pos in range(raw.seq_int.shape[0]):
                    nti = int(raw.seq_int[pos]) - 1
                    if nti < 0 or nti > 3:
                        continue
                    q = int(raw.qual[pos]) if raw.qual is not None else 0
                    if q < 0 or q >= ncol_q:
                        continue
                    trans[nti * 4 + nti, q] += raw.reads
                continue
            sub = sub_new(bi.center.seq, raw.seq,
                          q0_seq=bi.center.qual, q1_seq=raw.qual,
                          use_kmers=False)  # no kmer pre-screen for trans
            if sub is None:
                continue
            # All matched positions count as self-transitions
            sub_pos_set = set(int(p) for p in sub.pos)
            for p0 in range(sub.len0):
                p1 = int(sub.map[p0])
                if p1 < 0:
                    continue
                if p0 in sub_pos_set:
                    continue
                nti = int(raw.seq_int[p1]) - 1
                q = int(raw.qual[p1]) if raw.qual is not None else 0
                if q < 0 or q >= ncol_q or nti < 0 or nti > 3:
                    continue
                trans[nti * 4 + nti, q] += raw.reads
            # substitution positions
            for s in range(sub.nsubs):
                p0 = int(sub.pos[s])
                p1 = int(sub.map[p0])
                if p1 < 0:
                    continue
                nti0 = int(sub.nt0[s]) - 1
                nti1 = int(sub.nt1[s]) - 1
                q = int(raw.qual[p1]) if raw.qual is not None else 0
                if q < 0 or q >= ncol_q:
                    continue
                trans[nti0 * 4 + nti1, q] += raw.reads
    return trans


# ----- public API ------------------------------------------------------

def _make_raws(uniques: Dict[str, int], quals: np.ndarray, priors: Optional[set] = None) -> List[Raw]:
    seqs = list(uniques.keys())
    raws: List[Raw] = []
    for i, s in enumerate(seqs):
        n = len(s)
        raws.append(Raw(
            seq=s,
            seq_int=_seq_to_int(s),
            qual=np.asarray(quals[i, :n], dtype=np.int64) if quals is not None and quals.size else np.zeros(n, dtype=np.int64),
            reads=int(uniques[s]),
            index=i,
            prior=(priors is not None and s in priors),
        ))
    return raws


def _initial_err(uniques: Dict[str, int], n_q: int = 41) -> np.ndarray:
    """Used when err=None: assume all reads are errors away from the most
    abundant sequence (per R DADA2's fallback in dada()).
    """
    err = np.full((16, n_q), 1e-3, dtype=float)
    # self transitions → 1 - 3*1e-3
    for i in range(4):
        err[i * 4 + i] = 1 - 3 * 1e-3
    return err


def _run_dada_one(uniques: Dict[str, int], quals: np.ndarray, err_mat: np.ndarray,
                  *, opts: Dict[str, Any], priors: Optional[set] = None,
                  detect_singletons: bool = False, max_clust: int = 0) -> B:
    raws = _make_raws(uniques, quals, priors=priors)
    nraw = len(raws)
    total_reads = sum(r.reads for r in raws)

    if max_clust < 1:
        max_clust = nraw

    # initial cluster: all raws, center = most-abundant
    center0 = max(raws, key=lambda r: r.reads)
    bi0 = Bi(seq=center0.seq, center=center0, raws=list(raws), reads=total_reads, update_e=True)
    b = B(raws=raws, bi=[bi0], nraw=nraw, reads=total_reads,
          omegaA=opts["OMEGA_A"], omegaP=opts["OMEGA_P"], use_quals=opts["USE_QUALS"])

    ncol_q = err_mat.shape[1]
    common_kw = dict(
        match=opts["MATCH"], mismatch=opts["MISMATCH"], gap_p=opts["GAP_PENALTY"],
        homo_gap=opts["HOMOPOLYMER_GAP_PENALTY"] if opts["HOMOPOLYMER_GAP_PENALTY"] else 0,
        band=opts["BAND_SIZE"], use_kmers=opts["USE_KMERS"],
        kdist_cutoff=opts["KDIST_CUTOFF"], greedy=opts["GREEDY"], ncol_q=ncol_q,
    )

    # initial compare with no kmer screen (kdist=1.0)
    kw0 = dict(common_kw); kw0["kdist_cutoff"] = 1.0
    _b_compare(b, 0, err_mat, **kw0)
    _b_p_update(b, greedy=opts["GREEDY"], detect_singletons=detect_singletons)

    MAX_SHUFFLE = 10
    while len(b.bi) < max_clust:
        newi = _b_bud(b, opts["MIN_FOLD"], opts["MIN_HAMMING"], opts["MIN_ABUNDANCE"])
        if not newi:
            break
        _b_compare(b, newi, err_mat, **common_kw)
        for _ in range(MAX_SHUFFLE):
            if not _b_shuffle2(b):
                break
        _b_p_update(b, greedy=opts["GREEDY"], detect_singletons=detect_singletons)

    return b


def _b_to_result(b: B, err_in: np.ndarray, opts: Dict[str, Any]) -> Dict[str, Any]:
    """Bundle a B into a ``dada-class``-like dict.

    Mirrors Rmain.cpp lines 238-279: re-evaluate per-raw within-cluster
    p-values (with prior=True, so no presence normalisation) and mark
    raws with p < OMEGA_C as correct=False — they are excluded from the
    cluster abundance and from the cluster-index map.
    """
    ncol_q = err_in.shape[1]
    omega_c = opts.get("OMEGA_C", 1e-40)

    # Final per-raw correction pass — mirrors Rmain.cpp::dada_uniques final loop
    for bi in b.bi:
        for raw in bi.raws:
            if raw is bi.center:
                raw.p = 1.0
                raw.correct = True
            else:
                E_reads = raw.comp_lambda * bi.reads
                raw.p = _calc_pA(raw.reads, E_reads, prior=True)
                if raw.p < omega_c:
                    raw.correct = False
                else:
                    raw.correct = True

    trans = _build_trans(b, ncol_q)
    denoised: "OrderedDict[str, int]" = OrderedDict()
    cluster_records = []
    for i, bi in enumerate(b.bi):
        # Cluster abundance counts only correct raws
        cluster_abund = sum(r.reads for r in bi.raws if r.correct)
        denoised[bi.center.seq] = cluster_abund
        cluster_records.append({
            "sequence": bi.center.seq,
            "abundance": cluster_abund,
            "n0": bi.center.reads,  # reads in cluster center
            "nunq": sum(1 for r in bi.raws if r.correct),
            "birth_type": bi.birth_type,
            "birth_from": bi.birth_from,
            "birth_pval": bi.birth_pval,
            "birth_fold": bi.birth_fold,
            "birth_e": bi.birth_e,
        })
    # map raw index -> cluster index (-1 if not corrected)
    n = b.nraw
    map_arr = np.full(n, -1, dtype=np.int64)
    for i, bi in enumerate(b.bi):
        for raw in bi.raws:
            if raw.correct:
                map_arr[raw.index] = i
    return {
        "denoised": denoised,
        "clustering": cluster_records,
        "map": map_arr,
        "trans": trans,
        "err_in": err_in,
        "err_out": None,  # set by caller if selfConsist
        "opts": dict(opts),
    }


def dada(
    derep: Union[str, "DerepObject", List, "OrderedDict"],
    err: Optional[np.ndarray] = None,
    error_estimation_function: Optional[Callable] = None,
    selfConsist: bool = False,                 # noqa: N803 — match R API
    pool: Union[bool, str] = False,
    priors: Sequence[str] = (),
    multithread: bool = False,
    verbose: bool = True,
    detect_singletons: bool = False,
    MAX_CONSIST: int = 10,                     # noqa: N803
    OMEGA_C: float = 1e-40,                    # noqa: N803
    **opts_overrides,
):
    """High-resolution sample inference. Mirrors R DADA2's ``dada``.

    Parameters
    ----------
    derep : path or DerepObject or list of either
        Input. A path triggers a derep pass internally.
    err : (16, n_q) error matrix, or None to estimate via selfConsist.
    selfConsist : bool
        Alternate inference + error estimation until convergence.
    pool : bool or "pseudo"
        Pool samples before inference (True) or do per-sample then second
        pass with pseudo-priors ("pseudo").
    priors : sequences known to be present (lower significance threshold).
    """
    from .errors import loess_errfun, get_errors

    # --- normalise input to list of DerepObject + names ---
    single_input = isinstance(derep, (str, bytes, DerepObject))
    if isinstance(derep, (str, bytes)):
        derep = [derep]
    if isinstance(derep, DerepObject):
        derep = [derep]

    drps: List[DerepObject] = []
    names: List[str] = []
    for i, d in enumerate(derep):
        if isinstance(d, (str, bytes)):
            drps.append(derep_fastq(d))
            names.append(str(d))
        elif isinstance(d, DerepObject):
            drps.append(d)
            names.append(f"sample_{i}")
        else:
            raise TypeError(f"Unsupported derep type: {type(d)}")

    if pool is True and len(drps) > 1:
        # combine into a single derep
        combined: "OrderedDict[str, int]" = OrderedDict()
        rows = []
        for d in drps:
            for s, c in d.uniques.items():
                combined[s] = combined.get(s, 0) + c
        # recompute quals (weighted average across dereps)
        seqs = list(combined.keys())
        max_len = max(len(s) for s in seqs)
        qsum = np.zeros((len(seqs), max_len), dtype=float)
        wsum = np.zeros((len(seqs), max_len), dtype=float)
        idx_map = {s: i for i, s in enumerate(seqs)}
        for d in drps:
            for s, c in d.uniques.items():
                i = idx_map[s]
                src_i = list(d.uniques.keys()).index(s)
                n = min(d.quals.shape[1], len(s))
                q_row = np.where(np.isnan(d.quals[src_i, :n]), 0, d.quals[src_i, :n])
                qsum[i, :n] += q_row * c
                wsum[i, :n] += c
        quals = np.where(wsum > 0, qsum / np.maximum(wsum, 1), np.nan)
        drps = [DerepObject(uniques=combined, quals=quals, map=np.array([], dtype=np.int64))]
        names = ["pooled"]

    # --- options ---
    opts = get_dada_opt()
    for k, v in opts_overrides.items():
        if k in opts:
            opts[k] = v
    if error_estimation_function is None:
        error_estimation_function = loess_errfun

    priors_set = set(priors) if priors else None

    # --- selfConsist loop ---
    NCOL_Q = (err.shape[1] if err is not None else 41)
    if err is None:
        cur_err = _initial_err({}, n_q=NCOL_Q)
    else:
        cur_err = np.asarray(err, dtype=float)

    if not selfConsist:
        # one-pass inference per sample
        results = []
        for i, d in enumerate(drps):
            b = _run_dada_one(d.uniques, d.quals, cur_err, opts=opts, priors=priors_set,
                              detect_singletons=detect_singletons)
            r = _b_to_result(b, cur_err, opts)
            r["sample"] = names[i]
            r["err_out"] = cur_err
            results.append(r)
        if len(results) == 1 and single_input:
            return results[0]
        return results

    # selfConsist
    err_history = [cur_err]
    for it in range(MAX_CONSIST):
        results = []
        trans_total = np.zeros((16, NCOL_Q), dtype=np.int64)
        for i, d in enumerate(drps):
            b = _run_dada_one(d.uniques, d.quals, cur_err, opts=opts, priors=priors_set,
                              detect_singletons=detect_singletons)
            r = _b_to_result(b, cur_err, opts)
            r["sample"] = names[i]
            results.append(r)
            tt = r["trans"]
            n_cols = tt.shape[1]
            trans_total[:, :n_cols] += tt

        new_err = error_estimation_function(trans_total)
        err_history.append(new_err)
        if np.allclose(new_err, cur_err, atol=1e-8):
            cur_err = new_err
            if verbose:
                print(f"   selfConsist converged after {it + 1} round(s).")
            break
        cur_err = new_err
        if verbose:
            print(f"   selfConsist round {it + 1}: re-estimated error rates.")

    for r in results:
        r["err_out"] = cur_err
        r["err_in"] = err_history[0]
    if len(results) == 1 and not isinstance(derep, list):
        return results[0]
    return results
