"""Error rate models.

Mirrors R DADA2's ``errorModels.R``:

- ``loessErrfun``: per-transition log10-rate fit by quality score, using
  R's stats::loess (degree=2 polynomial, tricube weights, span=0.75 by default).
- ``noqualErrfun``: maximum-likelihood transition rate ignoring quality.
- ``inflateErr``: monotone fold-inflation while keeping rates in [0, 1].
- ``learnErrors``: self-consistent loop alternating dada inference + loess fit.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Callable, List, Optional, Sequence, Union

import numpy as np

# Mirror R's row order: A2A, A2C, A2G, A2T, C2A, ..., T2T (16 rows)
_NTS = ("A", "C", "G", "T")
ROW_NAMES: List[str] = [f"{i}2{j}" for i in _NTS for j in _NTS]
TRANSITION_ROWS: List[str] = [f"{i}2{j}" for i in _NTS for j in _NTS if i != j]
SELF_ROWS: List[str] = [f"{i}2{i}" for i in _NTS]


# ----- loess port (Python) ---------------------------------------------

def _loess_predict(x: np.ndarray, y: np.ndarray, weights: np.ndarray,
                   xpred: np.ndarray, span: float = 0.75, degree: int = 2) -> np.ndarray:
    """Re-implementation of R's ``stats::loess`` for the 1-D fit used by
    ``loessErrfun``: degree 2 polynomial, tricube weights, span 0.75.

    The R source's "exact" path is `loess.c`; this is a faithful Python port
    of the local quadratic fit used for each query point.
    """
    valid = np.isfinite(y) & (weights > 0)
    xv = x[valid].astype(float)
    yv = y[valid].astype(float)
    wv = weights[valid].astype(float)
    n_valid = xv.size
    if n_valid == 0:
        return np.full_like(xpred, np.nan, dtype=float)

    # number of nearest points used for each query: max(degree+1, span * n)
    span_n = max(degree + 1, int(np.ceil(span * n_valid)))
    span_n = min(span_n, n_valid)

    out = np.full(xpred.shape[0], np.nan, dtype=float)
    for k, q in enumerate(xpred):
        d = np.abs(xv - q)
        if span_n < n_valid:
            idx = np.argpartition(d, span_n - 1)[:span_n]
        else:
            idx = np.arange(n_valid)
        d_local = d[idx]
        h = d_local.max() if d_local.max() > 0 else 1.0
        # tricube weights: (1 - (d/h)^3)^3 for d<h, else 0
        u = d_local / h
        u_clip = np.minimum(u, 1.0)
        tw = (1.0 - u_clip ** 3) ** 3
        wfit = tw * wv[idx]
        # weighted least-squares polynomial fit of degree `degree`
        xx = xv[idx]
        # construct design matrix
        X = np.vander(xx - q, N=degree + 1, increasing=True)  # 1, dx, dx^2
        W = wfit
        # solve (X^T W X) b = X^T W y
        WX = X * W[:, None]
        try:
            b, *_ = np.linalg.lstsq(WX.T @ X, WX.T @ yv[idx], rcond=None)
            out[k] = b[0]  # value at dx=0 → b[0]
        except np.linalg.LinAlgError:
            out[k] = np.nan
    return out


def loess_errfun(trans: np.ndarray) -> np.ndarray:
    """Mirror of R's loessErrfun.

    Parameters
    ----------
    trans : (16, n_q) int matrix
        Observed transition counts. Row order MUST be ROW_NAMES.

    Returns
    -------
    err : (16, n_q) float matrix in [MIN_ERROR_RATE, MAX_ERROR_RATE].
    """
    MAX_ERR = 0.25
    MIN_ERR = 1e-7

    n_q = trans.shape[1]
    qq = np.arange(n_q, dtype=float)
    if trans.shape[0] != 16:
        raise ValueError("trans must have 16 rows.")

    # Compute per-transition loess prediction in original R order:
    # for nti in ACGT: for ntj in ACGT: if nti!=ntj
    est = []
    row_index = {n: i for i, n in enumerate(ROW_NAMES)}
    for nti in _NTS:
        for ntj in _NTS:
            if nti == ntj:
                continue
            errs = trans[row_index[f"{nti}2{ntj}"]].astype(float)
            tot_idx = [row_index[f"{nti}2{x}"] for x in _NTS]
            tot = trans[tot_idx].sum(axis=0).astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                rlogp = np.log10((errs + 1) / np.where(tot > 0, tot, np.nan))
            rlogp[np.isinf(rlogp)] = np.nan

            pred = _loess_predict(qq, rlogp, tot, qq, span=0.75, degree=2)

            # extend extrapolation flat at the edges (R's behaviour)
            valid = np.where(np.isfinite(pred))[0]
            if valid.size:
                lo, hi = valid.min(), valid.max()
                pred[:lo] = pred[lo]
                pred[hi + 1:] = pred[hi]
            est.append(10 ** pred)
    est = np.vstack(est)  # (12, n_q) in canonical order

    # Clamp
    est = np.clip(est, MIN_ERR, MAX_ERR)

    # Reassemble 16-row matrix with self-transitions = 1 - sum(others)
    # R order: A2A=1-A2C-A2G-A2T (rows 1-3), C2C=1-C2A-C2G-C2T (rows 4-6), etc.
    err = np.empty((16, n_q), dtype=float)
    err[0] = 1 - est[0:3].sum(axis=0)        # A2A
    err[1] = est[0]                           # A2C
    err[2] = est[1]                           # A2G
    err[3] = est[2]                           # A2T
    err[4] = est[3]                           # C2A
    err[5] = 1 - est[3:6].sum(axis=0)        # C2C
    err[6] = est[4]                           # C2G
    err[7] = est[5]                           # C2T
    err[8] = est[6]                           # G2A
    err[9] = est[7]                           # G2C
    err[10] = 1 - est[6:9].sum(axis=0)       # G2G
    err[11] = est[8]                          # G2T
    err[12] = est[9]                          # T2A
    err[13] = est[10]                         # T2C
    err[14] = est[11]                         # T2G
    err[15] = 1 - est[9:12].sum(axis=0)      # T2T
    return err


def no_qual_errfun(trans: np.ndarray, pseudocount: int = 1) -> np.ndarray:
    """Mirror noqualErrfun: per-transition rate ignoring quality, broadcast
    across all columns.
    """
    n_q = trans.shape[1]
    obs = trans.sum(axis=1) + pseudocount
    est = []
    row_index = {n: i for i, n in enumerate(ROW_NAMES)}
    for nti in _NTS:
        for ntj in _NTS:
            if nti == ntj:
                continue
            tot_trans = obs[row_index[f"{nti}2{ntj}"]]
            tot_init = sum(obs[row_index[f"{nti}2{x}"]] for x in _NTS)
            est.append(np.full(n_q, tot_trans / tot_init, dtype=float))
    est = np.vstack(est)
    err = np.empty((16, n_q), dtype=float)
    err[0] = 1 - est[0:3].sum(axis=0)
    err[1] = est[0]; err[2] = est[1]; err[3] = est[2]
    err[4] = est[3]; err[5] = 1 - est[3:6].sum(axis=0); err[6] = est[4]; err[7] = est[5]
    err[8] = est[6]; err[9] = est[7]; err[10] = 1 - est[6:9].sum(axis=0); err[11] = est[8]
    err[12] = est[9]; err[13] = est[10]; err[14] = est[11]; err[15] = 1 - est[9:12].sum(axis=0)
    return err


def inflate_err(err: np.ndarray, inflation: float, inflate_self: bool = False) -> np.ndarray:
    """Mirror inflateErr: rate * inflation / (1 + (inflation-1) * rate)."""
    out = np.array(err, dtype=float, copy=True)
    transition_idx = [i for i, n in enumerate(ROW_NAMES) if n in TRANSITION_ROWS]
    out[transition_idx] = (out[transition_idx] * inflation) / (
        1 + (inflation - 1) * out[transition_idx]
    )
    if inflate_self:
        self_idx = [i for i, n in enumerate(ROW_NAMES) if n in SELF_ROWS]
        out[self_idx] = (out[self_idx] * inflation) / (
            1 + (inflation - 1) * out[self_idx]
        )
    return out


def get_errors(obj, detailed: bool = False, enforce: bool = True):
    """Mirror getErrors: extract err_out from various inputs.

    Accepts: numeric matrix, dict with err_out/err_in/trans, dada result.
    """
    rval = {"err_out": None, "err_in": None, "trans": None}
    if isinstance(obj, np.ndarray) and obj.ndim == 2:
        rval["err_out"] = obj
    elif isinstance(obj, dict):
        if "err_out" in obj:
            rval["err_out"] = obj.get("err_out")
            rval["err_in"] = obj.get("err_in")
            rval["trans"] = obj.get("trans")
        elif "err" in obj:
            rval["err_out"] = obj["err"]
    elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
        # list of dada results — check err_out consistency, accumulate trans
        first = obj[0].get("err_out")
        if first is None:
            raise ValueError("First dada result has no err_out.")
        for o in obj[1:]:
            if not np.allclose(o.get("err_out"), first):
                raise ValueError("All dada results must share err_out.")
        rval["err_out"] = first
        rval["err_in"] = obj[0].get("err_in")
        # accumulate trans
        ts = [o.get("trans") for o in obj if o.get("trans") is not None]
        if ts:
            ncol = max(t.shape[1] for t in ts)
            acc = np.zeros((16, ncol), dtype=ts[0].dtype)
            for t in ts:
                acc[:, :t.shape[1]] += t
            rval["trans"] = acc
    if enforce:
        if rval["err_out"] is None:
            raise ValueError("Error matrix is NULL.")
        if rval["err_out"].shape[0] != 16:
            raise ValueError("Error matrix must have 16 rows.")
        if (rval["err_out"] < 0).any() or (rval["err_out"] > 1).any():
            raise ValueError("Error matrix entries must be in [0, 1].")
    if detailed:
        return rval
    return rval["err_out"]


def learn_errors(
    fls: Union[str, Sequence[str]],
    nbases: int = int(1e8),
    error_estimation_function: Callable = loess_errfun,
    multithread: bool = False,
    randomize: bool = False,
    MAX_CONSIST: int = 10,             # noqa: N803 — match R API
    OMEGA_C: float = 0.0,              # noqa: N803
    qualityType: str = "Auto",         # noqa: N803
    verbose: bool = True,
    **kwargs,
):
    """Self-consistent error learning. Mirrors R's learnErrors.

    Returns a dict with keys ``err_out``, ``err_in``, ``trans``.
    """
    from .io import derep_fastq
    from .dada import dada

    if isinstance(fls, str):
        fls = [fls]
    fls = list(fls)
    if randomize:
        rng = np.random.default_rng()
        rng.shuffle(fls)

    drps = []
    NBASES, NREADS = 0, 0
    for i, f in enumerate(fls):
        d = derep_fastq(f, qualityType=qualityType)
        drps.append(d)
        NREADS += int(d.abundances().sum())
        NBASES += int(sum(len(s) * c for s, c in d.uniques.items()))
        if NBASES > nbases:
            break
    if verbose:
        print(f"{NBASES} total bases in {NREADS} reads from {len(drps)} samples will be used for learning the error rates.")

    dds = dada(drps, err=None, error_estimation_function=error_estimation_function,
               selfConsist=True, multithread=multithread, verbose=verbose,
               MAX_CONSIST=MAX_CONSIST, OMEGA_C=OMEGA_C, **kwargs)
    return get_errors(dds, detailed=True)
