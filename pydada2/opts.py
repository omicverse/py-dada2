"""DADA2 options — exactly mirror the R `dada_opts` environment.

Defaults are taken from R/dada.R (lines 1-27 of the upstream source).
"""
from __future__ import annotations

from typing import Any, Dict

# Defaults from R/dada.R
_DEFAULTS: Dict[str, Any] = {
    "OMEGA_A": 1e-40,
    "OMEGA_P": 1e-4,
    "OMEGA_C": 1e-40,
    "DETECT_SINGLETONS": False,
    "USE_KMERS": True,
    "KDIST_CUTOFF": 0.42,
    "MAX_CONSIST": 10,
    "MATCH": 5,
    "MISMATCH": -4,
    "GAP_PENALTY": -8,
    "BAND_SIZE": 16,
    "VECTORIZED_ALIGNMENT": True,
    "MAX_CLUST": 0,
    "MIN_FOLD": 1,
    "MIN_HAMMING": 1,
    "MIN_ABUNDANCE": 1,
    "USE_QUALS": True,
    "HOMOPOLYMER_GAP_PENALTY": None,
    "SSE": 2,
    "GAPLESS": True,
    "GREEDY": True,
    "PSEUDO_PREVALENCE": 2,
    "PSEUDO_ABUNDANCE": float("inf"),
}

_OPTS: Dict[str, Any] = dict(_DEFAULTS)


def get_dada_opt(name: str | None = None) -> Any:
    """Return one option (by name) or the full dict if name is None."""
    if name is None:
        return dict(_OPTS)
    if name not in _OPTS:
        raise KeyError(f"{name!r} is not a valid DADA option.")
    return _OPTS[name]


def set_dada_opt(**kwargs: Any) -> None:
    """Set one or more DADA options. Unknown keys raise KeyError."""
    for k, v in kwargs.items():
        if k not in _OPTS:
            raise KeyError(f"{k!r} is not a valid DADA option.")
        _OPTS[k] = v


def reset_dada_opt() -> None:
    _OPTS.clear()
    _OPTS.update(_DEFAULTS)
