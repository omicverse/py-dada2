"""
pydada2: Pure-Python DADA2 for amplicon sequence variant inference.

A standalone mirror of the canonical implementation that lives in
``omicverse``. This repo exists for users who want DADA2 without
pulling in the full omicverse stack — all algorithmic work is
developed upstream in omicverse and synced here.

The pipeline mirrors the R DADA2 workflow:

>>> from pydada2 import (filter_and_trim, learn_errors, dada,
...                       merge_pairs, make_sequence_table,
...                       remove_bimera_denovo, assign_taxonomy)
>>> filter_and_trim(fwd, filtF, rev, filtR, trunc_len=(240, 160), max_ee=(2, 2))
>>> errF = learn_errors(filtF)
>>> ddF  = dada(filtF, err=errF)
>>> mergers = merge_pairs(ddF, filtF, ddR, filtR)
>>> seqtab = make_sequence_table(mergers)
>>> seqtab_nochim = remove_bimera_denovo(seqtab)
>>> taxa = assign_taxonomy(seqtab_nochim, ref_fasta="silva_nr99.fa.gz")
"""

from .opts import get_dada_opt, set_dada_opt
from .filter import filter_and_trim, fastq_filter, fastq_paired_filter
from .io import (
    derep_fastq,
    get_uniques,
    get_sequences,
    uniques_to_fasta,
)
from .align import (
    nwalign,
    nwhamming,
    rc,
)
from .kmers import kmer_dist, kord_dist, kmer_matches
from .errors import (
    learn_errors,
    loess_errfun,
    inflate_err,
    get_errors,
    no_qual_errfun,
)
from .dada import dada
from .paired import merge_pairs, eval_pair, pair_consensus
from .seqtab import (
    make_sequence_table,
    merge_sequence_tables,
    collapse_no_mismatch,
)
from .chimeras import (
    is_bimera,
    is_bimera_denovo,
    is_bimera_denovo_table,
    remove_bimera_denovo,
)
from .taxonomy import assign_taxonomy, assign_species, add_species

__version__ = "0.1.0"
__all__ = [
    # opts
    "get_dada_opt",
    "set_dada_opt",
    # filter
    "filter_and_trim",
    "fastq_filter",
    "fastq_paired_filter",
    # io
    "derep_fastq",
    "get_uniques",
    "get_sequences",
    "uniques_to_fasta",
    # align / kmers
    "nwalign",
    "nwhamming",
    "rc",
    "kmer_dist",
    "kord_dist",
    "kmer_matches",
    # error model
    "learn_errors",
    "loess_errfun",
    "inflate_err",
    "get_errors",
    "no_qual_errfun",
    # core
    "dada",
    # paired
    "merge_pairs",
    "eval_pair",
    "pair_consensus",
    # seqtab
    "make_sequence_table",
    "merge_sequence_tables",
    "collapse_no_mismatch",
    # chimeras
    "is_bimera",
    "is_bimera_denovo",
    "is_bimera_denovo_table",
    "remove_bimera_denovo",
    # taxonomy
    "assign_taxonomy",
    "assign_species",
    "add_species",
]
