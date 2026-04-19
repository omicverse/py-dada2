"""Taxonomy assignment.

Mirrors R DADA2's `assignTaxonomy` (RDP naive Bayes, kmer=8, 100 bootstraps)
and `assignSpecies` (exact 100% match against species reference).
"""
from __future__ import annotations

import gzip
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .align import rc
from .filter import _load_fasta_seqs


KMER_SIZE = 8
N_BOOTS = 100
MIN_REF_LEN = 20
MIN_TAX_LEN = 50


def _read_ref_fasta(path: Union[str, Path]) -> Tuple[List[str], List[str]]:
    """Return (sequences, taxonomy strings) from a fasta where each header
    line is the taxonomy ('Kingdom;Phylum;...').
    """
    p = Path(path)
    op = gzip.open if str(p).endswith(".gz") else open
    seqs: List[str] = []
    taxa: List[str] = []
    cur_seq: List[str] = []
    cur_tax: Optional[str] = None
    with op(p, "rt") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if cur_tax is not None:
                    seqs.append("".join(cur_seq).upper())
                    taxa.append(cur_tax)
                cur_tax = line[1:].strip()
                cur_seq = []
            else:
                cur_seq.append(line.strip())
    if cur_tax is not None:
        seqs.append("".join(cur_seq).upper())
        taxa.append(cur_tax)
    return seqs, taxa


def _kmer_set(seq: str, k: int = KMER_SIZE) -> set:
    """All distinct ACGT-only kmers in seq."""
    s = set()
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i + k]
        if all(c in "ACGT" for c in kmer):
            s.add(kmer)
    return s


def _genus_priors(ref_kmers: List[set], ref_to_genus: List[int], n_genus: int,
                  k: int = KMER_SIZE) -> Dict[str, np.ndarray]:
    """Per-genus log-priors of kmer presence (Naive Bayes RDP).

    P(kmer w | genus g) = (m_w_g + 0.5) / (M_g + 1) where M_g is the
    number of references in genus g and m_w_g is the count of references
    in genus g containing kmer w.
    """
    M_g = np.zeros(n_genus, dtype=np.int64)
    for g in ref_to_genus:
        M_g[g] += 1
    # gather kmer presence counts per genus
    kmer_to_g_counts: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(n_genus, dtype=np.int32))
    for kset, g in zip(ref_kmers, ref_to_genus):
        for kmer in kset:
            kmer_to_g_counts[kmer][g] += 1
    # Compute log-likelihood matrix lazily — store rates instead.
    # Returns: kmer -> log( (m_w_g + 0.5) / (M_g + 1) ) for each genus
    rates: Dict[str, np.ndarray] = {}
    for kmer, counts in kmer_to_g_counts.items():
        rates[kmer] = np.log((counts + 0.5) / (M_g + 1))
    return rates, M_g


def _classify(query_seq: str, kmer_logp: Dict[str, np.ndarray], n_genus: int,
              rng: np.random.Generator, k: int = KMER_SIZE,
              n_boots: int = N_BOOTS) -> Tuple[int, np.ndarray]:
    """Classify a single query against the prior table.

    Returns (best_genus, bootstrap_counts (n_genus,)).
    """
    kmers = list(_kmer_set(query_seq, k))
    if not kmers:
        return -1, np.zeros(n_genus, dtype=np.int32)
    log_default = np.log(0.5 / 1)  # p for unseen kmer (no support anywhere)
    # Compute log-prob per genus given all kmers
    score = np.zeros(n_genus, dtype=np.float64)
    for kmer in kmers:
        if kmer in kmer_logp:
            score += kmer_logp[kmer]
        else:
            score += log_default
    best = int(np.argmax(score))

    # Bootstrap
    n_kmers = len(kmers)
    n_sample = max(1, n_kmers // 8)  # standard RDP bootstrap fraction
    boot_counts = np.zeros(n_genus, dtype=np.int32)
    for _ in range(n_boots):
        idx = rng.integers(0, n_kmers, size=n_sample)
        bscore = np.zeros(n_genus, dtype=np.float64)
        for ii in idx:
            kmer = kmers[ii]
            if kmer in kmer_logp:
                bscore += kmer_logp[kmer]
            else:
                bscore += log_default
        boot_counts[int(np.argmax(bscore))] += 1
    return best, boot_counts


def assign_taxonomy(seqs, ref_fasta: Union[str, Path],
                    minBoot: int = 50,                 # noqa: N803
                    tryRC: bool = False,               # noqa: N803
                    outputBootstraps: bool = False,    # noqa: N803
                    taxLevels=("Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"),  # noqa: N803
                    multithread: bool = False,
                    verbose: bool = False,
                    seed: Optional[int] = None):
    """Mirror assignTaxonomy: RDP naive-Bayes classifier with bootstraps."""
    from .io import get_sequences
    seqs = get_sequences(seqs)
    if min(len(s) for s in seqs) < MIN_TAX_LEN:
        if verbose:
            print(f"Some sequences shorter than {MIN_TAX_LEN} nts will not be classified.")

    refs, taxa = _read_ref_fasta(ref_fasta)
    keep = [i for i, r in enumerate(refs) if len(r) >= MIN_REF_LEN]
    refs = [refs[i] for i in keep]
    taxa = [taxa[i] for i in keep]

    if not refs:
        raise ValueError("No reference sequences after length filter.")

    # Crude format check — must use ';' separator
    if ";" not in taxa[0]:
        raise ValueError("Incorrect reference file format for assign_taxonomy.")

    # Pad shallow taxonomies with _DADA2_UNSPECIFIED
    tax_depth = [t.count(";") + 1 for t in taxa]
    td = max(tax_depth)
    for i in range(len(taxa)):
        if tax_depth[i] < td:
            taxa[i] = taxa[i] + ("_DADA2_UNSPECIFIED;" * (td - tax_depth[i]))

    genus_unq = []
    genus_to_idx: Dict[str, int] = {}
    ref_to_genus = []
    for t in taxa:
        if t not in genus_to_idx:
            genus_to_idx[t] = len(genus_unq)
            genus_unq.append(t)
        ref_to_genus.append(genus_to_idx[t])
    n_genus = len(genus_unq)

    # kmer table
    ref_kmers = [_kmer_set(r, KMER_SIZE) for r in refs]
    kmer_logp, _M_g = _genus_priors(ref_kmers, ref_to_genus, n_genus)

    rng = np.random.default_rng(seed)

    tax_out = np.full((len(seqs), td), None, dtype=object)
    boots_out = np.zeros((len(seqs), td), dtype=np.int32)

    for qi, s in enumerate(seqs):
        if len(s) < MIN_TAX_LEN:
            continue
        best_g, boots = _classify(s, kmer_logp, n_genus, rng,
                                  k=KMER_SIZE, n_boots=N_BOOTS)
        if tryRC:
            best_g_rc, boots_rc = _classify(rc(s), kmer_logp, n_genus, rng,
                                            k=KMER_SIZE, n_boots=N_BOOTS)
            if boots_rc.max() > boots.max():
                best_g, boots = best_g_rc, boots_rc
        if best_g < 0:
            continue
        # bootstrap support per level — fraction of bootstraps that ended
        # up in a genus sharing this prefix at level L
        chosen_levels = genus_unq[best_g].split(";")
        # for each level, count bootstraps whose chosen_genus shares this prefix
        for lvl in range(td):
            prefix = ";".join(chosen_levels[:lvl + 1])
            cnt = 0
            for g_idx in range(n_genus):
                if ";".join(genus_unq[g_idx].split(";")[:lvl + 1]) == prefix:
                    cnt += boots[g_idx]
            boots_out[qi, lvl] = cnt
            if cnt >= minBoot:
                tax_out[qi, lvl] = chosen_levels[lvl] if chosen_levels[lvl] != "_DADA2_UNSPECIFIED" else None

    import pandas as pd
    out_df = pd.DataFrame(tax_out, index=seqs, columns=list(taxLevels[:td]))
    if outputBootstraps:
        boots_df = pd.DataFrame(boots_out, index=seqs, columns=list(taxLevels[:td]))
        return {"tax": out_df, "boot": boots_df}
    return out_df


def assign_species(seqs, ref_fasta: Union[str, Path], allowMultiple: bool = False,  # noqa: N803
                   tryRC: bool = False,                                              # noqa: N803
                   verbose: bool = False):
    """Mirror assignSpecies: exact match to a species reference."""
    from .io import get_sequences
    seqs = get_sequences(seqs)
    refs, taxa = _read_ref_fasta(ref_fasta)
    # ref_to_species: a list of "Genus species" strings
    species = [t.strip() for t in taxa]
    # build hash by sequence
    ref_index: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(refs):
        ref_index[r].append(i)
    out = np.full((len(seqs), 2), None, dtype=object)
    for qi, s in enumerate(seqs):
        hit = ref_index.get(s.upper(), [])
        if not hit and tryRC:
            hit = ref_index.get(rc(s).upper(), [])
        if not hit:
            continue
        # parse species names
        names = [species[i] for i in hit]
        gen = sorted({n.split()[0] for n in names if " " in n})
        spe = sorted({n.split(maxsplit=1)[1] for n in names if " " in n})
        if not allowMultiple and (len(gen) > 1 or len(spe) > 1):
            continue
        out[qi, 0] = "/".join(gen) if gen else None
        out[qi, 1] = "/".join(spe) if spe else None
    import pandas as pd
    return pd.DataFrame(out, index=seqs, columns=["Genus", "Species"])


def add_species(taxa, ref_fasta: Union[str, Path], allowMultiple: bool = False,  # noqa: N803
                verbose: bool = False, **kwargs):
    """Append Species column from assignSpecies to an assignTaxonomy output."""
    seqs = list(taxa.index)
    sp = assign_species(seqs, ref_fasta, allowMultiple=allowMultiple, verbose=verbose, **kwargs)
    out = taxa.copy()
    out["Species"] = sp["Species"].values
    return out
