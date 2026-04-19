# dada2-py

A **pure-Python re-implementation of DADA2** (Callahan et al., *Nature Methods* 2016) for exact amplicon sequence variant (ASV) inference from amplicon sequencing data.

- AnnData / pandas-friendly — drop-in for downstream microbiome analysis
- **No `rpy2`**, no R install, no Bioconductor dependency
- Same API as the R [dada2](https://github.com/benjjneb/dada2) workflow
  (`filter_and_trim` → `learn_errors` → `dada` → `merge_pairs` → `make_sequence_table` → `remove_bimera_denovo` → `assign_taxonomy`)
- ASV-level identity matches R DADA2 on the canonical MiSeq SOP test data

> This is a **standalone mirror** of the canonical implementation that lives in [`omicverse`](https://github.com/Starlitnightly/omicverse). All algorithmic work is developed upstream in omicverse and synced here for users who want DADA2 without the full omicverse stack.

## Install

```bash
pip install pydada2
```

## Quick-start

```python
from pydada2 import (
    filter_and_trim, learn_errors, dada,
    merge_pairs, make_sequence_table,
    remove_bimera_denovo, assign_taxonomy,
)

# 1) Quality filter + trim
filter_and_trim(
    fwd="raw/F.fastq.gz", filt="filt/F.fastq.gz",
    rev="raw/R.fastq.gz", filt_rev="filt/R.fastq.gz",
    trunc_len=(240, 160), max_ee=(2, 2), trunc_q=2,
)

# 2) Learn the error model
errF = learn_errors("filt/F.fastq.gz")
errR = learn_errors("filt/R.fastq.gz")

# 3) Run the divisive amplicon denoising algorithm
ddF = dada("filt/F.fastq.gz", err=errF)
ddR = dada("filt/R.fastq.gz", err=errR)

# 4) Merge paired-end reads
mergers = merge_pairs(ddF, "filt/F.fastq.gz", ddR, "filt/R.fastq.gz")

# 5) Sample × sequence table
seqtab = make_sequence_table(mergers)

# 6) Chimera removal
seqtab_nochim = remove_bimera_denovo(seqtab, method="consensus")

# 7) Taxonomy assignment
taxa = assign_taxonomy(seqtab_nochim, ref_fasta="silva_nr99.fa.gz")
```

## What's included

| Module | Function | Purpose |
|---|---|---|
| `pydada2.filter` | `filter_and_trim`, `fastq_filter` | Quality filtering + trimming |
| `pydada2.io` | `derep_fastq`, `get_uniques`, `get_sequences` | FASTQ I/O + dereplication |
| `pydada2.errors` | `learn_errors`, `loess_errfun`, `inflate_err` | Error-rate model |
| `pydada2.align` | `nwalign`, `nwhamming` | Needleman-Wunsch ends-free alignment |
| `pydada2.kmers` | `kmer_dist`, `kord_dist` | k-mer distance pre-screen |
| `pydada2.dada` | `dada` | The divisive amplicon denoising algorithm |
| `pydada2.paired` | `merge_pairs` | Paired-end merging |
| `pydada2.seqtab` | `make_sequence_table`, `merge_sequence_tables`, `collapse_no_mismatch` | ASV table assembly |
| `pydada2.chimeras` | `is_bimera_denovo`, `remove_bimera_denovo` | Chimera detection |
| `pydada2.taxonomy` | `assign_taxonomy`, `assign_species`, `add_species` | RDP naive Bayes + exact-match species |

## Relationship to R DADA2

This is a Python port of the canonical R/C++ DADA2 package
([benjjneb/dada2](https://github.com/benjjneb/dada2)). All algorithmic
behaviour is checked against the R reference on the MiSeq SOP test
data (see `tests/test_r_parity.py`). The R reference is invoked from
`/scratch/users/steorra/env/CMAP` during testing.

## Citation

If you use this package, please cite the original DADA2 paper:

> Callahan, B.J. *et al.* **DADA2: High-resolution sample inference from Illumina amplicon data.** *Nature Methods* 13, 581–583 (2016).

and acknowledge omicverse / this repo for the Python port.

## License

LGPL-2 — matches upstream R DADA2.
