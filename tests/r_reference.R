# Run R DADA2 on the canonical sam1F fixture and emit results to TSV
# so the Python side can diff it.
library(dada2)
fn <- "/scratch/users/steorra/analysis/omicverse_dev/cache/dada2_src/inst/extdata/sam1F.fastq.gz"
out_dir <- "/scratch/users/steorra/analysis/omicverse_dev/py-dada2/tests/r_out"
dir.create(out_dir, recursive=TRUE, showWarnings=FALSE)

drp <- derepFastq(fn)
cat("R uniques:", length(drp$uniques), "  reads:", sum(drp$uniques), "\n")

# Same flat error matrix as the Python smoke test
err <- matrix(1e-3, nrow=16, ncol=41)
rownames(err) <- c("A2A","A2C","A2G","A2T","C2A","C2C","C2G","C2T","G2A","G2C","G2G","G2T","T2A","T2C","T2G","T2T")
colnames(err) <- as.character(0:40)
for(i in 0:3) err[i*4 + i + 1, ] <- 1 - 3*1e-3

dd <- dada(drp, err=err, multithread=FALSE, verbose=TRUE)

cat("R clusters:", length(dd$denoised), "\n")

write.table(
  data.frame(
    cluster_idx = seq_along(dd$denoised),
    sequence = dd$clustering$sequence,
    abundance = dd$clustering$abundance,
    n0 = dd$clustering$n0,
    nunq = dd$clustering$nunq,
    birth_from = dd$clustering$birth_from,
    birth_pval = dd$clustering$birth_pval
  ),
  file=file.path(out_dir, "dada_sam1F_flat_err.tsv"),
  sep="\t", quote=FALSE, row.names=FALSE
)

# Also dump dereplication for input parity
write.table(
  data.frame(sequence=names(drp$uniques), abundance=as.integer(drp$uniques)),
  file=file.path(out_dir, "derep_sam1F.tsv"),
  sep="\t", quote=FALSE, row.names=FALSE
)
cat("Wrote:", file.path(out_dir, "dada_sam1F_flat_err.tsv"), "\n")
