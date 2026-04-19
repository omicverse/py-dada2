# Full R DADA2 reference: filter -> derep -> dada -> mergePairs ->
# makeSequenceTable -> isBimeraDenovoTable on the canonical paired
# fixtures (sam1F+sam1R, sam2F+sam2R).
suppressMessages(library(dada2))

extdata <- "/scratch/users/steorra/analysis/omicverse_dev/cache/dada2_src/inst/extdata"
out_dir <- "/scratch/users/steorra/analysis/omicverse_dev/py-dada2/tests/r_out"
dir.create(out_dir, recursive=TRUE, showWarnings=FALSE)

# Flat error matrix shared across all per-sample dada calls.
err <- matrix(1e-3, nrow=16, ncol=41)
rownames(err) <- c("A2A","A2C","A2G","A2T","C2A","C2C","C2G","C2T","G2A","G2C","G2G","G2T","T2A","T2C","T2G","T2T")
colnames(err) <- as.character(0:40)
for(i in 0:3) err[i*4 + i + 1, ] <- 1 - 3*1e-3

samples <- c("sam1", "sam2")
for(s in samples) {
  fF <- file.path(extdata, paste0(s, "F.fastq.gz"))
  fR <- file.path(extdata, paste0(s, "R.fastq.gz"))

  drpF <- derepFastq(fF)
  drpR <- derepFastq(fR)
  ddF <- dada(drpF, err=err, multithread=FALSE, verbose=FALSE)
  ddR <- dada(drpR, err=err, multithread=FALSE, verbose=FALSE)

  for(side in c("F", "R")) {
    dd <- get(paste0("dd", side))
    write.table(
      data.frame(
        sequence = dd$clustering$sequence,
        abundance = dd$clustering$abundance,
        n0 = dd$clustering$n0,
        nunq = dd$clustering$nunq,
        birth_from = dd$clustering$birth_from,
        birth_pval = dd$clustering$birth_pval
      ),
      file=file.path(out_dir, sprintf("dada_%s%s.tsv", s, side)),
      sep="\t", quote=FALSE, row.names=FALSE
    )
  }

  # mergePairs
  mer <- mergePairs(ddF, drpF, ddR, drpR, minOverlap=12, maxMismatch=0,
                    returnRejects=FALSE, verbose=FALSE)
  if(nrow(mer) > 0) {
    write.table(mer[, c("sequence","abundance","forward","reverse","nmatch","nmismatch","nindel","prefer","accept")],
                file=file.path(out_dir, sprintf("merge_%s.tsv", s)),
                sep="\t", quote=FALSE, row.names=FALSE)
  } else {
    cat("No merged pairs for", s, "\n")
  }
}

cat("Wrote R reference TSVs to", out_dir, "\n")
