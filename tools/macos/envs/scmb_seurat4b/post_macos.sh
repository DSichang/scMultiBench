#!/usr/bin/env bash
# scmb_seurat4b on osx-arm64: bioconda has no osx-arm64 build of Bioconductor 3.16 (R 4.2), so the
# lock's Bioconductor packages are compiled from Bioconductor's own source archives, at the lock's
# versions, in dependency order; then Signac 1.10.0 from CRAN's archive. No dependency is pulled in:
# every CRAN package they need is in the conda part.
set -euo pipefail
# Rsamtools 2.14.0 includes R's headers (which define a macro `length`) before C++ headers; the libc++ of
# today's clang uses that word in <locale> and the compile stops. Reading <locale> first avoids it.
export R_MAKEVARS_USER="$(cd "$(dirname "$0")" && pwd)/Makevars"
# Rhdf5lib 1.20.0 configures its own copy of HDF5 1.10.7, whose config.sub (2018) does not know the
# name conda's compiler activation gives this machine (arm64-apple-darwin20.0.0); it knows aarch64.
export host_alias=aarch64-apple-darwin20.0.0 build_alias=aarch64-apple-darwin20.0.0
Rscript - <<'RS'
options(warn = 1, Ncpus = 6)
bioc <- "https://bioconductor.org/packages/3.16"
pk <- c(BiocGenerics="0.44.0", S4Vectors="0.36.0", IRanges="2.32.0", zlibbioc="1.44.0", XVector="0.38.0",
        GenomeInfoDbData="1.2.9", GenomeInfoDb="1.34.9", Biostrings="2.66.0", GenomicRanges="1.50.0",
        BiocParallel="1.32.5", Rhtslib="2.0.0", Rsamtools="2.14.0", Biobase="2.58.0", MatrixGenerics="1.10.0",
        DelayedArray="0.24.0", SummarizedExperiment="1.28.0", SingleCellExperiment="1.20.0",
        Rhdf5lib="1.20.0", rhdf5filters="1.10.0", rhdf5="2.42.0", HDF5Array="1.26.0")
dir.create(tmp <- file.path(tempdir(), "src"))
for (p in names(pk)) {
  if (requireNamespace(p, quietly = TRUE) && as.character(packageVersion(p)) == pk[[p]]) next
  f <- sprintf("%s_%s.tar.gz", p, pk[[p]])
  urls <- c(sprintf("%s/bioc/src/contrib/%s", bioc, f), sprintf("%s/bioc/src/contrib/Archive/%s/%s", bioc, p, f),
            sprintf("%s/data/annotation/src/contrib/%s", bioc, f))
  got <- FALSE
  for (u in urls) {
    ok <- tryCatch(download.file(u, file.path(tmp, f), quiet = TRUE, mode = "wb") == 0, error = function(e) FALSE,
                   warning = function(w) FALSE)
    if (ok) { got <- TRUE; cat("source:", u, "\n"); break }
  }
  if (!got) stop("no source archive for ", f)
  install.packages(file.path(tmp, f), repos = NULL, type = "source")
  if (as.character(packageVersion(p)) != pk[[p]]) stop(p, " did not install")
}
if (!requireNamespace("Signac", quietly = TRUE)) {
  u <- "https://cloud.r-project.org/src/contrib/Archive/Signac/Signac_1.10.0.tar.gz"
  download.file(u, f <- file.path(tmp, "Signac_1.10.0.tar.gz"), quiet = TRUE, mode = "wb")
  install.packages(f, repos = NULL, type = "source")
}
for (p in c(names(pk), "Signac", "Seurat")) cat(p, as.character(packageVersion(p)), "\n")
suppressMessages({library(rhdf5); library(Seurat); library(Signac); library(HDF5Array)})
cat("seurat4b OK\n")
RS
