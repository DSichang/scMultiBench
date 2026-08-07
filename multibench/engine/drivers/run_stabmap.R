#!/usr/bin/env Rscript
# Package-side thin driver for StabMap.
#
# WHY THIS EXISTS: upstream tools_scripts/StabMap/main_StabMap.Rmd defines
# run_StabMap()/helpers but has NO top-level call (it is meant to be `source()`d
# from an external R driver, per the commented example in that file). The
# package builder only does `Rscript <entrypoint> <args>`, which would just
# define functions and exit. This driver sources the UPSTREAM .Rmd verbatim
# (so the method script stays byte-identical to upstream) and invokes
# run_StabMap() with file_paths assembled from CLI args.
#
# Args (all flagged; --rna/--adt/--atac repeatable, one file each):
#   --script_dir <dir>     dir containing main_StabMap.Rmd + util.R (upstream)
#   --save_path  <dir>     output dir (embedding.h5 written here)
#   --reference  <name>    StabMap reference label, e.g. "data3"
#   --rna <file> [--rna <file> ...]
#   --adt <file> [...]
#   --atac <file> [...]

args <- commandArgs(trailingOnly = TRUE)
script_dir <- NULL; save_path <- NULL; reference <- NULL
rna <- character(0); adt <- character(0); atac <- character(0)
i <- 1
while (i <= length(args)) {
  a <- args[i]
  if (a == "--script_dir")      { script_dir <- args[i + 1]; i <- i + 2 }
  else if (a == "--save_path")  { save_path  <- args[i + 1]; i <- i + 2 }
  else if (a == "--reference")  { reference  <- args[i + 1]; i <- i + 2 }
  else if (a == "--rna")        { rna  <- c(rna,  args[i + 1]); i <- i + 2 }
  else if (a == "--adt")        { adt  <- c(adt,  args[i + 1]); i <- i + 2 }
  else if (a == "--atac")       { atac <- c(atac, args[i + 1]); i <- i + 2 }
  else { i <- i + 1 }
}
stopifnot(!is.null(script_dir), !is.null(save_path), !is.null(reference))

# absolutize file paths BEFORE setwd (they may be relative to the caller cwd)
# "None" marks a batch with no file in this slot; it is a placeholder, not a
# path, so it must survive absolutisation to keep the per-batch positions.
is_gap  <- function(p) p %in% c("None", "NULL", "NA", "none")
abspath <- function(p) if (is_gap(p)) p else normalizePath(p, mustWork = TRUE)
if (length(rna))  rna  <- vapply(rna,  abspath, "")
if (length(adt))  adt  <- vapply(adt,  abspath, "")
if (length(atac)) atac <- vapply(atac, abspath, "")
save_path <- ifelse(grepl("/$", save_path), save_path, paste0(save_path, "/"))

old <- getwd()
setwd(script_dir)                 # so main_StabMap.Rmd`s source("util.R") resolves
source("main_StabMap.Rmd")        # defines run_StabMap + helpers (upstream, unmodified)


# Build a per-batch list preserving GAPS: the token "None" (or "NULL"/"NA") marks
# a batch that lacks this modality, and becomes a real NULL at that position.
# StabMap indexes these lists by batch, so a collapsed list silently pairs a
# modality with the wrong batch.
slot_list <- function(v) {
  if (length(v) == 0) return(NULL)
  out <- vector("list", length(v))
  for (i in seq_along(v)) {
    if (!is_gap(v[i])) out[[i]] <- v[i]
  }
  out
}

file_paths <- list()
if (length(rna))  file_paths$rna_path  <- slot_list(rna)
if (length(adt))  file_paths$adt_path  <- slot_list(adt)
if (length(atac)) file_paths$atac_path <- slot_list(atac)

run_StabMap(file_paths, save_path, reference)
setwd(old)
cat("finish\n")
