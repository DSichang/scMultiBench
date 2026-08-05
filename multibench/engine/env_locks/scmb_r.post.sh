#!/usr/bin/env bash
# Restore packages that `conda env export` cannot see.
#
# rliger is installed into scmb_r's R library by install.packages(), so conda has
# no record of it and the lockfile alone rebuilds an env without it. UINMF then
# fails with "there is no package called 'rliger'" - after the env has already
# built successfully, which makes it look like a method bug rather than a
# provisioning gap.
#
# Verified present in the working env as rliger 2.0.1 (CRAN, R 4.5.3).
set -euo pipefail

Rscript -e '
if (!requireNamespace("rliger", quietly = TRUE)) {
  if (!requireNamespace("remotes", quietly = TRUE))
    install.packages("remotes", repos = "https://cloud.r-project.org")
  remotes::install_version("rliger", version = "2.0.1",
                           repos = "https://cloud.r-project.org", upgrade = "never")
}
cat("rliger present:", requireNamespace("rliger", quietly = TRUE), "\n")
'
