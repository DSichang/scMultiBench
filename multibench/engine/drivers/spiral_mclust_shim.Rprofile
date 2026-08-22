## SPIRAL mclust shim (injected via R_PROFILE_USER by engine/drivers/run_spiral.py)
## Reason: rpy2's Python-closure call of mclust::Mclust passes the data matrix as an
## anonymous value, which breaks mclust 6.x internal dimnames<- ("length of 'dimnames'
## [2] not equal to array extent"). Binding the matrix to a local R symbol first fixes it.
## This defines Mclust in .GlobalEnv so robjects.r['Mclust'] resolves here.
local({
  .spiral_Mclust <- function(data, G = NULL, modelNames = NULL, ...) {
    D <- as.matrix(data)
    storage.mode(D) <- "double"
    mclust::Mclust(D, G = G, modelNames = modelNames, ...)
  }
  assign("Mclust", .spiral_Mclust, envir = globalenv())
})
