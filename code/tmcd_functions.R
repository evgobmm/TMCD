#===============================================================================
# 0) A Function to Compute the Inverse Square Root of a Symmetric Pos. Def. Matrix
#===============================================================================
# Fast inverse square-root using Cholesky
inverseSquareRootMatrix <- function(A) {
  # A is symmetric positive-definite:  A = R^T R,  R is upper-triangular (chol).
  # A^{-1/2} = (R^{-1})^T because  (R^{-1})^T R^{-1} = A^{-1}.
  tryCatch({
    R    <- chol(A)
    Rinv <- backsolve(R, diag(nrow(A)))   # Solve R * X = I  =>  X = R^{-1}.
    t(Rinv)
  }, error = function(e) diag(nrow(A)))   # on any error return identity
}



#===============================================================================
# 1) Generating Tensor-Valued Samples
#===============================================================================

rTensorNorm <- function(n, M, Sigma1, Sigma2, Sigma3) {
  
  
  # helper: matrix square-root via Cholesky  (Sigma = R^T R  ->  sqrt = R^T)
  
  cholSqrt <- function(S) t(chol(S))
  
  p1 <- nrow(Sigma1)
  p2 <- nrow(Sigma2)
  p3 <- nrow(Sigma3)
  
  # pre-compute square-roots
  sqrt1 <- cholSqrt(Sigma1)         
  sqrt2 <- cholSqrt(Sigma2)          
  sqrt3 <- cholSqrt(Sigma3)         
  
  
  # i.i.d. standard-normal core  —  layout  (p3 , p2 , p1 , n)
  
  Z <- array(rnorm(n * p1 * p2 * p3), dim = c(p3, p2, p1, n))
  
  
  # mode-3 multiplication  (first dimension : p3)
  dim(Z) <- c(p3, p2 * p1 * n)       # p3 x (rest)
  Z <- sqrt3 %*% Z                   # left-multiply by sqrt(Sigma3)
  dim(Z) <- c(p3, p2, p1, n)         # restore 4D shape
  
  # bring p2 to the front  …  (p2 , p3 , p1 , n)
  Z <- aperm(Z, c(2, 1, 3, 4))       # first aperm
  
  # mode-2 multiplication  (now first dimension is p2)
  dim(Z) <- c(p2, p3 * p1 * n)
  Z <- sqrt2 %*% Z
  dim(Z) <- c(p2, p3, p1, n)
  
  
  # mode-1 multiplication  (dimension p1)
  Z <- aperm(Z, c(3, 2, 1, 4))        # p1  p3  p2  n second aperm
  dim(Z) <- c(p1, p3 * p2 * n)        # p1 x (rest)
  Z <- sqrt1 %*% Z                    # left-multiply by sqrt(Sigma1)
  dim(Z) <- c(p1, p3, p2, n)
  
  
  
  # final layout we want as output  (n , p1 , p2 , p3)
  # ---------------------------------------------------------------------------
  Z <- aperm(Z, c(4, 1, 3, 2))       
  
  # ---------------------------------------------------------------------------
  # add mean tensor M  (broadcast over first dimension)
  # ---------------------------------------------------------------------------
  Z <- sweep(Z, MARGIN = c(2, 3, 4), STATS = M, FUN = "+")
  
  Z
}


#===============================================================================
# 2) Computing Tensor Mahalanobis Distances
# We also store local contributions for each cell
#===============================================================================

computeTensorMD <- function(C,                         # p3 x (p2*p1*n)
                            invSqrt1, invSqrt2, invSqrt3,
                            n, p1, p2, p3,
                            returnContributions = FALSE) {
  
  # --- mode-3 whitening ------------------------------------------------------
  
  C <- invSqrt3 %*% C                                 # p3 x (p2*p1*n)
  
  
  dim(C) <- c(p3, p2, p1, n)                          #reshape
  # bring mode-2 (p2) forward
  C <- aperm(C, c(2, 1, 3, 4))                        # (p2 , p3 , p1 , n)
  
  # --- mode-2 whitening ------------------------------------------------------
  dim(C) <- c(p2, p3 * p1 * n)                        # p2 x (…)
  C <- invSqrt2 %*% C                                 # p2 x (…)
  dim(C) <- c(p2, p3, p1, n)
  # bring mode-1 (p1) forward
  C <- aperm(C, c(3, 2, 1, 4))                        # (p1 , p3 , p2 , n)
  
  # --- mode-1 whitening ------------------------------------------------------
  dim(C) <- c(p1, p3 * p2 * n)                        # p1 x (…)
  C <- invSqrt1 %*% C                                 # p1 x (…)
  dim(C) <- c(p1, p3, p2, n)                          # 4D form
  
  # Squared distances:  flatten first three dims, sum over columns for each obs.
  dim(C) <- c(p1 * p3 * p2, n)                        # (cells , n)
  D <- C * C                                          # element-wise square
  TMDsq <- colSums(D)                                 # MD^2 for n obs
  
  if (!returnContributions)
    return(TMDsq)
  
  # local contributions wanted:  n x p1 x p2 x p3
  dim(D) <- c(p1, p3, p2, n)                          # back to 4D
  D <- aperm(D, c(4, 1, 3, 2))                        # (n , p1 , p2 , p3)
  
  list(TMDsq = TMDsq, contributions = D)
}







#===============================================================================
# 3) Updating One Covariance Matrix in the Flip-Flop Step
#===============================================================================


updateOneCov <- function(C, invSqrt2, invSqrt3, n, p1, p2, p3) {
  # C starts as  p3 x (p2 * n * p1)  
  C <- invSqrt3 %*% C                         # whiten mode-3, still p3 x ...
  
  dim(C) <- c(p3, p2, n, p1)                  # reshape:  p3  p2  n  p1
  C <- aperm(C, c(2, 1, 3, 4))                # reorder:  p2  p3  n  p1
  dim(C) <- c(p2, p3 * n * p1)                # mode-2 unfolding:  p2 x ...
  
  C <- invSqrt2 %*% C                         # whiten mode-2
  dim(C) <- c(p2 * p3 * n, p1)                # unfolding
  
  # Empirical covariance: (t(C) %*% C) / (n * p2 * p3); result is p1 x p1.
  crossprod(C) / (n * p2 * p3)
}


#===============================================================================
# 4) Flip-Flop MLE 
#===============================================================================
flipFlopMLE <- function(C1, C2, C3,
                        n, p1, p2, p3,
                        Sigma1init, Sigma2init, Sigma3init,
                        invSqrt2init, invSqrt3init,
                        maxIter, tol) {
  
  old1 <- Sigma1init; old2 <- Sigma2init; old3 <- Sigma3init
  invSqrt2 <- invSqrt2init; invSqrt3 <- invSqrt3init
  
  for (iter in seq_len(maxIter)) {
    
    # --- update Sigma1 -------------------------------------------------------
    Sigma1   <- updateOneCov(C1, invSqrt2, invSqrt3, n, p1, p2, p3)
    invSqrt1 <- inverseSquareRootMatrix(Sigma1)
    
    # --- update Sigma2 (rotate modes) ---------------------------------------
    Sigma2   <- updateOneCov(C2, invSqrt3, invSqrt1, n,
                             p1 = p2, p2 = p3, p3 = p1)
    invSqrt2 <- inverseSquareRootMatrix(Sigma2)
    
    # --- update Sigma3 ------------------------------------------------------
    Sigma3   <- updateOneCov(C3, invSqrt1, invSqrt2, n,
                             p1 = p3, p2 = p1, p3 = p2)
    
    
    if (!all(is.finite(Sigma1))) Sigma1 <- diag(p1)
    if (!all(is.finite(Sigma2))) Sigma2 <- diag(p2)
    if (!all(is.finite(Sigma3))) Sigma3 <- diag(p3)
    
    # --- rescale so that Sigma1[1,1] = Sigma2[1,1] = 1 ----------------------
    d1 <- Sigma1[1, 1]; d2 <- Sigma2[1, 1]
    #Sigma1 <- Sigma1 / d1
    #Sigma2 <- Sigma2 / d2
    #Sigma3 <- Sigma3 * (d1 * d2)           # keeps overall scale consistent
    
    if (d1 > 1e-12 && d2 > 1e-12) {       
      Sigma1 <- Sigma1 / d1
      Sigma2 <- Sigma2 / d2
      Sigma3 <- Sigma3 * (d1 * d2)     # keeps overall scale consistent
    }
    
    
    # --- convergence check --------------------------------------------------
    if (iter == maxIter) break
    
    
    
    f2 <- function(A) norm(A, "F")^2   # Frobenious norm ^2
    
    err <- f2(Sigma1 - old1) +
      f2(Sigma2 - old2) +
      f2(Sigma3 - old3)
    
    if (err < tol) break
    
    invSqrt3 <- inverseSquareRootMatrix(Sigma3)
    old1 <- Sigma1; old2 <- Sigma2; old3 <- Sigma3
  }
  
  list(Sigma1 = Sigma1,
       Sigma2 = Sigma2,
       Sigma3 = Sigma3,
       invSqrt1 = invSqrt1,
       invSqrt2 = invSqrt2)
}

#===============================================================================
# 5) Single C-Step Refinement
#===============================================================================
# (X : p3 x p2 x p1 x n,  C : p3 x (p2*p1*n))
cStep <- function(X,                    # full sample  (p3 , p2 , p1 , n)
                  C,                    # centred and unfolded  (p3 , p2*p1*n)
                  Sigma1, Sigma2, Sigma3,
                  invSqrt1, invSqrt2, invSqrt3,
                  alpha,
                  maxIterC,
                  tolC,
                  maxIterFF,
                  tolFF) {
  p3 <- dim(X)[1];  p2 <- dim(X)[2];  p1 <- dim(X)[3];  n <- dim(X)[4]
  h  <- floor(alpha * n)                               # subset size
  
  ldFun <- function(S1, S2, S3) {
    
    logdet <- function(S) sum(log(diag(chol(S))))
    
    (p2 * p3) * logdet(S1) +
      (p1 * p3) * logdet(S2) +
      (p1 * p2) * logdet(S3)
  }
  
  
  ldOld <- ldFun(Sigma1, Sigma2, Sigma3)
  
  for (iter in seq_len(maxIterC)) {
    # Mahalanobis distances for all n samples  (input C already centred)
    
    TMDsAll <- computeTensorMD(C,
                               invSqrt1, invSqrt2, invSqrt3,
                               n, p1, p2, p3)
    
    idx <- order(TMDsAll)[seq_len(h)]   # keep h smallest
    
    #---- build centred subset tensors -------------------------------------------
    
    Xsub <- X[ , , , idx, drop = FALSE]              # p3  p2  p1  h
    subMean <- apply(Xsub, c(1, 2, 3), mean)         # p3  p2  p1
    
    C1 <- sweep(Xsub, c(1, 2, 3), subMean, "-")      # centred tensor p3  p2  p1  h
    
    # C1     : rows = p3 (mode-3), cols enumerate  p2 x h x p1
    C1 <- aperm(C1, c(1, 2, 4, 3))                # p3  p2  h  p1
    
    
    # C2     : rows = p1,  cols enumerate  p3 × h × p2
    C2 <- aperm(C1, c(4, 1, 3, 2))                   # p1  p3  h  p2
    dim(C2) <- c(p1, p3 * h * p2)                    # p1 x (p3*h*p2)
    
    # C3     : rows = p2,  cols enumerate  p1 × h × p3
    C3 <- aperm(C1, c(2, 4, 3, 1))                   # p2  p1  h  p3
    dim(C3) <- c(p2, p1 * h * p3)                    # p2 x (p1*h*p3)
    
    dim(C1) <- c(p3, p2 * h * p1)                 # p3 x (p2*h*p1)
    
    # ------------------------------------------------------------------------
    # flip–flop MLE on the subset
    # ------------------------------------------------------------------------
    ff <- flipFlopMLE(C1, C2, C3,
                      n  = h, p1 = p1, p2 = p2, p3 = p3,
                      Sigma1init = Sigma1,
                      Sigma2init = Sigma2,
                      Sigma3init = Sigma3,
                      invSqrt2init = invSqrt2,
                      invSqrt3init = invSqrt3,
                      maxIter = maxIterFF,
                      tol      = tolFF)
    
    Sigma1 <- ff$Sigma1;  Sigma2 <- ff$Sigma2;  Sigma3 <- ff$Sigma3
    invSqrt1 <- ff$invSqrt1;  invSqrt2 <- ff$invSqrt2
    
    ldNew <- ldFun(Sigma1, Sigma2, Sigma3)
    
    # ------------------------------------------------------------------------
    # stop if no improvement or out of iterations
    # ------------------------------------------------------------------------
    if (iter == maxIterC || abs(ldNew - ldOld) < tolC) break
    ldOld <- ldNew
    
    # ------------------------------------------------------------------------
    # re-centre entire sample with new mean, rebuild C for next loop
    # ------------------------------------------------------------------------
    C <- sweep(X, c(1, 2, 3), subMean, "-")        # p3 p2 p1 n
    dim(C) <- c(p3, p2 * p1 * n)          # unfold to p3 x (…)
    
    invSqrt3 <- inverseSquareRootMatrix(Sigma3)
  }
  
  list(Sigma1        = Sigma1,
       Sigma2        = Sigma2,
       Sigma3        = Sigma3,
       invSqrt1      = invSqrt1,
       invSqrt2      = invSqrt2,
       subsetIndices = idx,
       TMDsAll       = TMDsAll,
       ld            = ldNew)
}



#===============================================================================
# 6) Full TMCD Procedure
#===============================================================================
tmcd <- function(X,
                 alpha,
                 nSubsets,
                 nBest,
                 maxIterCshort,
                 maxIterFFshort,
                 maxIterCfull,
                 maxIterFFfull,
                 tolC,
                 tolFF,
                 beta) {
  
  # convert to internal layout  (p3 , p2 , p1 , n)
  
  X <- aperm(X, c(4, 3, 2, 1))               # now p3 p2 p1 n
  
  p3 <- dim(X)[1];  p2 <- dim(X)[2];  p1 <- dim(X)[3];  n <- dim(X)[4]
  
  # small subset size  
  s <- ceiling(p1 / (p2 * p3) + p2 / (p1 * p3) + p3 / (p1 * p2)) + 2
  
  # identity matrices prepared once
  I1 <- diag(1, p1);  I2 <- diag(1, p2);  I3 <- diag(1, p3)
  
  # all random subsets of size s  (indices along obs dimension)
  allSubsets <- replicate(nSubsets, sample.int(n, s), simplify = FALSE)
  
  shortResults <- vector("list", nSubsets)
  
  # run short C-step on each random subset
  
  for (i in seq_len(nSubsets)) {
    idx <- allSubsets[[i]]                     # length s
    
    # subset mean
    Xsub  <- X[ , , , idx, drop = FALSE]       # p3 p2 p1 s
    subMu <- apply(Xsub, c(1, 2, 3), mean)     # p3 p2 p1
    
    # centred tensor and three unfoldings
    C1 <- sweep(Xsub, c(1, 2, 3), subMu, "-")  # p3 p2 p1 s
    C1 <- aperm(C1, c(1, 2, 4, 3))             # p3 p2 s p1
    C2 <- aperm(C1, c(4, 1, 3, 2))             # p1 p3 s p2
    C3 <- aperm(C1, c(2, 4, 3, 1))             # p2 p1 s p3
    
    dim(C1) <- c(p3, p2 * s * p1)
    dim(C2) <- c(p1, p3 * s * p2)
    dim(C3) <- c(p2, p1 * s * p3)
    
    # short MLE
    shortMLE <- flipFlopMLE(C1, C2, C3,
                            n = s, p1 = p1, p2 = p2, p3 = p3,
                            Sigma1init = I1, Sigma2init = I2, Sigma3init = I3,
                            invSqrt2init = I2, invSqrt3init = I3,
                            maxIter = maxIterFFshort,
                            tol      = tolFF)
    
    # build C for the whole sample (centered by subMu)
    C <- sweep(X, c(1, 2, 3), subMu, "-")         # p3 p2 p1 n
    dim(C) <- c(p3, p2 * p1 * n)  # p3 x (…)
    
    # current inv sqrt of Sigma3
    InvSqrt3 <- inverseSquareRootMatrix(shortMLE$Sigma3)
    
    shortResults[[i]] <- cStep(X         = X,
                               C         = C,
                               Sigma1    = shortMLE$Sigma1,
                               Sigma2    = shortMLE$Sigma2,
                               Sigma3    = shortMLE$Sigma3,
                               invSqrt1  = shortMLE$invSqrt1,
                               invSqrt2  = shortMLE$invSqrt2,
                               invSqrt3  = InvSqrt3,
                               alpha     = alpha,
                               maxIterC  = maxIterCshort,
                               tolC      = tolC,
                               maxIterFF = maxIterFFshort,
                               tolFF     = tolFF)
  }
  
  
  # Keep nBest solutions with smallest log-det
  
  k <- min(nBest, nSubsets)
  allLd <- vapply(shortResults, `[[`, numeric(1), "ld")
  bestIdx <- order(allLd)[seq_len(k)] 
  
  fullResults <- vector("list", length(bestIdx))
  for (j in seq_along(bestIdx)) {
    res <- shortResults[[bestIdx[j]]]
    
    idx <- res$subsetIndices
    Xsub  <- X[ , , , idx, drop = FALSE]
    subMu <- apply(Xsub, c(1, 2, 3), mean)
    
    C <- sweep(X, c(1, 2, 3), subMu, "-")
    dim(C) <- c(p3, p2 * p1 * n)
    
    invSqrt3 <- inverseSquareRootMatrix(res$Sigma3)
    
    fullResults[[j]] <- cStep(X         = X,
                              C         = C,
                              Sigma1    = res$Sigma1,
                              Sigma2    = res$Sigma2,
                              Sigma3    = res$Sigma3,
                              invSqrt1  = res$invSqrt1,
                              invSqrt2  = res$invSqrt2,
                              invSqrt3  = invSqrt3,
                              alpha     = alpha,
                              maxIterC  = maxIterCfull,
                              tolC      = tolC,
                              maxIterFF = maxIterFFfull,
                              tolFF     = tolFF)
  }
  
  
  # choose the best full result
  
  fullLd <- vapply(fullResults, `[[`, numeric(1), "ld")
  best <- fullResults[[ which.min(fullLd) ]]
  
  # Rescale Sigma3 by gamma(alpha)
  
  df  <- p1 * p2 * p3
  df2 <- df + 2
  gam <- alpha / pchisq(qchisq(alpha, df), df2)
  
  
  best$Sigma3 <- best$Sigma3 * gam
  
  invSqrt3 <- inverseSquareRootMatrix(best$Sigma3)
  
  
  # Outlier identification
  
  
  # threshold
  limit   <- qchisq(beta, df) * gam
  
  
  
  ## good / bad mask for all n observations
  isGood <- best$TMDsAll < limit        # logical length-n
  isGood[best$subsetIndices] <- TRUE    # keep initial subset
  
  ## final index sets
  finalGood <- which(isGood)
  outliers  <- which(!isGood)
  
  
  # Final estimates on good observations
  
  Xgood <- X[ , , , finalGood, drop = FALSE]
  Mint  <- apply(Xgood, c(1, 2, 3), mean)            # p3 p2 p1
  m <- length(finalGood)
  alphaHat <- m / n
  
  
  C1 <- sweep(Xgood, c(1, 2, 3), Mint, "-")          # p3 p2 p1 m
  C1 <- aperm(C1, c(1, 2, 4, 3));                    # p3 p2 m p1
  C2 <- aperm(C1, c(4, 1, 3, 2)); 
  C3 <- aperm(C1, c(2, 4, 3, 1)); 
  
  
  dim(C1) <- c(p3, p2 * m * p1)
  dim(C2) <- c(p1, p3 * m * p2)
  dim(C3) <- c(p2, p1 * m * p3)
  
  
  ff <- flipFlopMLE(C1, C2, C3,
                    n = m, p1 = p1, p2 = p2, p3 = p3,
                    Sigma1init = best$Sigma1,
                    Sigma2init = best$Sigma2,
                    Sigma3init = best$Sigma3,
                    invSqrt2init = best$invSqrt2,
                    invSqrt3init = invSqrt3,
                    maxIter = maxIterFFfull,
                    tol      = tolFF)
  
  
  gamHat <- alphaHat / pchisq(qchisq(alphaHat, df), df2)
  ff$Sigma3 <- ff$Sigma3 * gamHat
  
  # convert mean back to user layout  (p1 , p2 , p3)
  Mout <- aperm(Mint, c(3, 2, 1))
  
  list(M         = Mout,
       Sigma1    = ff$Sigma1,
       Sigma2    = ff$Sigma2,
       Sigma3    = ff$Sigma3,
       outliers  = outliers,
       finalGood = finalGood)
}


#===============================================================================
# Kullback-Leibler Divergence
#===============================================================================
KL <- function(Sigma1_1, Sigma2_1, Sigma3_1,
               Sigma1_2, Sigma2_2, Sigma3_2)
{
  #     KL( (Sigma1_1, Sigma2_1, Sigma3_1) || (Sigma1_2, Sigma2_2, Sigma3_2) )
  #
  #  = (1/2) * [
  #     tr[(Sigma3_2^-1 * Sigma3_1)] * tr[(Sigma2_2^-1 * Sigma2_1)] * tr[(Sigma1_2^-1 * Sigma1_1)]
  #     - p1*p2 * ln det(Sigma3_2^-1 * Sigma3_1)
  #     - p1*p3 * ln det(Sigma2_2^-1 * Sigma2_1)
  #     - p2*p3 * ln det(Sigma1_2^-1 * Sigma1_1)
  #     - p1*p2*p3
  #  ]
  #
  
  p1 <- nrow(Sigma1_1)
  p2 <- nrow(Sigma2_1)
  p3 <- nrow(Sigma3_1)
  
  A3 <- solve(Sigma3_2, Sigma3_1)
  A2 <- solve(Sigma2_2, Sigma2_1)
  A1 <- solve(Sigma1_2, Sigma1_1)
  
  tr3 <- sum(diag(A3))
  tr2 <- sum(diag(A2))
  tr1 <- sum(diag(A1))
  
  det3 <- det(A3)
  det2 <- det(A2)
  det1 <- det(A1)
  
  val <- 0.5 * (
    (tr3 * tr2 * tr1)
    - (p1 * p2) * log(det3)
    - (p1 * p3) * log(det2)
    - (p2 * p3) * log(det1)
    - (p1 * p2 * p3)
  )
  
  return(val)
}

#===============================================================================
# Covariance Matrices Generation
#===============================================================================

generateCovMatrices <- function(p1, p2, p3,
                                rangeVar = c(0.1, 1),
                                covMethod = "onion") {
  if (!requireNamespace("clusterGeneration", quietly = TRUE))
    stop("Package 'clusterGeneration' is required")
  
  # Sigma1: random positive-definite matrix
  Sigma1 <- clusterGeneration::genPositiveDefMat(dim = p1,
                                                 covMethod = covMethod,
                                                 rangeVar  = rangeVar)$Sigma
  
  # Sigma2: 0.7 on the diagonal, 0.5 off diagonal
  Sigma2 <- matrix(0.5, nrow = p2, ncol = p2)
  diag(Sigma2) <- 0.7
  
  # Sigma3: 0.7 on the diagonal, 0.5^|i-j| off diagonal
  Sigma3 <- matrix(0, nrow = p3, ncol = p3)
  for (i in seq_len(p3)) {
    for (j in seq_len(p3)) {
      if (i == j) {
        Sigma3[i, j] <- 0.7
      } else {
        Sigma3[i, j] <- 0.5 ^ abs(i - j)
      }
    }
  }
  
  list(Sigma1 = Sigma1,
       Sigma2 = Sigma2,
       Sigma3 = Sigma3)
}



#===============================================================================
# Tensor Maximum Likelihood Estimation (TMLE) wrapper
#===============================================================================


TMLE <- function(X,
                 efficient = FALSE,          # FALSE:  X is  (n , p1 , p2 , p3)
                 # TRUE :  X is  (p3 , p2 , n , p1)
                 maxIter = 100,
                 tol     = 1e-3) {
  
  # ----------  reshape to internal C1 layout  (p3 , p2 , n , p1) ----------
  if (!efficient) {
    X <- aperm(X, c(4, 3, 1, 2))             # bring to p3 p2 n p1
  }
  p3 <- dim(X)[1] ; p2 <- dim(X)[2]
  n  <- dim(X)[3] ; p1 <- dim(X)[4]
  

  
  
  # ---------- compute mean tensor and centre the sample -------------------
  Mu  <- apply(X, c(1, 2, 4), mean)          # p3 p2 p1   (average over n)
  C1  <- sweep(X, c(1, 2, 4), Mu, "-")       # centred data p3 p2 n p1
  
 
  
  
  # ----------  build three mode–unfoldings ---------------------------------

  C2 <- aperm(C1, c(4, 1, 3, 2))              # p1 p3 n p2
  C3 <- aperm(C1, c(2, 4, 3, 1))              # p2 p1 n p3

  
  dim(C1) <- c(p3, p2 * n * p1)
  dim(C2) <- c(p1, p3 * n * p2)
  dim(C3) <- c(p2, p1 * n * p3)

  # ---------- flip–flop maximum-likelihood estimation ---------------------
  I1 <- diag(1, p1)
  I2 <- diag(1, p2)
  I3 <- diag(1, p3)
  
  ff <- flipFlopMLE(C1, C2, C3,
                    n = n, p1 = p1, p2 = p2, p3 = p3,
                    Sigma1init = I1,
                    Sigma2init = I2,
                    Sigma3init = I3,
                    invSqrt2init = I2,
                    invSqrt3init = I3,
                    maxIter = maxIter,
                    tol      = tol)
  
  # ---------- arrange mean tensor back to caller layout -------------------
  if (!efficient) {
      Mu <- aperm(Mu, c(3, 2, 1))             # p1 p2 p3  (match input)
  }
  
  list(M      = Mu,
       Sigma1 = ff$Sigma1,
       Sigma2 = ff$Sigma2,
       Sigma3 = ff$Sigma3)
}

#===============================================================================
# Tensor Mahalanobis Distance wrapper
#===============================================================================

tMD <- function(X, M, Sigma1, Sigma2, Sigma3, returnContributions = FALSE) {
  # X is (n, p1, p2, p3); M is (p1, p2, p3)
  
  n  <- dim(X)[1]
  p1 <- dim(X)[2]
  p2 <- dim(X)[3]
  p3 <- dim(X)[4]
  
  invSqrt1 <- inverseSquareRootMatrix(Sigma1)
  invSqrt2 <- inverseSquareRootMatrix(Sigma2)
  invSqrt3 <- inverseSquareRootMatrix(Sigma3)
  

  
  # centre X in its original layout (n , p1 , p2 , p3)
  Xcentered <- sweep(X, c(2, 3, 4), M, FUN = "-") 
  
  # reorder the centred tensor to internal order (p3 , p2 , p1 , n)
  C <- aperm(Xcentered, c(4, 3, 2, 1))
  
  # unfold for computeTensorMD    rows = p3,  cols enumerate p2 × p1 × n
  dim(C) <- c(p3, p2 * p1 * n)
  
  # compute squared tensor Mahalanobis distances (and optionally contributions)
  result <- computeTensorMD(
    C,
    invSqrt1, invSqrt2, invSqrt3,
    n, p1, p2, p3,
    returnContributions = returnContributions
  )
  
  result
}


#===============================================================================
# A function to print side-by-side matrices (Sigma1, Sigma2, Sigma3) 
#===============================================================================



compareSigmas <- function(res, digits = 3, col_gap = 6) {
  
  # Helper: round and right‑justify each numeric value
  
  fmt <- function(x, width) {
    format(
      round(x, digits),
      justify = "right",
      width   = width,
      trim    = TRUE
    )
  }
  
  # Build a single Sigma block and print it
  printSigma <- function(true_mat, est_mat, name) {
    cat("\n", name, " (scaled): Left = True, Right = Estimated\n", sep = "")
    
    # Determine cell width from the largest rounded number
    cell_width <- max(nchar(fmt(c(true_mat, est_mat), 0)))
    gap <- strrep(" ", col_gap)  # the requested extra spacing
    
    for (row_i in seq_len(nrow(true_mat))) {
      pair_text <- sapply(seq_len(ncol(true_mat)), function(col_j) {
        paste0(
          fmt(true_mat[row_i, col_j], cell_width), " | ",
          fmt(est_mat [row_i, col_j], cell_width)
        )
      })
      cat(paste(pair_text, collapse = gap), "\n")
    }
  }
  
  
  printSigma(res$Sigma1_true_scaled, res$Sigma1_est, "Sigma1")
  printSigma(res$Sigma2_true_scaled, res$Sigma2_est, "Sigma2")
  printSigma(res$Sigma3_true_scaled, res$Sigma3_est, "Sigma3")
}


