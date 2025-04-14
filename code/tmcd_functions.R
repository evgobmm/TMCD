#===============================================================================
# 0) A Function to Compute the Inverse Square Root of a Symmetric Pos. Def. Matrix
#===============================================================================
inverseSquareRootMatrix <- function(A) {
  #eigen-decomposition for such tasks
  e <- eigen(A, symmetric = TRUE)
  # We form A^{-1/2} = Q * diag(1/sqrt(lambda_i)) * Q^T
  invSqrtA <- e$vectors %*% diag(1 / sqrt(e$values)) %*% t(e$vectors)
  invSqrtA
}


#===============================================================================
# 1) Generating Tensor-Valued Samples
#===============================================================================
rTensorNorm <- function(n, M, Sigma1, Sigma2, Sigma3) {
  
  p1 <- nrow(Sigma1)
  p2 <- nrow(Sigma2)
  p3 <- nrow(Sigma3)
  
  # Precompute the matrix square roots of Sigma1, Sigma2, Sigma3

  e1 <- eigen(Sigma1, symmetric = TRUE)
  e2 <- eigen(Sigma2, symmetric = TRUE)
  e3 <- eigen(Sigma3, symmetric = TRUE)
  sqrtSigma1 <- e1$vectors %*% diag(sqrt(e1$values)) %*% t(e1$vectors)
  sqrtSigma2 <- e2$vectors %*% diag(sqrt(e2$values)) %*% t(e2$vectors)
  sqrtSigma3 <- e3$vectors %*% diag(sqrt(e3$values)) %*% t(e3$vectors)
  
  # Allocate a 4D array Z of size (n, p1, p2, p3) and fill it with i.i.d. N(0,1).
  Z <- array(rnorm(n * p1 * p2 * p3), dim = c(n, p1, p2, p3))
  
  # --- Mode-3 multiplication by Sigma3^(1/2) ---
  #   dim(Z) <- c(n*p1*p2, p3)  (reshape to a matrix)
  dim(Z) <- c(n * p1 * p2, p3)
  #   multiply on the right
  Z <- Z %*% sqrtSigma3
  #   revert dimension to (n, p1, p2, p3)
  dim(Z) <- c(n, p1, p2, p3)
  
  # --- Mode-2 multiplication by Sigma2^(1/2) ---
  #   aperm to shape (n, p1, p3, p2)
  Z <- aperm(Z, c(1, 2, 4, 3))
  #   reshape to (n*p1*p3, p2)
  dim(Z) <- c(n * p1 * p3, p2)
  #   multiply on the right
  Z <- Z %*% sqrtSigma2
  #   revert dimension to (n, p1, p3, p2)
  dim(Z) <- c(n, p1, p3, p2)
  
  # --- Mode-1 multiplication by Sigma1^(1/2) ---
  #   aperm to shape (n, p2, p3, p1)
  Z <- aperm(Z, c(1, 4, 3, 2))
  #   reshape to (n*p2*p3, p1)
  dim(Z) <- c(n * p2 * p3, p1)
  #   multiply on the right
  Z <- Z %*% sqrtSigma1
  #   revert dimension to (n, p2, p3, p1) then back to (n, p1, p2, p3)
  dim(Z) <- c(n, p2, p3, p1)
  Z <- aperm(Z, c(1, 4, 2, 3))
  
  # Finally, add the mean tensor M to each observation
  # "For all k = 1..n: X_k = Z_k + M"
  # The dimension of M is (p1, p2, p3), so we add it across the first dim
  Z <- sweep(Z, MARGIN = c(2, 3, 4), STATS = M, FUN = "+")
  
  # Return the generated sample
  Z
}


#===============================================================================
# 2) Computing Tensor Mahalanobis Distances
# We also store local contributions for each cell
#===============================================================================
computeTensorMD <- function(C, invSqrt1, invSqrt2, invSqrt3, returnContributions = FALSE) {
  # C is the 4D array of centered observations of size (n, p1, p2, p3).
  # invSqrt1, invSqrt2, invSqrt3 are the inverse square roots 
  # Finally, form D = \dddot{C} \odot \dddot{C} and sum up for TMD^2.
  
  # Extract dimension
  n  <- dim(C)[1]
  p1 <- dim(C)[2]
  p2 <- dim(C)[3]
  p3 <- dim(C)[4]
  
  # --- mode-4 multiplication by invSqrt3 ---
  dim(C) <- c(n * p1 * p2, p3)
  C <- C %*% invSqrt3
  dim(C) <- c(n, p1, p2, p3)
  
  # --- mode-3 multiplication by invSqrt2 ---
  C <- aperm(C, c(1, 2, 4, 3))
  dim(C) <- c(n * p1 * p3, p2)
  C <- C %*% invSqrt2
  dim(C) <- c(n, p1, p3, p2)
  
  # --- mode-2 multiplication by invSqrt1 ---
  C <- aperm(C, c(1, 4, 3, 2))
  dim(C) <- c(n * p2 * p3, p1)
  C <- C %*% invSqrt1
  dim(C) <- c(n, p2, p3, p1)
  # revert to shape (n, p1, p2, p3)
  C <- aperm(C, c(1, 4, 2, 3))
  
  # Now C is \dddot{C}. We form D = C \odot C (elementwise product)
  D <- C * C
  
  # TMD^2_k is the sum of D for each observation k
  
  dim(D) <- c(n, p1 * p2 * p3)
  TMDsq <- rowSums(D)
  
  # If we also want local contributions (the 4D array), we revert D's shape
  if (returnContributions) {
    dim(D) <- c(n, p1, p2, p3)
    return(list(TMDsq = TMDsq, contributions = D))
  } else {
    # Just return TMD^2
    return(TMDsq)
  }
}


#===============================================================================
# 3) Updating One Covariance Matrix in the Flip-Flop Step
#===============================================================================
updateOneCov <- function(C, invSqrt2, invSqrt3) {

  # Extract needed dims
  n  <- dim(C)[1]
  p1 <- dim(C)[2]
  p2 <- dim(C)[3]
  p3 <- dim(C)[4]
  
  # --- multiply along mode-4 by invSqrt3 ---
  dim(C) <- c(n * p1 * p2, p3)
  C <- C %*% invSqrt3
  dim(C) <- c(n, p1, p2, p3)
  
  # --- multiply along mode-3 by invSqrt2 ---
  C <- aperm(C, c(1, 2, 4, 3))
  dim(C) <- c(n * p1 * p3, p2)
  C <- C %*% invSqrt2
  dim(C) <- c(n, p1, p3, p2)
  
  # Compute Sigma1 = (1 / (n * p2 * p3)) * (mat %*% t(mat)),

  C <- aperm(C, c(2, 1, 4, 3))   # shape (p1, n, p2, p3)
  dim(C) <- c(p1, n * p2 * p3)
  
  Sigma1 <- (C %*% t(C)) / (n * p2 * p3)
  
  Sigma1
}


#===============================================================================
# 4) Flip-Flop MLE 
#===============================================================================
flipFlopMLE <- function(C1, C2, C3,
                        Sigma1init, Sigma2init, Sigma3init,
                        invSqrt2init, invSqrt3init,
                        maxIter, tol) {
  # C1: the subset of centered observations with dimension order (n, p1, p2, p3)
  # C2: the same subset but dimension order (n, p2, p3, p1)
  # C3: the same subset but dimension order (n, p3, p1, p2)
  #
  # Sigma1init, Sigma2init, Sigma3init: initial estimates for the 3 covariance mats
  # invSqrt2init, invSqrt3init: initial inverse square roots for Sigma2, Sigma3
  # maxIter, tol: stopping criteria
  #
  # The output must be the final Sigma1, Sigma2, Sigma3, and also
  # their inverse square roots for Sigma1 and Sigma2
  #
  # We also do a rescaling so that the [1,1]-entries of Sigma1 and Sigma2 are 1
  # after each full iteration.
  
  old1 <- Sigma1init
  old2 <- Sigma2init
  old3 <- Sigma3init
  invSqrt2 <- invSqrt2init
  invSqrt3 <- invSqrt3init
  
  
  for (iter in seq_len(maxIter)) {

    
    # --- Update Sigma1 ---

    Sigma1 <- updateOneCov(C1, invSqrt2, invSqrt3)
    # compute inverse sqrt of Sigma1
    invSqrt1 <- inverseSquareRootMatrix(Sigma1)
    
    # --- Update Sigma2 ---

    Sigma2 <- updateOneCov(C2, invSqrt3, invSqrt1)
    invSqrt2 <- inverseSquareRootMatrix(Sigma2)
    
    # --- Update Sigma3 ---
    Sigma3 <- updateOneCov(C3, invSqrt1, invSqrt2)
    
    # Rescaling step: enforce Sigma1[1,1] = 1, Sigma2[1,1] = 1
    # multiply Sigma3 by that product
    diag11_1 <- Sigma1[1,1]
    diag11_2 <- Sigma2[1,1]
    Sigma1 <- Sigma1 / diag11_1
    Sigma2 <- Sigma2 / diag11_2
    Sigma3 <- Sigma3 * (diag11_1 * diag11_2)
    
    if (iter == maxIter) {
      break
    }
    
    # We check if we should stop because of convergance
    frobDiff <- sum((Sigma1 - old1)^2) + sum((Sigma2 - old2)^2) + sum((Sigma3 - old3)^2)
    if (frobDiff < tol) {
      # converged
      break
    }
    
    
    # If not converged, we compute inverse sqrt of Sigma3 for next iteration
    invSqrt3 <- inverseSquareRootMatrix(Sigma3)
    old1 <- Sigma1
    old2 <- Sigma2
    old3 <- Sigma3
    
  }
  

  
  # Return them
  list(Sigma1 = Sigma1,
       Sigma2 = Sigma2,
       Sigma3 = Sigma3,
       invSqrt1 = invSqrt1,
       invSqrt2 = invSqrt2)
}

#===============================================================================
# 5) Single C-Step Refinement
#===============================================================================
cStep <- function(
  X,                   # Full sample of size (n, p1, p2, p3)
  C,                   # 4D tensor of size (n, p1, p2, p3), X centered by current subset's mean
  Sigma1, Sigma2, Sigma3,    # Current covariance estimates
  invSqrt1, invSqrt2, invSqrt3,  # Current inverse square roots
  alpha,              # Fraction of observations to keep: h = floor(alpha * n)
  maxIterC,           # Maximum number of C-step iterations
  tolC,               # Convergence tolerance for the C-step (based on ld)
  maxIterFF,          # Maximum iterations for the Flip-Flop MLE
  tolFF               # Tolerance for the Flip-Flop MLE
) {
  
  n  <- dim(X)[1]
  p1 <- dim(X)[2]
  p2 <- dim(X)[3]
  p3 <- dim(X)[4]
  
  # Subset size
  h <- floor(alpha * n)
  
  # A small helper for log-determinant-based objective
  computeLD <- function(S1, S2, S3) {
    (p2 * p3) * log(det(S1)) +
      (p1 * p3) * log(det(S2)) +
      (p1 * p2) * log(det(S3))
  }
  
  # Current ld from the provided covariance estimates
  ldOld <- computeLD(Sigma1, Sigma2, Sigma3)
  
  # Start the main C-step loop
  for (iter in seq_len(maxIterC)) {
    
  
    # Compute TMD^2 for all observations, using the full centered data 'C'
    
    TMDsAll <- computeTensorMD(C, invSqrt1, invSqrt2, invSqrt3)
    
    
    # Select top h with smallest TMD^2
    
    sortedIdx <- order(TMDsAll)
    subsetIndices <- sortedIdx[seq_len(h)]
    
    
    # Build the new subset's centered data in three dimension orders
    # Compute the mean of these h observations from the original data X

    Xsub <- X[subsetIndices, , , , drop = FALSE]
    subMean <- apply(Xsub, c(2, 3, 4), mean)

    # Center them by subMean

    C1 <- sweep(Xsub, MARGIN = c(2, 3, 4), STATS = subMean, FUN = "-")
    # C1:  (h, p1, p2, p3)
    
    C2 <- aperm(C1, c(1, 3, 4, 2))  # C2:  (h, p2, p3, p1)
    
    C3 <- aperm(C1, c(1, 4, 2, 3))  # C3:  (h, p3, p1, p2)
    
    
    # Run the Flip-Flop MLE on that subset
    
    ffRes <- flipFlopMLE(
      C1, C2, C3,
      Sigma1init = Sigma1,
      Sigma2init = Sigma2,
      Sigma3init = Sigma3,
      invSqrt2init = invSqrt2,
      invSqrt3init = invSqrt3,
      maxIter = maxIterFF,
      tol = tolFF
    )
    
    # Extract the updated estimates
    Sigma1 <- ffRes$Sigma1
    Sigma2 <- ffRes$Sigma2
    Sigma3 <- ffRes$Sigma3
    invSqrt1 <- ffRes$invSqrt1
    invSqrt2 <- ffRes$invSqrt2
    
    
    # Compute new ld
    
    ldNew <- computeLD(Sigma1, Sigma2, Sigma3)
    
   
    # Check for stopping conditions
    #    1) If we have reached maxIterC, we must stop
    #    2) If the improvement is under tolC, we stop
    
  
    if (iter == maxIterC || abs(ldNew - ldOld) < tolC) {
      break
    }
    
    
    ldOld <- ldNew
    
   
    # Re-center the entire sample
    
    C <- sweep(X, MARGIN = c(2, 3, 4), STATS = subMean, FUN = "-")
    
    # Also update invSqrt3 
    
    invSqrt3 <- inverseSquareRootMatrix(Sigma3)
  }
  
  
  list(
    Sigma1         = Sigma1,
    Sigma2         = Sigma2,
    Sigma3         = Sigma3,
    invSqrt1       = invSqrt1,
    invSqrt2       = invSqrt2,
    subsetIndices  = subsetIndices,
    TMDsAll        = TMDsAll,
    ld             = ldNew
  )
}

#===============================================================================
# 6) Full TMCD Procedure
#===============================================================================
tmcd <- function(
  X,                   # the full sample, a 4D array of size (n, p1, p2, p3)
  alpha,               # fraction in [0.5, 1)
  nSubsets,            # number of initial small subsets
  nBest,               # how many "best" solutions to keep for the full C-step
  maxIterCshort,       # max iterations in the short C-step
  maxIterFFshort,      # max iterations in the short flip-flop MLE (inside the short C-step)
  maxIterCfull,        # max iterations in the full C-step
  maxIterFFfull,       # max iterations in the full flip-flop MLE (inside the full C-step)
  tolC,                # convergence tolerance in the C-step
  tolFF,               # convergence tolerance in the flip-flop MLE
  beta                 # quantile (e.g. 0.99) for final outlier threshold
) {
  
  n  <- dim(X)[1]
  p1 <- dim(X)[2]
  p2 <- dim(X)[3]
  p3 <- dim(X)[4]
  
  # small-subset size s 
  s <- ceiling(p1 / (p2 * p3) + p2 / (p1 * p3) + p3 / (p1 * p2)) + 2
  
  
  # Generate all random subsets of size s (indices) 
  
  allSubsets <- replicate(nSubsets, sample.int(n, s), simplify = FALSE)
  

  # Store results (each is a list that includes an 'ld' for log-determinant)
  shortResults <- vector("list", nSubsets)
  
 
  # Run a short C-step on each subset
  
  for (i in seq_len(nSubsets)) {
      # subset indices
      idx <- allSubsets[[i]]
      
      # We form the mean from just these s observations
      xSub <- X[idx, , , , drop = FALSE]
      subMean <- apply(xSub, c(2,3,4), mean)
      
      # We create the needed subset in three dimension-order variants
      
      C1 <- sweep(xSub, MARGIN = c(2,3,4), STATS = subMean, FUN = "-")  # shape (s, p1, p2, p3)
      C2 <- aperm(C1, c(1, 3, 4, 2))                                    # shape (s, p2, p3, p1)
      C3 <- aperm(C1, c(1, 4, 2, 3))                                    # shape (s, p3, p1, p2)
      
      # We start with identity matrices (and their identity inverse roots)
      initSig1 <- diag(1, p1)
      initSig2 <- diag(1, p2)
      initSig3 <- diag(1, p3)
      initInvSqrt2 <- diag(1, p2)
      initInvSqrt3 <- diag(1, p3)
      
      # First, run a short flip-flop MLE on these s observations:
      shortMLE <- flipFlopMLE(
        C1, C2, C3,
        Sigma1init = initSig1,
        Sigma2init = initSig2,
        Sigma3init = initSig3,
        invSqrt2init = initInvSqrt2,
        invSqrt3init = initInvSqrt3,
        maxIter = maxIterFFshort,
        tol = tolFF
      )
      
      # Then center the entire sample by subMean
      C <- sweep(X, MARGIN = c(2,3,4), STATS = subMean, FUN = "-")
      
      # Compute current Sigma3^{-1/2} from the short MLE
      curInvSqrt3 <- inverseSquareRootMatrix(shortMLE$Sigma3)
      
      # Now pass everything to a short C-step
      shortRes <- cStep(
        X = X,
        C = C,
        Sigma1 = shortMLE$Sigma1,
        Sigma2 = shortMLE$Sigma2,
        Sigma3 = shortMLE$Sigma3,
        invSqrt1 = shortMLE$invSqrt1,
        invSqrt2 = shortMLE$invSqrt2,
        invSqrt3 = curInvSqrt3,
        alpha = alpha,
        maxIterC = maxIterCshort,
        tolC = tolC,
        maxIterFF = maxIterFFshort,
        tolFF = tolFF
      )
      
      shortResults[[i]] <- shortRes
  }
  
  

  # Pick the nBest solutions (based on 'ld')

  allLd <- sapply(shortResults, function(x) x$ld)
  rankLd <- order(allLd)  # smaller ld is better
  topIdx <- rankLd[seq_len(min(nBest, nSubsets))]
  
  
  # Do a full C-step on those top candidates
  
  fullResults <- vector("list", length(topIdx))
  for (j in seq_along(topIdx)) {
    # Short-run result: shortResults[[topIdx[j]]]
   
    xSub <- X[shortResults[[topIdx[j]]]$subsetIndices, , , , drop = FALSE]
    subMean <- apply(xSub, c(2,3,4), mean)
    C <- sweep(X, MARGIN = c(2,3,4), STATS = subMean, FUN = "-")
    
    
    invSqrt3 <- inverseSquareRootMatrix(shortResults[[topIdx[j]]]$Sigma3)
    
    fullRes <- cStep(
      X = X,
      C = C,
      Sigma1 = shortResults[[topIdx[j]]]$Sigma1,
      Sigma2 = shortResults[[topIdx[j]]]$Sigma2,
      Sigma3 = shortResults[[topIdx[j]]]$Sigma3,
      invSqrt1 = shortResults[[topIdx[j]]]$invSqrt1,
      invSqrt2 = shortResults[[topIdx[j]]]$invSqrt2,
      invSqrt3 = invSqrt3,
      alpha = alpha,
      maxIterC = maxIterCfull,
      tolC = tolC,
      maxIterFF = maxIterFFfull,
      tolFF = tolFF
    )
    fullResults[[j]] <- fullRes
  }
  
  # pick the best (lowest ld) among the full results
  allLdFull <- sapply(fullResults, function(x) x$ld)
  bestFullIdx <- which.min(allLdFull)
  bestRaw <- fullResults[[bestFullIdx]]
  
  # Rescaling
  
  # gamma(alpha) = alpha / F_{chi^2_(p1*p2*p3+2)}( qchisq(alpha, df=p1*p2*p3) )
  
  dfMain <- p1 * p2 * p3
  dfPlus <- dfMain + 2
  chiAlpha <- qchisq(alpha, dfMain)
  cdfVal   <- pchisq(chiAlpha, dfPlus)
  gammaAlpha <- alpha / cdfVal
  
  # multiply Sigma3 by gammaAlpha
  
  S1 <- bestRaw$Sigma1
  S2 <- bestRaw$Sigma2
  S3 <- bestRaw$Sigma3 * gammaAlpha
  
  invSqrt2 <- bestRaw$invSqrt2
  invSqrt3 <- inverseSquareRootMatrix(S3)
  
  
  
  TMDsqAll <- bestRaw$TMDsAll
  
  # outlier criterion: TMD^2 / gammaAlpha < chi^2_{(dfMain), beta}
  cutoff <- qchisq(beta, dfMain)
  goodSet <- which((TMDsqAll / gammaAlpha) < cutoff)
  
  # We union them with the bestRaw subset
  finalGood <- union(bestRaw$subsetIndices, goodSet)
  outliers <- setdiff(seq_len(n), finalGood)
  
  # Final mean
  Xgood <- X[finalGood, , , , drop = FALSE]
  M <- apply(Xgood, c(2,3,4), mean)
  
  # define alphaHat
  alphaHat <- length(finalGood) / n
  
  # final flip-flop MLE on the "good" subset
  
  C1 <- sweep(Xgood, MARGIN = c(2,3,4), STATS = M, FUN = "-")
  C2 <- aperm(C1, c(1, 3, 4, 2))
  C3 <- aperm(C1, c(1, 4, 2, 3))
  
  ffFinal <- flipFlopMLE(
    C1 = C1,
    C2 = C2,
    C3 = C3,
    Sigma1init = S1,
    Sigma2init = S2,
    Sigma3init = S3,
    invSqrt2init = invSqrt2,
    invSqrt3init = invSqrt3,
    maxIter = maxIterFFfull,
    tol = tolFF
  )
  
  # final Sigmas
  S1 <- ffFinal$Sigma1
  S2 <- ffFinal$Sigma2
  S3 <- ffFinal$Sigma3
  
  # now multiply Sigma3 by gamma(alphaHat)
  chiAlphaHat <- qchisq(alphaHat, dfMain)
  cdfValHat   <- pchisq(chiAlphaHat, dfPlus)
  gammaAlphaHat <- alphaHat / cdfValHat
  S3 <- S3 * gammaAlphaHat
  

  
  
  list(
    M = M,  
    Sigma1 = S1,
    Sigma2 = S2,
    Sigma3 = S3,
    outliers = outliers,
    finalGood = finalGood
  )
}