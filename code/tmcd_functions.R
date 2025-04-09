###################################################################################
# Tensor Minimum Covariance Determinant (TMCD) procedure for 3rd-order Tensor Data
# ---------------------------------------------------------------------------
# This script provides a robust approach for estimating the location and 
# scatter matrices of 3D (third-order) tensor data under a separable 
# covariance structure: 
#   Cov(X) = Sigma1 \otimes Sigma2 \otimes Sigma3
# where Sigma1, Sigma2, Sigma3 are covariance matrices corresponding to modes 
# (dimensions) p, q, and r respectively.
#
# Major steps/components included in this script:
#  1. rtensornorm           - Generate samples from a 3D tensor normal distribution
#  2. tensorMD_vectorized   - Compute the (squared) Mahalanobis distances for 
#                             tensor observations in a vectorized manner
#  3. update_Sigma_l        - Update a single covariance factor (Sigma1, Sigma2, 
#                             or Sigma3) in the tensor MLE "flip-flop" procedure
#  4. tensor_MLE            - Perform flip-flop MLE for the tensor normal 
#                             distribution (optionally with shrinkage and 
#                             normalization)
#  5. tensor_C_step         - Execute a single C-step in the TMCD algorithm to 
#                             refine a given subset and re-estimate parameters
#  6. tensor_MCD            - Main procedure, performing multiple short runs 
#                             on small subsets, refining via full C-steps, and 
#                             reweighting/rescaling for consistency
#  7. tensor_KL_divergence  - Compute the Kullback-Leibler divergence between 
#                             two separable 3D tensor normal distributions
###################################################################################
library(clusterGeneration)


#-------------------------------------------------------------------------------
#' Generate samples from a 3D tensor normal distribution
#'
#' @description 
#' This function simulates data from a separable 3D tensor normal distribution 
#' with mean tensor \code{Mu} and mode-wise covariance factors \code{Sigma1}, 
#' \code{Sigma2}, \code{Sigma3}. It applies sequential mode multiplications 
#' to a standard normal array, thereby achieving the desired separable covariance
#' structure \(\Sigma_1 \otimes \Sigma_2 \otimes \Sigma_3\) 
#' without computing Kronecker products.
#'
#' This version carefully minimizes permutations.
#' The mean tensor \code{Mu} is added vectorized via \code{sweep()}, avoiding 
#' explicit loops.
#'
#' @param n Integer. Number of samples (tensor observations) to generate.
#' @param Mu 3D array of dimension \code{[p, q, r]} representing the mean 
#'           tensor. If \code{NULL} or all zeros, a zero tensor of shape 
#'           \code{[p, q, r]} is used.
#' @param Sigma1 A \code{p x p} positive-definite matrix (covariance for mode-1).
#' @param Sigma2 A \code{q x q} positive-definite matrix (covariance for mode-2).
#' @param Sigma3 A \code{r x r} positive-definite matrix (covariance for mode-3).
#'
#' @return A 4D array \code{X_array} of shape \code{[n, p, q, r]}, where each 
#'         slice along the first dimension is one tensor observation.
#'
#' @details
#' Internally, the function:
#' \enumerate{
#'   \item Performs eigen-decomposition on each \(\Sigma_i\) to obtain 
#'         \(\Sigma_i^{1/2}\).
#'   \item Generates an i.i.d. standard normal array \([n,p,q,r]\).
#'   \item Multiplies sequentially:
#'         \itemize{
#'           \item \(\Sigma_3^{1/2}\) with no permutation (\(r\) is last).
#'           \item \(\Sigma_2^{1/2}\) with one permutation to put \(q\) last, 
#'                 resulting in \([n,p,r,q]\).
#'           \item \(\Sigma_1^{1/2}\) from the shape \([n,p,r,q]\) with a single
#'                 permutation to put \(p\) last, then revert to \([n,p,q,r]\).
#'         }
#'   \item Adds the mean \(\Mu\) in a fully vectorized manner.
#' }
#'
#' @export
rtensornorm <- function(n, Mu = NULL, Sigma1, Sigma2, Sigma3) {
  #---------------------------------------------------------------------------
  # Retrieve p, q, r from Sigma1, Sigma2, Sigma3.
  # Initialize Mu if needed.
  #---------------------------------------------------------------------------
  p <- nrow(Sigma1)
  q <- nrow(Sigma2)
  r <- nrow(Sigma3)
  
  if (is.null(Mu) || all(Mu == 0)) {
    Mu <- array(0, dim = c(p, q, r))
  }
  
  #---------------------------------------------------------------------------
  # Eigen-decompose each Sigma to get Sigma_sqrt1, Sigma_sqrt2, Sigma_sqrt3.
  #---------------------------------------------------------------------------
  eig1 <- eigen(Sigma1, symmetric = TRUE)
  eig2 <- eigen(Sigma2, symmetric = TRUE)
  eig3 <- eigen(Sigma3, symmetric = TRUE)
  
  Sigma_sqrt1 <- eig1$vectors %*% diag(sqrt(eig1$values)) %*% t(eig1$vectors)
  Sigma_sqrt2 <- eig2$vectors %*% diag(sqrt(eig2$values)) %*% t(eig2$vectors)
  Sigma_sqrt3 <- eig3$vectors %*% diag(sqrt(eig3$values)) %*% t(eig3$vectors)
  
  #--------------------------------------------------------------------------
  # Directly generate i.i.d. normal data in [n*p*q, r] for Mode-3 multiplication.
  #--------------------------------------------------------------------------
  Z_mat <- matrix(rnorm(n * p * q * r), nrow = n * p * q, ncol = r)  # => [n*p*q, r]
  Z_mat <- Z_mat %*% Sigma_sqrt3                       # multiply by Sigma3^{1/2}
  Z_array <- array(Z_mat, dim = c(n, p, q, r))         # => [n, p, q, r]
  
  #---------------------------------------------------------------------------
  # Mode-2 multiplication (dimension q).
  #---------------------------------------------------------------------------
  Z_temp <- aperm(Z_array, c(1, 2, 4, 3))              # => [n, p, r, q]
  Z_mat  <- matrix(Z_temp, nrow = n * p * r, ncol = q) # => [n*p*r, q]
  Z_mat  <- Z_mat %*% Sigma_sqrt2                      # multiply by Sigma2^{1/2}
  Z_array <- array(Z_mat, dim = c(n, p, r, q))         # => [n, p, r, q]
  
  #---------------------------------------------------------------------------
  # Mode-1 multiplication (dimension p).
  #---------------------------------------------------------------------------
  Z_temp <- aperm(Z_array, c(1, 3, 4, 2))              # => [n, r, q, p]
  
  Z_mat  <- matrix(Z_temp, nrow = n * r * q, ncol = p) # => [n*r*q, p]
  Z_mat  <- Z_mat %*% Sigma_sqrt1                      # multiply by Sigma1^{1/2}
  
  Z_temp <- array(Z_mat, dim = c(n, r, q, p))          # => [n, r, q, p]
  
  Z_array <- aperm(Z_temp, c(1, 4, 3, 2))              # => [n, p, q, r]
  
  #---------------------------------------------------------------------------
  # Add mean tensor Mu over [p, q, r] for all n.
  #---------------------------------------------------------------------------
  Z_array <- sweep(Z_array, MARGIN = c(2, 3, 4), STATS = Mu, FUN = "+")
  
  
  return(Z_array)
}


#-------------------------------------------------------------------------------
#' Vectorized TMD^2 (and optionally Shapley values) for all observations
#'
#' @description
#' Computes the (squared) tensor Mahalanobis distance (TMD^2) for each 3D tensor 
#' observation in \code{X} against a specified mean tensor \code{Mu} and 
#' separable covariance factors \(\Sigma_1, \Sigma_2, \Sigma_3\). By default, 
#' this function returns a length-\code{n} numeric vector of TMD^2 values. 
#' Optionally (if \code{shapley = TRUE}), the function instead returns a 4D array 
#' of \emph{cellwise Shapley values}, which decompose \(\mathrm{TMD}^2\) into 
#' individual contributions from each cell \code{(j, k, l)} of each observation.
#'
#' @param X A 4D array of shape \code{[n, p, q, r]}, where each slice along 
#'          the first dimension (\code{n}) is a single 3D tensor observation.
#' @param Mu A 3D array of shape \code{[p, q, r]} representing the mean tensor.
#' @param Sigma1 A \code{p x p} covariance matrix or its inverse (depending 
#'               on \code{inverted}).
#' @param Sigma2 A \code{q x q} covariance matrix or its inverse (depending 
#'               on \code{inverted}).
#' @param Sigma3 A \code{r x r} covariance matrix or its inverse (depending 
#'               on \code{inverted}).
#' @param inverted Logical. \code{FALSE} (default) if \code{Sigma1}, \code{Sigma2}, 
#'                 and \code{Sigma3} are covariance matrices to be inverted 
#'                 internally; \code{TRUE} if they are already inverted (i.e., 
#'                 are precision matrices).
#' @param shapley Logical flag. If \code{FALSE} (default), the function returns 
#'                the numeric vector of TMD^2 values (one per observation). If 
#'                \code{TRUE}, the function returns a 4D array of shape 
#'                \code{[n, p, q, r]} containing the cellwise Shapley values.
#'
#' @return 
#' \itemize{
#'   \item If \code{shapley = FALSE}: a numeric vector of length \code{n}, where 
#'         each entry is the TMD^2 for one observation.
#'   \item If \code{shapley = TRUE}: a 4D array of shape \code{[n, p, q, r]}, 
#'         where each element \code{(i,j,k,l)} gives the Shapley value (i.e., 
#'         cellwise contribution) for cell \code{(j,k,l)} in observation \code{i}. 
#'         Summing over \code{j,k,l} for observation \code{i} recovers its 
#'         \eqn{\mathrm{TMD}^2}.
#' }
#'
#' @details
#' 
#' When \code{shapley=FALSE}:
#' \enumerate{
#'   \item (Optional) Invert each \(\Sigma\) if \code{inverted=FALSE}.
#'   \item Center the data by subtracting \code{Mu}.
#'   \item Perform three consecutive mode-multiplications by 
#'         \(\Sigma_3^{-1}, \Sigma_2^{-1}, \Sigma_1^{-1}\).
#'   \item Take the row-wise dot product of the centered array with the 
#'         transformed array to get \(\mathrm{TMD}^2\).
#' }
#' 
#' When \code{shapley=TRUE}, the same steps are performed except that, instead 
#' of summing over all cells for each observation, we keep the cellwise products 
#' \code{(X_centered * X_transformed)}. This yields the Shapley contributions 
#' \(\phi_{j,k,l}(\mathbf{X}_i)\) for each cell \(\{j,k,l\}\).
#'
#' @export
tensorMD_vectorized <- function(X, 
                                Mu, 
                                Sigma1, 
                                Sigma2, 
                                Sigma3, 
                                inverted = FALSE,
                                shapley = FALSE)
{
  #---------------------------------------------------------------------------
  # Extract dimensions: [n, p, q, r]
  #---------------------------------------------------------------------------
  n <- dim(X)[1]
  p <- dim(X)[2]
  q <- dim(X)[3]
  r <- dim(X)[4]
  
  #---------------------------------------------------------------------------
  # Possibly invert each Sigma if necessary
  #---------------------------------------------------------------------------
  if (!inverted) {
    invert_symmetric <- function(S) {
      eig <- eigen(S, symmetric = TRUE)
      eig$vectors %*% diag(1 / eig$values) %*% t(eig$vectors)
    }
    Sigma1_inv <- invert_symmetric(Sigma1)  
    Sigma2_inv <- invert_symmetric(Sigma2)  
    Sigma3_inv <- invert_symmetric(Sigma3)  
  } else {
    Sigma1_inv <- Sigma1
    Sigma2_inv <- Sigma2
    Sigma3_inv <- Sigma3
  }
  
  #---------------------------------------------------------------------------
  # Center the data: X_centered = X - Mu
  # => [n, p, q, r]
  #---------------------------------------------------------------------------
  X_centered <- sweep(X, MARGIN = c(2, 3, 4), STATS = Mu, FUN = "-")
  
  #---------------------------------------------------------------------------
  # Mode-3 multiplication by Sigma3_inv
  # => reshape [n*p*q, r], multiply, reshape back to [n, p, q, r]
  #---------------------------------------------------------------------------
  Xc_mode3_mat <- matrix(X_centered, nrow = n * p * q, ncol = r)  # => [n*p*q, r]
  Xc_mode3_mat <- Xc_mode3_mat %*% Sigma3_inv                     # multiply by Sigma3^{-1}
  Xc_mode3     <- array(Xc_mode3_mat, dim = c(n, p, q, r))        # => [n, p, q, r]
  
  #---------------------------------------------------------------------------
  # Mode-2 multiplication by Sigma2_inv
  # => permute to [n, p, r, q], reshape, multiply, reshape back
  #---------------------------------------------------------------------------
  perm_mode2       <- aperm(Xc_mode3, c(1, 2, 4, 3))              # => [n, p, r, q]
  perm_mode2_mat   <- matrix(perm_mode2, nrow = n * p * r, ncol = q)  # => [n*p*r, q]
  perm_mode2_mat   <- perm_mode2_mat %*% Sigma2_inv               # multiply by Sigma2^{-1}
  perm_mode2_array <- array(perm_mode2_mat, dim = c(n, p, r, q))  # => [n, p, r, q]
  
  #---------------------------------------------------------------------------
  # Mode-1 multiplication by Sigma1_inv
  # => permute to [n, q, r, p], reshape, multiply, reshape back
  #---------------------------------------------------------------------------
  perm_mode1       <- aperm(perm_mode2_array, c(1, 4, 3, 2))    # => [n, q, r, p]
  perm_mode1_mat   <- matrix(perm_mode1, nrow = n * q * r, ncol = p)  # => [n*q*r, p]
  perm_mode1_mat   <- perm_mode1_mat %*% Sigma1_inv                  # multiply by Sigma1^{-1}
  perm_mode1_array <- array(perm_mode1_mat, dim = c(n, q, r, p))     # => [n, q, r, p]
  
  #---------------------------------------------------------------------------
  # Revert to original orientation => [n, p, q, r]
  #---------------------------------------------------------------------------
  X_transformed <- aperm(perm_mode1_array, c(1, 4, 2, 3))        # => [n, p, q, r]
  
  #---------------------------------------------------------------------------
  # If shapley=TRUE, return the 4D array of cellwise products
  # => [n, p, q, r]
  # Otherwise, sum over all cells (p, q, r) to get TMD^2 for each observation
  #---------------------------------------------------------------------------
  
  
  # Compute the 4D array of cellwise contributions:
  ShapValues <- X_centered * X_transformed  # [n, p, q, r] 
  
  if (shapley) {
    
    
    # Return the entire array of Shapley contributions:
    return(ShapValues)
    
  } else {
    
    # Flatten each observation into a single row, so we get an n x (p*q*r) matrix:
    D_mat <- matrix(ShapValues, nrow = n, ncol = p * q * r)
    
    # Sum each row to get the TMD^2 per observation:
    MD2 <- rowSums(D_mat)
    
    return(MD2)
  }  
  
  
  
}#-------------------------------------------------------------------------------
#' Update a single covariance factor \(\Sigma_l\) in the "flip-flop" step
#'
#' @description
#' A helper function used by the tensor MLE procedure. It updates exactly one 
#' of the three covariance matrices (\(\Sigma_1\), \(\Sigma_2\), or \(\Sigma_3\)) 
#' in the presence of the other two being fixed. This is part of the alternating 
#' ("flip-flop") procedure for maximum likelihood estimation under the separable 
#' 3D tensor covariance model.
#'
#' @param l An integer in \code{1,2,3}, indicating which \(\Sigma_l\) to update.
#' @param X A 4D array of shape \code{[n, p, q, r]}, where the first 
#'          dimension (\code{n}) indexes observations. The remaining 
#'          dimensions correspond to modes \code{p, q, r}. 
#'          (Already arranged so that \code{X[i,,,]} is the 
#'          \code{i}-th observation.)
#' @param Mu A 3D array of shape \code{[p, q, r]} representing the current 
#'           estimate of the mean tensor.
#' @param Sigma1 The current estimate of \(\Sigma_1\) (a \code{p x p} matrix).
#' @param Sigma2 The current estimate of \(\Sigma_2\) (a \code{q x q} matrix).
#' @param Sigma3 The current estimate of \(\Sigma_3\) (a \code{r x r} matrix).
#'
#' @return The updated covariance factor, a matrix of dimension \code{p x p}, 
#'         \code{q x q}, or \code{r x r}, depending on \code{l}.
#'
#' @details
#' \enumerate{
#'   \item Depending on \code{l}, selects the two other covariance factors 
#'         (\(\Sigma_i\), \(\Sigma_j\)) to treat as fixed.
#'   \item Computes the inverse of their square roots.
#'   \item Centers the data by subtracting \code{Mu}.
#'   \item Applies two sequential mode transformations (involving the fixed 
#'         factors) to the data via permutations and matrix multiplications 
#'         \emph{on the right}.
#'   \item Finally, reshapes the transformed data along mode \code{l} 
#'         and computes \(\Sigma_l\) via a cross-product, normalized by 
#'         \(\bigl(n \cdot \dim_i \cdot \dim_j\bigr)\).
#' }
#'
#' This function is used internally by the \code{tensor_MLE} flip-flop procedure.
#' @export
update_Sigma_l <- function(l, X, Mu, Sigma1, Sigma2, Sigma3) {
  
  #---------------------------------------------------------------------------
  # Extract dimensions: [n, p, q, r]
  #---------------------------------------------------------------------------
  n <- dim(X)[1]
  p <- dim(X)[2]
  q <- dim(X)[3]
  r <- dim(X)[4]
  
  #---------------------------------------------------------------------------
  # Decide which Sigma factors are fixed based on l, and which dimensions 
  # to transform (transform_dims). Also identify dim_i, dim_j for normalization.
  #
  #   l=1 => updating Sigma1 => (Sigma2, Sigma3) fixed => transform_dims = c(4,3)
  #   l=2 => updating Sigma2 => (Sigma1, Sigma3) fixed => transform_dims = c(4,2)
  #   l=3 => updating Sigma3 => (Sigma1, Sigma2) fixed => transform_dims = c(3,2)
  #
  # dim_i, dim_j correspond to the sizes of the two fixed modes.
  #---------------------------------------------------------------------------
  if (l == 1) {
    SigmaA <- Sigma2    # q-dimension
    SigmaB <- Sigma3    # r-dimension
    transform_dims <- c(4, 3)  # first transform dimension r, then dimension q
    dim_i <- q
    dim_j <- r
  } else if (l == 2) {
    SigmaA <- Sigma1    # p-dimension
    SigmaB <- Sigma3    # r-dimension
    transform_dims <- c(4, 2) # first transform dimension r, then dimension p
    dim_i <- p
    dim_j <- r
  } else {
    SigmaA <- Sigma1    # p-dimension
    SigmaB <- Sigma2    # q-dimension
    transform_dims <- c(3, 2) # first transform dimension q, then dimension p
    dim_i <- p
    dim_j <- q
  }
  
  #---------------------------------------------------------------------------
  # Compute inverse square-roots of the two fixed covariance factors
  #---------------------------------------------------------------------------
  eigA <- eigen(SigmaA, symmetric = TRUE)
  eigB <- eigen(SigmaB, symmetric = TRUE)
  SigmaA_sqrt_inv <- eigA$vectors %*% diag(1 / sqrt(eigA$values)) %*% t(eigA$vectors)
  SigmaB_sqrt_inv <- eigB$vectors %*% diag(1 / sqrt(eigB$values)) %*% t(eigB$vectors)
  
  #---------------------------------------------------------------------------
  # Center the data: X_centered = X - Mu
  #  (Mu has shape [p, q, r], so replicate along n)
  #---------------------------------------------------------------------------
  X_centered <- sweep(X, MARGIN = c(2, 3, 4), STATS = Mu, FUN = "-")
  
  
  # We'll keep track of the current order of axes for X_centered.
  # Start with "original_axes" = c(1,2,3,4), i.e. [n, p, q, r].
  current_axes <- 1:4
  
  #---------------------------------------------------------------------------
  # Helper: Permute so that 'dim_to_transform' becomes the last dimension (4),
  # only if it's not already last. We then multiply on the right by M.
  # We do NOT permute back immediately; we keep the shape for the next step.
  #---------------------------------------------------------------------------
  transform_once <- function(X_in, axes_in, dim_to_transform, M) {
    
    # If the dimension we want to transform is already axis #4, skip permutation
    if (tail(axes_in, 1) != dim_to_transform) {
      # Create the permutation that moves 'dim_to_transform' from its current 
      # position to the 4th (last) position, leaving other axes in the same order.
      new_axes <- c(setdiff(axes_in, dim_to_transform), dim_to_transform)
      # Apply permutation
      X_perm <- aperm(X_in, match(new_axes, axes_in))  # reorder
      axes_out <- new_axes
    } else {
      # No permutation needed
      X_perm <- X_in
      axes_out <- axes_in
    }
    
    # Multiply on the right by M (now that dim_to_transform is last)
    dtmp <- dim(X_perm)
    mat  <- matrix(X_perm, nrow = prod(dtmp[-4]), ncol = dtmp[4])  # => [*, dim_of_transform]
    mat  <- mat %*% M                                              # multiply on the right
    X_out <- array(mat, dim = dtmp)
    
    # Return updated array + updated axes
    list(
      X_new   = X_out,
      axes_new = axes_out
    )
  }
  
  #---------------------------------------------------------------------------
  # First transformation: multiply on the right by SigmaB_sqrt_inv, 
  # after ensuring transform_dims[1] is in last position if needed.
  #---------------------------------------------------------------------------
  res1 <- transform_once(X_centered, current_axes, transform_dims[1], SigmaB_sqrt_inv)
  X_curr <- res1$X_new
  current_axes <- res1$axes_new
  
  #---------------------------------------------------------------------------
  # Second transformation: multiply on the right by SigmaA_sqrt_inv,
  # from the newly updated shape/orientation.
  #---------------------------------------------------------------------------
  res2 <- transform_once(X_curr, current_axes, transform_dims[2], SigmaA_sqrt_inv)
  X_curr <- res2$X_new
  current_axes <- res2$axes_new
  
  #---------------------------------------------------------------------------
  # Now revert to the original orientation [1,2,3,4] => [n, p, q, r],
  # if it's not already in that orientation.
  #---------------------------------------------------------------------------
  if (!identical(current_axes, 1:4)) {
    X_curr <- aperm(X_curr, match(1:4, current_axes))
    current_axes <- 1:4
  }
  
  #---------------------------------------------------------------------------
  # Finally, reshape along mode l (dimension l+1 in [n, p, q, r]) to compute Sigma_l.
  #   Sigma_l_new = (1 / (n * dim_i * dim_j)) * (X_mat %*% t(X_mat)).
  #---------------------------------------------------------------------------
  final_perm_l <- c(l + 1, setdiff(1:4, l + 1))  # bring axis (l+1) to the front
  X_tmp_l      <- aperm(X_curr, final_perm_l)    # e.g., if l=1 => [p, n, q, r]
  dl           <- dim(X_tmp_l)
  
  X_mat       <- matrix(X_tmp_l, nrow = dl[1], ncol = prod(dl[-1]))
  Sigma_l_new <- (X_mat %*% t(X_mat)) / (n * dim_i * dim_j)
  
  return(Sigma_l_new)
}



#-------------------------------------------------------------------------------
#' Flip-flop MLE for the 3D tensor normal distribution
#'
#' @description
#' Performs maximum likelihood estimation of the separable covariance factors 
#' \(\Sigma_1, \Sigma_2, \Sigma_3\) for a 3D tensor normal model by 
#' iteratively updating each factor in turn (the "flip-flop" procedure). 
#' Optionally applies a shrinkage term \(\lambda\) and a normalization step 
#' that forces \code{Sigma1} and \code{Sigma2} to have their \code{(1,1)} 
#' element equal to 1, with \code{Sigma3} scaled accordingly.
#'
#' @param X A 4D array of shape \code{[n, p, q, r]}, where \code{n} is the 
#'          number of observations, and \code{p,q,r} are the mode dimensions.
#' @param Mu (Optional) A 3D mean array of shape \code{[p, q, r]}. If 
#'           \code{NULL}, the empirical mean is used (i.e., the average over 
#'           all \code{n} observations).
#' @param Sigma1,Sigma2,Sigma3 (Optional) Initial guesses for the covariance 
#'                              factors. If any is \code{NULL}, it defaults 
#'                              to an identity matrix of the appropriate size.
#' @param max_iter Integer. Maximum number of flip-flop iterations.
#' @param tol Numeric. Convergence tolerance on the sum of squared differences 
#'            in \(\Sigma_1, \Sigma_2, \Sigma_3\) across iterations.
#' @param lambda Numeric in \([0,1]\). Shrinkage intensity (0 => no shrinkage, 
#'               1 => full identity). Applied after each factor update.
#' @param silent Logical. If \code{FALSE}, prints iteration and convergence info.
#' @param return_inverses Logical. If \code{TRUE}, also computes and returns 
#'                        the inverse matrices \(\Sigma_i^{-1}\) upon completion.
#'
#' @return A list with elements:
#' \describe{
#'   \item{\code{Mu}}{The estimated mean array, shape \code{[p, q, r]}.}
#'   \item{\code{Sigma1}}{A \code{p x p} covariance factor.}
#'   \item{\code{Sigma2}}{A \code{q x q} covariance factor.}
#'   \item{\code{Sigma3}}{A \code{r x r} covariance factor.}
#'   \item{\code{Sigma1_inv}}{Inverse of \code{Sigma1}, if \code{return_inverses=TRUE}.}
#'   \item{\code{Sigma2_inv}}{Inverse of \code{Sigma2}, if \code{return_inverses=TRUE}.}
#'   \item{\code{Sigma3_inv}}{Inverse of \code{Sigma3}, if \code{return_inverses=TRUE}.}
#' }
#'
#' @details
#' The routine updates each \(\Sigma_l\) by calling \code{\link{update_Sigma_l}}
#' with the current estimates of the other two factors, in the order 
#' \(\Sigma_1 \rightarrow \Sigma_2 \rightarrow \Sigma_3\). After each update, 
#' an optional shrinkage is applied:
#' \[
#'   \Sigma_l \leftarrow \lambda I + (1 - \lambda) \,\Sigma_l.
#' \]
#' Then a normalization enforces \(\Sigma_1[1,1] = \Sigma_2[1,1] = 1\), 
#' rescaling \(\Sigma_3\) accordingly so that the product of these diagonal 
#' elements is preserved. Convergence is declared when 
#' \(\sum \|\Sigma_l - \Sigma_l^\text{old}\|^2 \le \text{tol}\) or when 
#' \code{max_iter} iterations are reached.
#'
#' @export
tensor_MLE <- function(X,
                       Mu = NULL,
                       Sigma1 = NULL,
                       Sigma2 = NULL,
                       Sigma3 = NULL,
                       max_iter = 100,
                       tol = 1e-3,
                       lambda = 0,
                       silent = TRUE,
                       return_inverses = FALSE) {
  
  #---------------------------------------------------------------------------
  # Extract dimensions: [n, p, q, r]
  #---------------------------------------------------------------------------
  n <- dim(X)[1]
  p <- dim(X)[2]
  q <- dim(X)[3]
  r <- dim(X)[4]
  
  #---------------------------------------------------------------------------
  # Initialize the mean tensor (Mu) if not provided
  #---------------------------------------------------------------------------
  if (is.null(Mu)) {
    # Compute empirical mean across observations
    Mu <- apply(X, c(2, 3, 4), mean)
  }
  
  #---------------------------------------------------------------------------
  # Initialize covariance factors (Sigma1, Sigma2, Sigma3) if not provided
  #---------------------------------------------------------------------------
  if (is.null(Sigma1)) Sigma1 <- diag(1, p)
  if (is.null(Sigma2)) Sigma2 <- diag(1, q)
  if (is.null(Sigma3)) Sigma3 <- diag(1, r)
  
  
  #---------------------------------------------------------------------------
  # Flip-flop iterations
  #---------------------------------------------------------------------------
  for (iter in seq_len(max_iter)) {
    
    # Keep old versions to measure convergence
    old_Sigma1 <- Sigma1
    old_Sigma2 <- Sigma2
    old_Sigma3 <- Sigma3
    
    # Update each Sigma in turn
    Sigma1 <- update_Sigma_l(l = 1, X, Mu, Sigma1, Sigma2, Sigma3)
    if (lambda > 0) {
      Sigma1 <- lambda * diag(p) + (1 - lambda) * Sigma1
    }
    
    Sigma2 <- update_Sigma_l(l = 2, X, Mu, Sigma1, Sigma2, Sigma3)
    if (lambda > 0) {
      Sigma2 <- lambda * diag(q) + (1 - lambda) * Sigma2
    }
    
    Sigma3 <- update_Sigma_l(l = 3, X, Mu, Sigma1, Sigma2, Sigma3)
    if (lambda > 0) {
      Sigma3 <- lambda * diag(r) + (1 - lambda) * Sigma3
    }
    
    #---------------------------------------------------------------------------
    # Mode-wise normalization: 
    # enforce Sigma1[1,1] = Sigma2[1,1] = 1, and rescale Sigma3.
    #---------------------------------------------------------------------------
    s1 <- Sigma1[1, 1]
    s2 <- Sigma2[1, 1]
    
    Sigma1 <- Sigma1 / s1
    Sigma2 <- Sigma2 / s2
    Sigma3 <- Sigma3 * (s1 * s2)
    
    #---------------------------------------------------------------------------
    # Check convergence: sum of squared differences in the 3 factors
    #---------------------------------------------------------------------------
    diff_val <- sum((Sigma1 - old_Sigma1)^2) +
      sum((Sigma2 - old_Sigma2)^2) +
      sum((Sigma3 - old_Sigma3)^2)
    
    if (!silent) {
      cat(sprintf("Iteration %d: diff = %.6f\n", iter, diff_val))
    }
    
    if (diff_val < tol) {
      # Converged
      if (!silent) {
        cat("Converged after", iter, "iterations.\n")
      }
      break
    }
  }
  
  
  
  #---------------------------------------------------------------------------
  # Optionally compute and return inverses
  #---------------------------------------------------------------------------
  if (return_inverses) {
    Sigma1_inv <- chol2inv(chol(Sigma1))
    Sigma2_inv <- chol2inv(chol(Sigma2))
    Sigma3_inv <- chol2inv(chol(Sigma3))
    
    return(list(
      Mu        = Mu,
      Sigma1    = Sigma1, 
      Sigma2    = Sigma2, 
      Sigma3    = Sigma3,
      Sigma1_inv = Sigma1_inv,
      Sigma2_inv = Sigma2_inv,
      Sigma3_inv = Sigma3_inv
    ))
  } else {
    # Without inverses
    return(list(
      Mu        = Mu,
      Sigma1    = Sigma1,
      Sigma2    = Sigma2,
      Sigma3    = Sigma3
    ))
  }
}


#-------------------------------------------------------------------------------
#' Perform the C-step for TMCD
#'
#' @description 
#' Given an initial subset of observations (or initial parameter estimates), 
#' this function refines the subset by selecting the \code{h} observations 
#' with the smallest TMD^2 in each iteration, then re-estimates the parameters 
#' via MLE on that refined subset. Iterates until convergence or until 
#' \code{max_iter} is reached.
#'
#' @param X A 4D array of shape \code{[n, p, q, r]} where the first dimension 
#'          (\code{n}) indexes observations, and \code{p, q, r} are the 
#'          mode dimensions.
#' @param subset (Optional) Vector of observation indices defining the starting 
#'               subset. If \code{NULL}, a random subset of size \code{h_init} 
#'               is used.
#' @param Mu (Optional) A 3D mean array \code{[p, q, r]} as the current estimate 
#'           of the mean. If \code{NULL}, it is initialized from the chosen subset.
#' @param Sigma1,Sigma2,Sigma3 (Optional) Current estimates of the covariance 
#'                              factors (\code{p x p}, \code{q x q}, 
#'                              \code{r x r} respectively). If any is \code{NULL}, 
#'                              it is initialized via MLE on the chosen subset.
#' @param Sigma1_inv,Sigma2_inv,Sigma3_inv (Optional) Current inverses of 
#'        \code{Sigma1}, \code{Sigma2}, \code{Sigma3}. If not provided, they are 
#'        computed internally whenever TMD^2 calculation is required.
#' @param h_init If \code{subset} is not provided, a subset of size \code{h_init} 
#'               is drawn at random. If \code{NULL}, a minimal default is used.
#' @param alpha The fraction of observations to keep each iteration, i.e., 
#'              \code{h = floor(alpha * n)}.
#' @param lambda Shrinkage parameter passed to \code{\link{tensor_MLE}} 
#'               (0 => no shrinkage).
#' @param max_iter Maximum number of C-step iterations to refine the subset.
#' @param max_iter_MLE Maximum number of iterations in each call to 
#'                     \code{\link{tensor_MLE}}.
#' @param tol_C Convergence threshold for the log-determinant difference across 
#'              successive iterations. Defaults to \code{1e-4}.
#' @param silent Logical. If \code{FALSE}, prints iteration-level info.
#' @param return_inverses Logical. If \code{TRUE}, the final output also includes 
#'                        \code{Sigma1_inv}, \code{Sigma2_inv}, \code{Sigma3_inv}.
#'                        Otherwise, they are omitted from the return list.
#'
#' @return A list containing:
#' \describe{
#'   \item{\code{Mu}}{Refined mean tensor (\code{[p, q, r]}) after the final iteration.}
#'   \item{\code{Sigma1}}{\code{p x p} covariance estimate for mode-1.}
#'   \item{\code{Sigma2}}{\code{q x q} covariance estimate for mode-2.}
#'   \item{\code{Sigma3}}{\code{r x r} covariance estimate for mode-3.}
#'   \item{\code{Sigma1_inv, Sigma2_inv, Sigma3_inv}}{The final inverses, if 
#'         \code{return_inverses=TRUE}.}
#'   \item{\code{det}}{Final log-determinant sum of \(\Sigma_1, \Sigma_2, \Sigma_3\).}
#'   \item{\code{sub}}{The final subset of observations of size \code{h}.}
#' }
#'
#' @details
#' \enumerate{
#'   \item If no parameters are provided (i.e., if any of \code{Mu, Sigma1, ...} 
#'         are \code{NULL}), a small subset is taken (of size \code{h_init} or 
#'         \code{subset}) to initialize them via \code{\link{tensor_MLE}}.
#'   \item Each iteration:
#'         \itemize{
#'           \item Compute TMD^2 for all observations (via \code{\link{tensorMD_vectorized}}).
#'           \item Select the \code{h} observations with smallest TMD^2.
#'           \item Re-run \code{\link{tensor_MLE}} on that subset 
#'                 to obtain updated estimates.
#'         }
#'   \item Convergence is declared if the log-determinant sum difference falls 
#'         below \code{tol_C} or if \code{max_iter} iterations have occurred.
#' }
#'
#' This C-step is a crucial part of the TMCD algorithm to refine subsets and 
#' re-estimate robustly under a separable tensor covariance model.
#'
#' @export
tensor_C_step <- function(X,
                          subset         = NULL,
                          Mu             = NULL,
                          Sigma1         = NULL,
                          Sigma2         = NULL,
                          Sigma3         = NULL,
                          Sigma1_inv     = NULL,
                          Sigma2_inv     = NULL,
                          Sigma3_inv     = NULL,
                          h_init         = NULL,
                          alpha          = 0.75,
                          lambda         = 0,
                          max_iter       = 100,
                          max_iter_MLE   = 100,
                          tol_C          = 1e-4,
                          silent         = TRUE,
                          return_inverses = FALSE) {
  
  #---------------------------------------------------------------------------
  # Extract [n, p, q, r] and define h = floor(alpha * n)
  #---------------------------------------------------------------------------
  n <- dim(X)[1]
  p <- dim(X)[2]
  q <- dim(X)[3]
  r <- dim(X)[4]
  h <- floor(alpha * n)
  
  #---------------------------------------------------------------------------
  # If no initial parameters are provided, obtain them from a small subset
  #---------------------------------------------------------------------------
  need_init <- (is.null(Mu) || is.null(Sigma1) || is.null(Sigma2) || is.null(Sigma3))
  
  if (need_init) {
    
    # If user did not provide subset, pick random subset of size h_init
    if (is.null(subset)) {
      if (!is.numeric(h_init)) {
        # minimal fallback: ensure p/(q*r), q/(p*r), r/(p*q) are feasible
        h_init <- ceiling(p/(q*r) + q/(p*r) + r/(p*q)) + 2
        
        # The theory is currently insufficiently developed; some believe that formula is:
        #h_init <- ceiling(max(p/(q*r), q/(p*r), r/(p*q))) + 1
      }
      if (h_init > n) {
        stop("Requested h_init exceeds total number of observations.")
      }
      subset_init <- sample(seq_len(n), size = h_init)
    } else {
      subset_init <- subset
    }
    
    # Empirical mean on that initial subset
    Mu_init <- apply(X[subset_init, , , ], c(2, 3, 4), mean)
    
    # MLE on subset to get all needed parameters
    # We ask for inverses because we need TMD^2 soon
    init_fit <- tensor_MLE(X[subset_init, , , ],
                           Mu            = Mu_init,
                           Sigma1        = Sigma1,
                           Sigma2        = Sigma2,
                           Sigma3        = Sigma3,
                           lambda        = lambda,
                           max_iter      = max_iter_MLE,
                           silent        = silent,
                           return_inverses = TRUE)
    
    # Extract the newly initialized parameters
    Mu     <- init_fit$Mu
    Sigma1 <- init_fit$Sigma1
    Sigma2 <- init_fit$Sigma2
    Sigma3 <- init_fit$Sigma3
    Sigma1_inv <- init_fit$Sigma1_inv
    Sigma2_inv <- init_fit$Sigma2_inv
    Sigma3_inv <- init_fit$Sigma3_inv
    
  } else {
    #---------------------------------------------------------------------------
    # Use user-supplied Mu, Sigma1, Sigma2, Sigma3 if they are not NULL
    # Check if we have the inverses; if not, we will compute them
    #---------------------------------------------------------------------------
    if (is.null(Sigma1_inv) || is.null(Sigma2_inv) || is.null(Sigma3_inv)) {
      
      Sigma1_inv <- if (is.null(Sigma1_inv)) chol2inv(chol(Sigma1)) else Sigma1_inv
      Sigma2_inv <- if (is.null(Sigma2_inv)) chol2inv(chol(Sigma2)) else Sigma2_inv
      Sigma3_inv <- if (is.null(Sigma3_inv)) chol2inv(chol(Sigma3)) else Sigma3_inv
    }
  }
  
  # If user did provide subset, store it now (used later for next iteration)
  if (!is.null(subset)) {
    subset_init <- subset
  }
  
  #---------------------------------------------------------------------------
  # Allocate vector to store log-det sums to check convergence
  #---------------------------------------------------------------------------
  det_history <- numeric(max_iter)  # track log-determinant sums
  
  #---------------------------------------------------------------------------
  # Main C-step loop: refine subset, re-estimate, check convergence
  #---------------------------------------------------------------------------
  current_subset <- if (!is.null(subset_init)) subset_init else integer(0)
  
  for (iterC in seq_len(max_iter)) {
    
    # Compute TMD^2 for all observations with current parameters
    
    MD2_all <- tensorMD_vectorized(X, Mu, Sigma1_inv, Sigma2_inv, Sigma3_inv, 
                                   inverted = TRUE)
    
    # Pick h observations with smallest TMD^2
    new_subset <- order(MD2_all)[1:h]
    
    # Compute new mean on that subset
    Mu_new <- apply(X[new_subset, , , ], c(2, 3, 4), mean)
    
    # Run MLE on that subset
    fit_new <- tensor_MLE(X[new_subset, , , ],
                          Mu            = Mu_new,
                          Sigma1        = Sigma1,
                          Sigma2        = Sigma2,
                          Sigma3        = Sigma3,
                          lambda        = lambda,
                          max_iter      = max_iter_MLE,
                          silent        = silent)
    
    # Extract updated parameters
    Mu         <- fit_new$Mu
    Sigma1     <- fit_new$Sigma1
    Sigma2     <- fit_new$Sigma2
    Sigma3     <- fit_new$Sigma3
    
    
    # Compute log-determinant sum => each Sigma factor repeated along other modes
    ld1 <- determinant(Sigma1, logarithm = TRUE)$modulus * (q * r)
    ld2 <- determinant(Sigma2, logarithm = TRUE)$modulus * (p * r)
    ld3 <- determinant(Sigma3, logarithm = TRUE)$modulus * (p * q)
    det_history[iterC] <- ld1 + ld2 + ld3
    
    # Check convergence: difference in log-det sums
    if (iterC > 1) {
      diff_ld <- abs(det_history[iterC] - det_history[iterC - 1])
      if (diff_ld < tol_C) {
        if (!silent) {
          cat(sprintf("C-step converged at iteration %d (log-det diff=%.6f)\n", 
                      iterC, diff_ld))
        }
        current_subset <- new_subset
        break
      }
    }
    
    # If not converged, keep going
    current_subset <- new_subset
    
    if (iterC != max_iter) {
      Sigma1_inv <- chol2inv(chol(Sigma1))
      Sigma2_inv <- chol2inv(chol(Sigma2))
      Sigma3_inv <- chol2inv(chol(Sigma3))
    }
  }
  
  # If we exit by reaching max_iter, we have current_subset and last iteration
  final_ld <- det_history[iterC]  # final log-det sum
  
  #---------------------------------------------------------------------------
  # Optionally remove the inverses from the output if not requested
  #---------------------------------------------------------------------------
  if (!return_inverses) {
    return(list(
      Mu      = Mu,
      Sigma1  = Sigma1,
      Sigma2  = Sigma2,
      Sigma3  = Sigma3,
      det     = final_ld,
      sub     = current_subset
    ))
  } else {
    
    Sigma1_inv <- chol2inv(chol(Sigma1))
    Sigma2_inv <- chol2inv(chol(Sigma2))
    Sigma3_inv <- chol2inv(chol(Sigma3))
    
    return(list(
      Mu          = Mu,
      Sigma1      = Sigma1,
      Sigma2      = Sigma2,
      Sigma3      = Sigma3,
      Sigma1_inv  = Sigma1_inv,
      Sigma2_inv  = Sigma2_inv,
      Sigma3_inv  = Sigma3_inv,
      det         = final_ld,
      sub         = current_subset
    ))
  }
}

#-------------------------------------------------------------------------------
#' Main TMCD procedure for 3D tensor data
#'
#' @description 
#' Implements the Tensor Minimum Covariance Determinant (TMCD) procedure, here 
#' named \code{tensor_MCD}, for robust estimation of location and scatter under 
#' a separable 3D tensor covariance model. It performs multiple short C-step 
#' runs on small random subsets, refines the best candidates with full C-steps, 
#' rescales, and reweights outliers to finalize the robust estimates.
#'
#' @param X A 4D array of shape \code{[n, p, q, r]} containing the data, where 
#'          \code{n} indexes observations, and \code{p,q,r} are the mode 
#'          dimensions.
#' @param nsamp Integer. Number of small random subsets for initial short runs.
#' @param alpha Fraction of observations to keep; the subset size is 
#'              \(\lfloor \alpha \cdot n \rfloor\).
#' @param lambda Shrinkage parameter passed to \code{\link{tensor_MLE}} 
#'               (0 => no shrinkage).
#' @param max_iter_MLE Integer. Maximum iterations in each \code{tensor_MLE} call.
#' @param max_iter_C_step Integer. Maximum iterations for each full C-step refinement.
#' @param silent Logical. If \code{FALSE}, prints basic progress messages.
#' @param return_inverses Logical. If \code{TRUE}, returns the final inverse 
#'                        matrices \code{Sigma1_inv}, \code{Sigma2_inv}, 
#'                        \code{Sigma3_inv} in addition to the covariance factors.
#'
#' @return A list with components:
#' \describe{
#'   \item{\code{Mu}}{The final robust mean tensor, shape \code{[p,q,r]}.}
#'   \item{\code{Sigma1, Sigma2, Sigma3}}{Final robust covariance factors 
#'         (\code{p x p}, \code{q x q}, \code{r x r}).}
#'   \item{\code{Sigma1_inv, Sigma2_inv, Sigma3_inv}}{If \code{return_inverses=TRUE}, 
#'         the inverses of the final covariance factors.}
#'   \item{\code{sub}}{The best C-step subset indices.}
#'   \item{\code{clean_obs}}{A logical vector indicating which observations were 
#'         reweighted as non-outliers in the final step.}
#' }
#'
#' @details 
#' This procedure follows:
#' \enumerate{
#'   \item Determine a small elemental subset size, \code{d}. 
#'   \item For each of \code{nsamp} random subsets of size \code{d}, run a short 
#'         (2-iteration) C-step via \code{\link{tensor_C_step}}.
#'   \item Keep the top 10 solutions by the final log-determinant criterion.
#'   \item Refine these 10 solutions with up to \code{max_iter_C_step} C-step 
#'         iterations each; pick the best.
#'   \item Apply a consistency factor to \code{Sigma1}, \code{Sigma2}, \code{Sigma3} 
#'         so that the median of TMD^2 matches the theoretical median of the 
#'         \(\chi^2_{p \times q \times r}\) distribution.
#'   \item Reweight observations as non-outliers (clean) if their rescaled TMD^2 
#'         is below the \(\chi^2_{p \times q \times r,\,0.9}\) cutoff.
#'   \item Re-run \code{\link{tensor_MLE}} on the clean subset, and apply a final 
#'         consistency factor. 
#' }
#'
#' This yields the final robust mean tensor \code{Mu} and covariance factors 
#' \code{Sigma1}, \code{Sigma2}, \code{Sigma3}. The user can optionally obtain 
#' their inverses and the final subset of selected observations.
#'
#' @export
tensor_MCD <- function(X, 
                       nsamp           = 500, 
                       alpha           = 0.75, 
                       lambda          = 0,
                       max_iter_MLE    = 100,
                       max_iter_C_step = 100,
                       silent          = TRUE,
                       return_inverses = FALSE) {
  
  #---------------------------------------------------------------------------
  # Extract dimensions and define h = floor(alpha * n)
  #---------------------------------------------------------------------------
  n <- dim(X)[1]
  p <- dim(X)[2]
  q <- dim(X)[3]
  r <- dim(X)[4]
  h <- floor(alpha * n)
  
  #---------------------------------------------------------------------------
  # Elemental subset size: minimal size for initial robust start
  
  #---------------------------------------------------------------------------
  
  
  d <- ceiling(p/(q*r) + q/(p*r) + r/(p*q)) + 2
  
  # The theory is currently insufficiently developed; some believe that formula is:
  #d <- ceiling(max(p/(q*r), q/(p*r), r/(p*q))) + 1
  
  
  
  
  
  #---------------------------------------------------------------------------
  # Initial short runs (2-iteration C-steps) on random subsets of size d
  #---------------------------------------------------------------------------
  init_results <- vector("list", nsamp)
  
  for (k in seq_len(nsamp)) {
    # Random subset of size d
    subset_idx <- sample(seq_len(n), size = d)
    
    # Short C-step: just 2 iterations of refinement
    cstep_res  <- tensor_C_step(X,
                                subset         = subset_idx,
                                alpha          = alpha, 
                                lambda         = lambda,
                                max_iter       = 2, 
                                max_iter_MLE   = 2,
                                tol_C          = 1e-4,
                                silent         = silent)
    init_results[[k]] <- cstep_res
  }
  
  # Evaluate log-dets from these short runs
  det_values <- sapply(init_results, function(res) res$det)
  
  # Keep top solutions by smallest log-det
  keep_count <- min(10, nsamp)
  top_indices <- order(det_values)[seq_len(keep_count)]
  
  #---------------------------------------------------------------------------
  # Full refinement with up to max_iter_C_step on top 10 solutions
  #---------------------------------------------------------------------------
  best_results <- vector("list", keep_count)
  
  for (i in seq_len(keep_count)) {
    idx_top <- top_indices[i]
    init_res <- init_results[[idx_top]]
    
    # Now run a full C-step with more iterations
    cstep_full <- tensor_C_step(X,
                                # Re-use final estimates from the short run
                                subset  = init_res$sub,
                                Mu      = init_res$Mu,
                                Sigma1  = init_res$Sigma1,
                                Sigma2  = init_res$Sigma2,
                                Sigma3  = init_res$Sigma3,
                                alpha   = alpha,
                                lambda  = lambda,
                                max_iter     = max_iter_C_step, 
                                max_iter_MLE = max_iter_MLE,
                                tol_C        = 1e-4,
                                silent       = silent)
    best_results[[i]] <- cstep_full
  }
  
  # Among these refined solutions, pick best by minimal log-det
  final_dets <- sapply(best_results, function(res) res$det)
  best_idx   <- which.min(final_dets)
  final_raw  <- best_results[[best_idx]]
  
  #---------------------------------------------------------------------------
  # Scale for consistency: The scale factor ensures that median(MDs) equals
  # qchisq(0.5, dof), where dof = p*q*r.  Then reweight outliers by cutoff
  # at qchisq(0.9, dof).
  #---------------------------------------------------------------------------
  # We first need the inverses for TMD^2.
  # We do a quick step computing them from final_raw if not present.
  #---------------------------------------------------------------------------
  Sigma1_inv <- chol2inv(chol(final_raw$Sigma1))
  Sigma2_inv <- chol2inv(chol(final_raw$Sigma2))
  Sigma3_inv <- chol2inv(chol(final_raw$Sigma3))
  
  # Compute TMD^2 on all obs
  MDs <- tensorMD_vectorized(X, 
                             final_raw$Mu, 
                             Sigma1_inv, Sigma2_inv, Sigma3_inv,
                             inverted = TRUE)
  
  dof <- p * q * r
  # Compute scale factor so that median(MDs) = qchisq(0.5, dof)
  MCD_scale <- median(MDs) / qchisq(0.5, dof)
  
  # Apply scale^(1/3) to each Sigma
  Sigma1_scaled <- final_raw$Sigma1 * MCD_scale^(1/3)
  Sigma2_scaled <- final_raw$Sigma2 * MCD_scale^(1/3)
  Sigma3_scaled <- final_raw$Sigma3 * MCD_scale^(1/3)
  
  
  # Identify outliers by 90% cutoff
  MDs_scaled <- MDs / MCD_scale
  cutoff_90  <- qchisq(0.9, dof)
  clean_obs  <- (MDs_scaled < cutoff_90)
  
  #---------------------------------------------------------------------------
  # Re-run MLE on the "clean" subset, then apply final consistency factor again
  #---------------------------------------------------------------------------
  # Mean for the clean subset
  Mu_clean <- apply(X[clean_obs, , , ], c(2,3,4), mean)
  
  # MLE with the scaled starting points
  fit_clean <- tensor_MLE(X[clean_obs, , , ],
                          Mu      = Mu_clean,
                          Sigma1  = Sigma1_scaled,
                          Sigma2  = Sigma2_scaled,
                          Sigma3  = Sigma3_scaled,
                          max_iter = max_iter_MLE,
                          tol      = 1e-3,
                          lambda   = lambda,
                          silent   = silent,
                          return_inverses = TRUE)
  
  # Extract final parameters
  Mu_final  <- fit_clean$Mu
  Sigma1_final  <- fit_clean$Sigma1
  Sigma2_final  <- fit_clean$Sigma2
  Sigma3_final  <- fit_clean$Sigma3
  Sigma1_inv_final <- fit_clean$Sigma1_inv
  Sigma2_inv_final <- fit_clean$Sigma2_inv
  Sigma3_inv_final <- fit_clean$Sigma3_inv
  
  # Final re-scaling
  final_MDs <- tensorMD_vectorized(X, Mu_final, Sigma1_inv_final, Sigma2_inv_final, Sigma3_inv_final,
                                   inverted = TRUE)
  
  final_scale <- median(final_MDs) / qchisq(0.5, dof)
  
  Sigma1_final  <- Sigma1_final  * final_scale^(1/3)
  Sigma2_final  <- Sigma2_final  * final_scale^(1/3)
  Sigma3_final  <- Sigma3_final  * final_scale^(1/3)
  
  
  
  #---------------------------------------------------------------------------
  # Return final robust estimates
  #---------------------------------------------------------------------------
  if (!return_inverses) {
    return(list(
      Mu       = Mu_final,
      Sigma1   = Sigma1_final,
      Sigma2   = Sigma2_final,
      Sigma3   = Sigma3_final,
      sub      = final_raw$sub,
      clean_obs= clean_obs
    ))
  } else {
    
    # Scale inverses down
    Sigma1_inv_final <- Sigma1_inv_final / final_scale^(1/3)
    Sigma2_inv_final <- Sigma2_inv_final / final_scale^(1/3)
    Sigma3_inv_final <- Sigma3_inv_final / final_scale^(1/3)
    
    return(list(
      Mu          = Mu_final,
      Sigma1      = Sigma1_final,
      Sigma2      = Sigma2_final,
      Sigma3      = Sigma3_final,
      Sigma1_inv  = Sigma1_inv_final,
      Sigma2_inv  = Sigma2_inv_final,
      Sigma3_inv  = Sigma3_inv_final,
      sub         = final_raw$sub,
      clean_obs   = clean_obs
    ))
  }
}

#-------------------------------------------------------------------------------
#' Kullback-Leibler divergence between two 3D tensor normal distributions
#'
#' @description
#' Computes the Kullback-Leibler (KL) divergence \(\mathrm{D_{KL}(\cdot \parallel \cdot)}\) 
#' between two separable 3D tensor normal distributions:
#' \[
#'   \mathrm{N}_{\mathrm{tensor}}\bigl(\Mu_{\text{true}}, 
#'        \Sigma_{1,\text{true}}\otimes \Sigma_{2,\text{true}}\otimes \Sigma_{3,\text{true}}\bigr)
#'   \quad\text{and}\quad
#'   \mathrm{N}_{\mathrm{tensor}}\bigl(\Mu_{\text{est}}, 
#'        \Sigma_{1,\text{est}}\otimes \Sigma_{2,\text{est}}\otimes \Sigma_{3,\text{est}}\bigr).
#' \]
#' Only the covariance factors matter for KL divergence (the means cancel out).
#'
#' @param Sigma1_true A \code{p x p} positive-definite matrix (true covariance 
#'                    factor for mode-1).
#' @param Sigma2_true A \code{q x q} positive-definite matrix (true covariance 
#'                    factor for mode-2).
#' @param Sigma3_true A \code{r x r} positive-definite matrix (true covariance 
#'                    factor for mode-3).
#' @param Sigma1_est A \code{p x p} positive-definite matrix (estimated covariance 
#'                   factor for mode-1).
#' @param Sigma2_est A \code{q x q} positive-definite matrix (estimated covariance 
#'                   factor for mode-2).
#' @param Sigma3_est A \code{r x r} positive-definite matrix (estimated covariance 
#'                   factor for mode-3).
#'
#' @return A single numeric value giving the KL divergence 
#' \(\mathrm{D_{KL}}(\text{true} \parallel \text{est})\).
#'
#' @details
#' For a 3D tensor normal distribution with separable covariance factors 
#' \(\Sigma_1 \otimes \Sigma_2 \otimes \Sigma_3\), the effective dimension is 
#' \(p \times q \times r\). The KL divergence splits into mode-wise terms, 
#' each scaled by the product of the remaining mode sizes:
#' \[
#'   \mathrm{D_{KL}}(\text{true}\|\text{est}) 
#'   = \tfrac12 \bigl( c_1 \bigl[\mathrm{tr}(\Sigma_{1,\text{est}}^{-1}\,\Sigma_{1,\text{true}}) 
#'     - \ln\!\det(\Sigma_{1,\text{est}}^{-1}\,\Sigma_{1,\text{true}}) - p \bigr]
#'     + c_2 \bigl[\cdots\bigr]
#'     + c_3 \bigl[\cdots\bigr]\bigr),
#' \]
#' where \(c_1 = q \times r\), \(c_2 = p \times r\), and \(c_3 = p \times q\).
#'
#' @export
tensor_KL_divergence <- function(Sigma1_true, Sigma2_true, Sigma3_true,
                                 Sigma1_est,  Sigma2_est,  Sigma3_est) {
  
  #---------------------------------------------------------------------------
  # Extract dimensions from the true covariance factors
  #---------------------------------------------------------------------------
  p <- nrow(Sigma1_true)  # mode-1 dimension
  q <- nrow(Sigma2_true)  # mode-2 dimension
  r <- nrow(Sigma3_true)  # mode-3 dimension
  
  #---------------------------------------------------------------------------
  # Invert the "est" covariance factors
  #---------------------------------------------------------------------------
  Sigma1_inv_est <- chol2inv(chol(Sigma1_est))
  Sigma2_inv_est <- chol2inv(chol(Sigma2_est))
  Sigma3_inv_est <- chol2inv(chol(Sigma3_est))
  
  #---------------------------------------------------------------------------
  # c1, c2, c3 scale each mode's contribution
  #---------------------------------------------------------------------------
  c1 <- q * r
  c2 <- p * r
  c3 <- p * q
  
  #---------------------------------------------------------------------------
  # Compute traces: tr(Sigma_inv_est * Sigma_true)
  #---------------------------------------------------------------------------
  tr1 <- sum(diag(Sigma1_inv_est %*% Sigma1_true))
  tr2 <- sum(diag(Sigma2_inv_est %*% Sigma2_true))
  tr3 <- sum(diag(Sigma3_inv_est %*% Sigma3_true))
  
  #---------------------------------------------------------------------------
  # Compute log-dets: ln(det(Sigma_inv_est * Sigma_true))
  #---------------------------------------------------------------------------
  ld1 <- log(det(Sigma1_inv_est %*% Sigma1_true))
  ld2 <- log(det(Sigma2_inv_est %*% Sigma2_true))
  ld3 <- log(det(Sigma3_inv_est %*% Sigma3_true))
  
  #---------------------------------------------------------------------------
  # Mode-wise KL contributions
  #---------------------------------------------------------------------------
  D1 <- c1 * (tr1 - ld1 - p)
  D2 <- c2 * (tr2 - ld2 - q)
  D3 <- c3 * (tr3 - ld3 - r)
  
  #---------------------------------------------------------------------------
  # Final KL divergence (scalar)
  #---------------------------------------------------------------------------
  KL_value <- 0.5 * (D1 + D2 + D3)
  return(KL_value)
  
}