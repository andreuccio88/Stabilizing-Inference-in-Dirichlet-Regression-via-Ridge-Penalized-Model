##############################################################################
##############################################################################
##  R Code for paper:                                                       ##
## "Stabilizing Inference in Dirichlet Regression via Ridge-Penalized Model"##
##                                                                          ##
## by Andrea Nigri (andrea.nigri@unifg.it)                                  ##
##############################################################################
##############################################################################



############################################################
## 0. Library
############################################################
# install.packages(c("MCMCpack", "matrixStats", "MASS", "ggplot2"))
library(MCMCpack)
library(matrixStats)
library(MASS)
library(ggplot2)

set.seed(123)  # per riproducibilità globale

############################################################
## 1. Utility
############################################################

softmax <- function(eta) {
  # eta: matrice T x D
  exp_eta <- exp(eta - rowLogSumExps(eta))
  exp_eta / rowSums(exp_eta)
}

safe_log <- function(x, eps = 1e-8) log(pmax(x, eps))

############################################################
## 2. Dirichlet ridge fit 
############################################################

dirichlet_ridge_fit <- function(Y, X, lambda = 1e-3,
                                max_iter = 500, tol = 1e-6,
                                verbose = FALSE) {
  Y <- as.matrix(Y)
  X <- as.matrix(X)
  Tn <- nrow(Y)
  p  <- ncol(X)
  D  <- ncol(Y)
  
  B <- matrix(0, p, D)   
  phi <- 300
  lr0 <- 0.001           
  loglik_trace <- numeric(max_iter)
  
  for (iter in 1:max_iter) {
    eta <- X %*% B
    mu  <- softmax(eta)
    alpha <- mu * phi
    lr <- lr0 / sqrt(iter)
    
    ## Update B
    for (d in 1:D) {
      grad <- rep(0, p)
      for (t in 1:Tn) {
        
        val_alpha <- pmax(alpha[t, d], 1e-6)
        A_td <- safe_log(Y[t, d]) - digamma(val_alpha) + digamma(phi)
        grad <- grad + phi * A_td * mu[t, d] * (1 - mu[t, d]) * X[t, ]
      }
      B[, d] <- B[, d] + lr * (grad - 2 * lambda * B[, d])
    }
    
    
    score <- 0
    hess  <- 0
    for (t in 1:Tn) {
      for (d in 1:D) {
        val <- pmax(phi * mu[t, d], 1e-6)
        score <- score + digamma(phi) -
          mu[t, d] * digamma(val) +
          mu[t, d] * safe_log(Y[t, d])
        hess  <- hess  + trigamma(phi) -
          (mu[t, d]^2) * trigamma(val)
      }
    }
    
    if (is.finite(score) && is.finite(hess) && hess != 0) {
      delta <- score / hess
      delta <- max(min(delta, 10), -10)  
      phi <- phi - delta
      phi <- max(min(phi, 1e3), 1e-3)
    }
    
    ll <- sum(lgamma(phi) - rowSums(lgamma(alpha)) +
                rowSums((alpha - 1) * safe_log(Y)))
    loglik_trace[iter] <- ll
    
    if (!is.finite(ll)) return(NULL)
    if (verbose && iter %% 50 == 0) {
      cat("Iter", iter, "phi =", round(phi, 3),
          "logLik =", round(ll, 3), "\n")
    }
    if (iter > 1 && abs(loglik_trace[iter] - loglik_trace[iter - 1]) < tol) {
      break
    }
  }
  
  if (!is.finite(loglik_trace[iter])) return(NULL)
  
  list(B = B, phi = phi, loglik = loglik_trace[1:iter])
}

############################################################
## 3. Cross-validation lambda 
############################################################

cv_dirichlet_ridge <- function(Y, X,
                               lambda_seq,
                               K = 5, seed = 123) {
  set.seed(seed)
  Y <- as.matrix(Y)
  X <- as.matrix(X)
  Tn <- nrow(Y)
  folds <- sample(rep(1:K, length.out = Tn))
  scores <- rep(NA_real_, length(lambda_seq))
  
  for (i in seq_along(lambda_seq)) {
    lambda <- lambda_seq[i]
    fold_loglik <- rep(NA_real_, K)
    
    for (k in 1:K) {
      train_idx <- which(folds != k)
      test_idx  <- which(folds == k)
      
      fit <- tryCatch(
        dirichlet_ridge_fit(Y[train_idx, , drop = FALSE],
                            X[train_idx, , drop = FALSE],
                            lambda = lambda,
                            verbose = FALSE),
        error = function(e) NULL
      )
      
      if (!is.null(fit)) {
        eta   <- X[test_idx, , drop = FALSE] %*% fit$B
        mu    <- softmax(eta)
        alpha <- mu * fit$phi
        loglik <- sum(lgamma(fit$phi) - rowSums(lgamma(alpha)) +
                        rowSums((alpha - 1) * safe_log(Y[test_idx, , drop = FALSE])))
        fold_loglik[k] <- loglik
      }
    }
    scores[i] <- if (all(is.na(fold_loglik))) NA else mean(fold_loglik, na.rm = TRUE)
  }
  
  if (all(is.na(scores))) {
    warning("CV: fit no valid, use first value.")
    best_idx <- 1
  } else {
    best_idx <- which.max(scores)
  }
  
  list(best_lambda = lambda_seq[best_idx],
       logliks     = scores,
       lambda_seq  = lambda_seq)
}

############################################################
## 4. Simulations: S1, S2, S3
############################################################

simulate_data <- function(scenario = c("S1", "S2", "S3"),
                          D = 4,
                          seed = NULL) {
  scenario <- match.arg(scenario)
  if (!is.null(seed)) set.seed(seed)
  
  if (scenario == "S1") {
    # Orthogonal, low noise
    Tn <- 50; p <- 50
    X  <- diag(1, Tn, p)
    phi_true <- 500
    B_true <- matrix(rnorm(p * D, 0, 0.4), p, D)
    
  } else if (scenario == "S2") {
    # Correlated, moderate noise
    Tn <- 100; p <- 20
    rho <- 0.7
    Sigma <- rho ^ abs(outer(1:p, 1:p, "-"))
    Sigma <- Sigma + 1e-6 * diag(p)
    X <- mvrnorm(Tn, mu = rep(0, p), Sigma = Sigma)
    phi_true <- 100
    B_true <- matrix(rnorm(p * D, 0, 0.4), p, D)
    
  } else {  # S3: correlated + sparse coefficients
    Tn <- 100; p <- 50
    rho <- 0.7
    Sigma <- rho ^ abs(outer(1:p, 1:p, "-"))
    Sigma <- Sigma + 1e-6 * diag(p)
    X <- mvrnorm(Tn, mu = rep(0, p), Sigma = Sigma)
    phi_true <- 100
    
   
    B_true <- matrix(0, p, D)
    s <- 10  
    for (d in 1:D) {
      idx <- sample(1:p, s)
      B_true[idx, d] <- rnorm(s, 0, 0.4)
    }
  }
  
  eta_true <- X %*% B_true
  mu_true  <- softmax(eta_true)
  alpha_true <- mu_true * phi_true
  Y <- t(apply(alpha_true, 1, function(a) rdirichlet(1, a)))
  Y[Y < 1e-8] <- 1e-8
  
  list(
    scenario   = scenario,
    X          = X,
    Y          = Y,
    B_true     = B_true,
    phi_true   = phi_true
  )
}

############################################################
## 5. Sim Monte Carlo per uno scenario
##    - S1: lambda_star = 0 (ridge = unpen)
##    - S2, S3: CV on lambda ∈ [10^-3, 1]
############################################################

run_simulation_scenario <- function(scenario,
                                    R_reps = 200,
                                    K = 5,
                                    seed = 123) {
  set.seed(seed)
  results <- data.frame()
  
  for (r in 1:R_reps) {
    sim <- simulate_data(scenario = scenario, seed = seed + r)
    X <- sim$X
    Y <- sim$Y
    B_true <- sim$B_true
    
   
    if (scenario == "S1") {
      lambda_star <- 0
    } else {
      lambda_seq <- 10^seq(-3, 0, length.out = 7)  # da 0.001 a 1
      cv <- cv_dirichlet_ridge(Y, X,
                               lambda_seq = lambda_seq,
                               K = K,
                               seed = seed + 1000 + r)
      lambda_star <- cv$best_lambda
    }
    
   
    fit_ridge <- tryCatch(
      dirichlet_ridge_fit(Y, X, lambda = lambda_star, verbose = FALSE),
      error = function(e) NULL
    )
    
    fit_unpen <- tryCatch(
      dirichlet_ridge_fit(Y, X, lambda = 0, verbose = FALSE),
      error = function(e) NULL
    )
    
   
    nz_mask <- (B_true != 0)
    z_mask  <- (B_true == 0)
    
    if (!is.null(fit_ridge) && !is.null(fit_unpen)) {
      B_ridge <- fit_ridge$B
      B_unpen <- fit_unpen$B
      
      rmse_ridge    <- sqrt(mean((B_ridge - B_true)^2))
      rmse_unpen    <- sqrt(mean((B_unpen - B_true)^2))
      
      loglik_ridge  <- tail(fit_ridge$loglik, 1)
      loglik_unpen  <- tail(fit_unpen$loglik, 1)
      
      rmse_ridge_nz <- if (any(nz_mask)) sqrt(mean((B_ridge[nz_mask] - B_true[nz_mask])^2)) else NA
      rmse_unpen_nz <- if (any(nz_mask)) sqrt(mean((B_unpen[nz_mask] - B_true[nz_mask])^2)) else NA
      
      rmse_ridge_z  <- if (any(z_mask)) sqrt(mean((B_ridge[z_mask] - B_true[z_mask])^2)) else NA
      rmse_unpen_z  <- if (any(z_mask)) sqrt(mean((B_unpen[z_mask] - B_true[z_mask])^2)) else NA
      
      results <- rbind(results, data.frame(
        scenario        = scenario,
        rep             = r,
        rmse_ridge      = rmse_ridge,
        rmse_unpen      = rmse_unpen,
        loglik_ridge    = loglik_ridge,
        loglik_unpen    = loglik_unpen,
        lambda_star     = lambda_star,
        rmse_ridge_nz   = rmse_ridge_nz,
        rmse_unpen_nz   = rmse_unpen_nz,
        rmse_ridge_z    = rmse_ridge_z,
        rmse_unpen_z    = rmse_unpen_z
      ))
    } else {
      results <- rbind(results, data.frame(
        scenario        = scenario,
        rep             = r,
        rmse_ridge      = NA,
        rmse_unpen      = NA,
        loglik_ridge    = NA,
        loglik_unpen    = NA,
        lambda_star     = lambda_star,
        rmse_ridge_nz   = NA,
        rmse_unpen_nz   = NA,
        rmse_ridge_z    = NA,
        rmse_unpen_z    = NA
      ))
    }
    
    if (r %% 20 == 0 || r == R_reps) {
      cat(sprintf("Scenario %s - rep %d/%d (%.1f%%)\n",
                  scenario, r, R_reps, 100 * r / R_reps))
    }
  }
  
  results
}

############################################################
## 6. Run S1, S2, S3 - R = 200
############################################################

R_reps <- 200  

scenarios <- c("S1", "S2", "S3")
res_list  <- list()

for (sc in scenarios) {
  cat("\n==========================\n")
  cat("Running scenario", sc, "\n")
  cat("==========================\n")
  res_list[[sc]] <- run_simulation_scenario(sc,
                                            R_reps = R_reps,
                                            K = 5,
                                            seed = 123)
}

results_all <- do.call(rbind, res_list)

cat("\Number of simulation each scenario:\n")
print(table(results_all$scenario))

############################################################
## 7. Errors Tab (RMSE + log-likelihood)
############################################################

summary_fun <- function(x) c(mean = mean(x, na.rm = TRUE),
                             sd   = sd(x,   na.rm = TRUE))

agg_rmse <- aggregate(cbind(rmse_ridge, rmse_unpen) ~ scenario,
                      data = results_all, FUN = summary_fun)
agg_loglik <- aggregate(cbind(loglik_ridge, loglik_unpen) ~ scenario,
                        data = results_all, FUN = summary_fun)

## Tab
err_table <- data.frame(
  scenario          = agg_rmse$scenario,
  rmse_ridge_mean   = agg_rmse$rmse_ridge[, "mean"],
  rmse_ridge_sd     = agg_rmse$rmse_ridge[, "sd"],
  rmse_unpen_mean   = agg_rmse$rmse_unpen[, "mean"],
  rmse_unpen_sd     = agg_rmse$rmse_unpen[, "sd"]
)

cat("\nRMSE summary by scenario (mean, sd):\n")
print(err_table)

loglik_table <- data.frame(
  scenario            = agg_loglik$scenario,
  loglik_ridge_mean   = agg_loglik$loglik_ridge[, "mean"],
  loglik_ridge_sd     = agg_loglik$loglik_ridge[, "sd"],
  loglik_unpen_mean   = agg_loglik$loglik_unpen[, "mean"],
  loglik_unpen_sd     = agg_loglik$loglik_unpen[, "sd"]
)

cat("\nLog-likelihood summary by scenario (mean, sd):\n")
print(loglik_table)



