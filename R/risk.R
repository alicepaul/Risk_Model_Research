source("utils.R")
source("helper_functions.R")
library(tidyverse) 

# Update documentation

#' Partial derivative of the negative log-likelihood
#' 
#' Calculates the partial derivative for beta_j of the objective function
#' @param X input matrix with dimension n x p, every row is an observation
#' @param y numeric vector for the response variable (binomial)
#' @param gamma scalar to rescale betas for prediction
#' @param beta numeric vector with p coefficients 
#' @param weights numeric vector of length n with weights for each 
#' observation
#' @param j index of beta
#' @return the numeric partial derivative value
par_deriv <- function(X, y, gamma, beta, weights, j) {
  
  # Calculate partial derivative for NLL
  pd_1 <- sum(gamma * weights * (y * X[,j]))
  exp_pred <- exp(clip_exp_vals(gamma * (X %*% beta)))
  pd_2 <- sum(gamma*weights * X[,j] * (exp_pred / (1.0 + exp_pred)))
  nll_pd <- (-1/nrow(X))*(pd_1-pd_2)
  
  return(nll_pd)
}

#' Objective function for NLL+penalty
#' 
#' Calculates the objective function for gamma, beta (NLL+penalty)
#' @param X input matrix with dimension n x p, every row is an observation
#' @param y numeric vector for the response variable (binomial)
#' @param gamma scalar to rescale betas for prediction
#' @param beta numeric vector with p coefficients 
#' @param weights numeric vector of length n with weights for each 
#' observation
#' @param lambda0 penalty coefficient for L0 term (default 0)
#' @return numeric objective function value
obj_fcn <- function(X, y, gamma, beta, weights, lambda0=0) {

  # Calculate partial derivative for NLL
  v <- gamma * (X %*% beta)
  v <- clip_exp_vals(v) # avoids numeric errors
  nll_fcn <- (-1/nrow(X))*sum(weights * (y * v - log(1+exp(v))))
  
  # Penalty term for lambda0*||beta||_0 
  pen_fcn <- lambda0*sum(beta[-1] != 0) 
  return (nll_fcn + pen_fcn)
}

#' Bisection search for coordinate descent
#' 
#' Returns optimal value on beta_j using bisection search
#' @param X input matrix with dimension n x p, every row is an observation
#' @param y numeric vector for the response variable (binomial)
#' @param gamma scalar to rescale betas for prediction
#' @param beta numeric vector with p coefficients 
#' @param weights numeric vector of length n with weights for each 
#' observation
#' @param j index of beta
#' @param lambda0 penalty coefficient for L0 term (default 0)
#' @param a integer lower bound for beta_j (default -10)
#' @param b integer upper bound for beta_j (default 10)
#' @return numeric vector beta with optimal value for beta[j] updated
bisec_search <- function(X, y, gamma, beta, weights, j, lambda0 = 0, 
                         a = -10, b = 10) {
  
  # Initial betas to compare
  beta_a <- beta
  beta_a[j] <- a
  beta_b <- beta
  beta_b[j] <- b
  beta_0 <- beta
  beta_0[j] <- 0

  # If no zero derivative in range, skip while loop
  der_a <- par_deriv(X, y, gamma, beta_a, weights, j)
  der_b <- par_deriv(X, y, gamma, beta_b, weights, j)
  search <- TRUE
  if (sign(der_a) == sign(der_b)) search <- FALSE

  while (((b - a) > 1) & search){
    # Find partial derivative at midpoint
    c <- floor((a+b)/2)
    beta_c <- beta
    beta_c[j] <- c
    der_c <- par_deriv(X, y, gamma, beta_c, weights, j)
    
    # Update interval
    if (der_c == 0)
    {
      # If partial derivative is zero then break loop
      beta_a <- beta_c
      beta_b <- beta_c
      break
    } else if (sign(der_c) == sign(der_a)){ 
      # Move to right
      a <- c
      beta_a <- beta_c
      der_a <- der_c
    } else {
      # Move to left
      b <- c
      beta_b <- beta_c
      der_b <- der_c
    }
  }
  
  # Find best of a, b, and 0 in objective function
  obj_a <- obj_fcn(X, y, gamma, beta_a, weights, lambda0)
  obj_b <- obj_fcn(X, y, gamma, beta_b, weights, lambda0)
  obj_0 <- obj_fcn(X, y, gamma, beta_0, weights, lambda0)
  
  # Return optimal solution
  if ((obj_0 <= obj_a) & (obj_0 <= obj_b)){
    return (beta_0)
  } else if (obj_a <= obj_b) {
    return (beta_a)
  } 
  return (beta_b)
}

#' Update gamma and beta[1]
#' 
#' Finds optimal gamma value and intercept
#' @param X input matrix with dimension n x p, every row is an observation
#' @param y numeric vector for the response variable (binomial)
#' @param beta numeric vector with p coefficients 
#' @param weights numeric vector of length n with weights for each 
#' observation
#' @return gamma and beta (with updated beta[1] value) in a list
update_gamma_intercept <- function(X, y, beta, weights) {
  
  # Calculate current integer scores and run logistic regression
  z <- X %*% beta - beta[1]*X[,1]
  lr_mod <- glm(y ~ z, weights = weights, family="binomial")
  
  # Find gamma and beta[1]
  coef_vec <- unname(coef(lr_mod))
  gamma <- coef_vec[2]
  if (is.na(gamma)){
    gamma <- 1
  }
  beta[1] <- coef_vec[1] / gamma

  return (list(gamma=gamma, beta=beta))
}

#' Coordinate descent to find the optimal risk model
#' 
#' Returns the estimated gamma and beta for the risk score model
#' @param X input matrix with dimension n x p, every row is an observation
#' @param y numeric vector for the response variable (binomial)
#' @param gamma scalar to rescale betas for prediction
#' @param beta numeric vector with p coefficients 
#' @param weights numeric vector of length n with weights for each 
#' observation
#' @param lambda0 penalty coefficient for L0 term (default 0)
#' @param a integer lower bound for betas (default -10)
#' @param b integer upper bound for betas (default 10)
#' @param max_iters maximum number of iterations (default 100)
#' @param tol tolerance for convergence
#' @return optimal gamma (numeric) and beta (numeric vector) as a list
risk_coord_desc <- function(X, y, gamma, beta, weights, lambda0 = 0, 
                            a = -10, b = 10, max_iters = 100, tol= 1e-5) {
  
  # Run for maximum number of iterations
  iters <- 1
  while (iters < max_iters)
  {
    # Keep track of old value to check convergence
    old_beta <- beta
    
    # Iterate through all variables and update intercept/gamma after each
    for (j in (2:ncol(X))){
      beta <- bisec_search(X, y, gamma, beta, weights, j, lambda0, a, b)
      upd <- update_gamma_intercept(X, y, beta, weights)
      gamma <- upd$gamma
      beta <- upd$beta

      # Check for NaN
      if (is.nan(gamma) | sum(is.nan(beta)) > 0){
        stop("Algorithm did not converge - encountered NaN")
      }
    }
  
    # Check if change in beta is within tolerance to converge
    if (max(abs(old_beta - beta)) < tol){
      break
    }
    iters <- iters+1
  }
  
  # Check if max iterations
  if(iters >= max_iters) warning("Algorithm reached maximum number of 
                                 iterations")
  return(list(gamma=gamma, beta=beta))
}

#' Risk model estimation
#' 
#' Returns the estimated gamma and beta for the risk score model along
#' with a glm object with the corresponding coefficients
#' @param X input matrix with dimension n x p, every row is an observation
#' @param y numeric vector for the response variable (binomial)
#' @param gamma starting value to rescale betas for prediction (default NULL)
#' @param beta starting numeric vector with p coefficients (default NULL)
#' @param weights numeric vector of length n with weights for each 
#' observation (defult NULL - will give equal weights)
#' @param lambda0 penalty coefficient for L0 term (default 0)
#' @param a integer lower bound for betas (default -10)
#' @param b integer upper bound for betas (default 10)
#' @param max_iters maximum number of iterations (default 100)
#' @param tol tolerance for convergence
#' @return optimal gamma (numeric) and beta (numeric vector) and corresponding
#' glm object (mod) as a list
risk_mod <- function(X, y, gamma = NULL, beta = NULL, weights = NULL, 
                     lambda0 = 0, a = -10, b = 10, max_iters = 100, tol= 1e-5) {
  
  # Weights
  if (is.null(weights))
    weights <- rep(1, nrow(X))
  
  # If initial gamma is null but have betas then use update function
  if (is.null(gamma) & (!is.null(beta))){
    upd <- update_gamma_intercept(X, y, beta, weights)
    gamma <- upd$gamma
    beta <- upd$beta
  }
  
  # Initial beta is null then round LR coefficients using median 
  if (is.null(beta)){
    # Initial model 
    init_mod <- glm(y~X-1, family = "binomial", weights = weights)
    
    # Replace NA's with 0's
    coef_vals <- unname(coef(init_mod))
    coef_vals[is.na(coef_vals)] <- 0
    
    # Round so betas within range
    gamma <- min(abs(a), abs(b))/max(abs(coef_vals[-1]))
    beta <- coef_vals*gamma
    beta[-1] <- round(beta[-1])
  }
  
  # Check no numeric issues
  if (is.nan(gamma) | sum(is.nan(beta)) > 0){
    stop("Initial gamma or beta is NaN - check starting value for beta")
  }
  if (is.na(gamma) | sum(is.na(beta)) > 0){
    stop("Initial gamma or beta is NA - check starting value for beta")
  }
  if (length(beta) != ncol(X)) stop("beta and X non-compatible")
  if (length(y) != nrow(X)) stop("y and X non-compatible")
  #print(beta)
  # Run coordinate descent from initial solution
  res <- risk_coord_desc(X, y, gamma, beta, weights, lambda0, a, b, max_iters,
                         tol)
  gamma <- res$gamma
  beta <- res$beta
  
  # Convert to GLM object and return
  glm_mod <- glm(y~X-1, family = "binomial", weights = weights, 
      start = gamma*beta, method=glm_fit_risk)
  names(beta) <- names(coef(glm_mod))
  mod <- list(gamma=gamma, beta=beta, glm_mod=glm_mod, X=X, y=y, weights=weights,
                 lambda0 = lambda0)
  class(mod) <- "risk_mod"
  return(mod)
}

#' Cross-Validation to set lambda0
#' 
#' Runs k-fold cross-validation and records class accuracy and deviance
#' @param X input matrix with dimension n x p, every row is an observation
#' @param y numeric vector for the response variable (binomial)
#' @param weights numeric vector of length n with weights for each 
#' observation (defult NULL - will give equal weights)
#' @param a integer lower bound for betas (default -10)
#' @param b integer upper bound for betas (default 10)
#' @param max_iters maximum number of iterations (default 100)
#' @param tol tolerance for convergence
#' @param nlambda number of lambda values to try (default 10)
#' @param lambda_min_ratio smallest value for lambda, as a fraction of
#' lambda_max, the (data derived) entry value (i.e. the smallest value
#' for which all coefficients are zero). The default depends on the sample size 
#' (n) relative to the number of variables (p). If n > p, the default is 0.0001, 
#' close to zero.  If n < p, the default is 0.01.
#' @param lambda0 optional sequence of lambda values (default NULL)
#' @param nfolds number of folds, implied if foldids provided (default 10)
#' @param foldids optional vector of values between 1 and nfolds (default NULL)
#' @return class of cv_risk_mod with a list containing a data.frame of results
#' along with the lambda_min and lambda_1se
cv_risk_mod <- function(X, y, weights = NULL, a = -10, b = 10, max_iters = 100, 
                        tol= 1e-5, nlambda = 25, 
                        lambda_min_ratio = ifelse(nrow(X) < ncol(X), 0.01, 1e-04), 
                        lambda0 = NULL, nfolds = 10, foldids = NULL) {
  # Get folds 
  if (is.null(foldids) & is.null(nfolds)) stop("Must provide foldids or nfolds")
  if (is.null(foldids)){
    foldids <- sample(rep(seq(nfolds), length = nrow(X)))
  } else {
    nfolds <- max(foldids)
  }
  
  # Check at least 3 folds
  if (nfolds <= 3) stop("Must have more than 3 folds")
  
  # Get lambda sequence
  if (is.null(lambda0)){
    sd_n <- function(y) sqrt(sum((y-mean(y))^2)/length(y))
    
    X_scaled <- scale(X[,-1], scale=apply(X[,-1], 2, sd_n))
    X_scaled <- as.matrix(X_scaled, ncol = ncol(X[,-1]), nrow = nrow(X[,-1]))
    y_weighted <- ifelse(y==0, -mean(y == 1), mean(y == 0))
    
    lambda_max <- max(abs(colSums(X_scaled*y_weighted)))/length(y_weighted)
    lambda0 <- exp(seq(log(lambda_max), log(lambda_max * lambda_min_ratio), 
                       length.out=nlambda))
  } 
  
  num_lambda0 <- length(lambda0)
  if (num_lambda0 < 2) stop("Need at least two values for lambda0")

  # Results data frame
  res_df <- data.frame(lambda0 = rep(lambda0, nfolds), 
                       fold = sort(rep(1:nfolds, num_lambda0)),
                       dev = rep(0, nfolds*num_lambda0),
                       acc = rep(0, nfolds*num_lambda0), 
                       non_zeros = rep(0, nfolds*num_lambda0))
  
  # Function to run for single fold and lambda0
  fold_fcn <- function(l0, foldid){
    X_train <- X[foldids != foldid, ]
    y_train <- y[foldids != foldid]
    weight_train <- weights[foldids != foldid]
    mod <- risk_mod(X_train, y_train, gamma = NULL, beta = NULL, 
                    weights = weight_train, lambda0 = l0, a = a, b = b, 
                    max_iters = max_iters, tol= 1e-5)
    res <- get_metrics(mod, X[foldids == foldid,], y[foldids == foldid])
    non_zeros <- sum(mod$beta != 0)
    return(c(res$dev, res$acc, non_zeros))
  }
  
  # Run through all folds
  res_df[,3:5] <- t(sapply(1:nrow(res_df), 
                         function(i) fold_fcn(res_df$lambda0[i], 
                                              res_df$fold[i])))
  # Summarize
  res_df <- res_df %>%
    group_by(lambda0) %>%
    summarize(mean_dev = mean(dev), sd_dev = sd(dev),
              mean_acc = mean(acc), sd_acc = sd(acc))
  
  # Find lambda_min and lambda1_se for deviance
  lambda_min_ind <- which.min(res_df$mean_dev)
  lambda_min <- res_df$lambda0[lambda_min_ind]
  min_dev_1se <- res_df$mean_dev[lambda_min_ind] + 
    res_df$sd_dev[lambda_min_ind] 
  lambda_1se <- res_df$lambda0[max(which(res_df$mean_dev <= min_dev_1se))]
  
  cv_obj <- list(results = res_df, lambda_min = lambda_min, 
                 lambda_1se =lambda_1se)
  class(cv_obj) <- "cv_risk_mod"
  return(cv_obj)
}

#' Get model score card and map scores to risks
#'
#'Summarizes the number of points per feature and maps the each possible total 
#'score to its associated risk.
#' @param mod risk_mod object
#' @return List with model_card and score_map tables
get_model_card <- function(mod) {
  
  # get coefficients
  intercept <- mod$beta[1]
  nonzero_beta <- mod$beta[mod$beta != 0][-1]
  n_nonzero <- length(nonzero_beta)
  
  # save model score card
  model_card <- data.frame(Points = nonzero_beta)

  # find all possible scores
  all_combinations <- expand.grid(rep(list(0:1), n_nonzero))
  for (i in 1:n_nonzero) {
    all_combinations[all_combinations[,i] == 1, i] <- nonzero_beta[i]
  }
  all_scores <- unique(rowSums(all_combinations)) %>% sort()
  
  # save score-to-risk map
  score_map <- data.frame(Score = all_scores, 
                          Risk = round(100*(exp(intercept + all_scores) / (1 + exp(intercept + all_scores))),1))

  return(list(model_card = model_card, score_map = score_map))
}

#' Plot cross-validation results
#'
#'Creates a plot of the deviance for each lambda
#' @param cv_obj cv_risk_mod object
#' @return Plot
plot_cv_results <- function(cv_obj) {

  cv_plot <- ggplot(cv_obj$results,aes(x = lambda0, y = mean_dev)) + 
    geom_point() + 
    geom_errorbar(aes(ymin = mean_dev - sd_dev, ymax= mean_dev + sd_dev)) +
    labs(x = "Lambda", y = "Deviance") + 
    theme_minimal()
  
  return(cv_plot)
}




