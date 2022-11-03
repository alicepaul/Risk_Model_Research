library(dplyr)
library(glmnet)
library(ggplot2)
library(tidyverse)
# Calculates partial derivative for beta_j in logistic regression with
# l0 is L0 penalization, alpha*beta are current coefficients
# Partial derivative only on NLL
par_deriv_nll <- function(alpha, beta, X, y, weights, j)
                  { n <- nrow(X)
                    # Partial derivate for loss
                    # weights = data_weight
                    pd_1 <- alpha * (weights %*% (y * X[,j]))
                    pr <- exp(alpha * (X %*% beta))
                    pr <- pr / (1.0 + pr)
                    pd_2 <- alpha * (weights %*% (X[,j] * pr))
                    #partial derivative
                    db_j_nll <- (-1.0 / n) * (pd_1 - pd_2)
                    return(db_j_nll)
                  }


# Calculates current objective function value in logistic regression with
# l1*L1 loss penalty and alpha*beta are current coefficients
obj_f_nll <- function(alpha, beta, X, y, weights, l0)
            { 
              n <- nrow(X)
              # Calculate probs
              v <- alpha * (X %*% beta)
              obj_1 <- weights %*% (y * v)
              obj_2 <- weights %*% log(1 + exp(v))
              minimize_j_nll <- (-1.0/n) * (obj_1-obj_2) + l0 * sum((beta!=0))
              return (minimize_j_nll)
            }
bisec_search <- function(alpha, beta, X, y, weights, l0, j, a=-10, b=10, TOL=1.0)
                { 
                  beta_a <- copy(beta)
                  beta_a[j] <- a
                  beta_b <- copy(beta)
                  beta_b[j] <- b
                  #NLL(0)
                  beta_0 <- rep(0, dim(beta_a)) # use matrix(0,dim(beta_a)[2], dim(beta_a)[1])?
                  der_f_a <- par_deriv_nll(alpha, beta_a, X, y, weights, j)
                  der_f_b <- par_deriv_nll(alpha, beta_b, X, y, weights, j)
                  
                  # Check that 0 derivative in range
                  search <- TRUE
                  if ((der_f_a > 0) || (der_f_b < 0))
                  {
                    search <- FALSE
                  }
                  while ((b-a > 1) && search)
                  {
                    c <- floor((a+b) / 2)
                    beta_c <- copy(beta)
                    beta_c[j] <- c
                    der_f_c <- par_deriv_nll(alpha, beta_c, X, y, weights, j)
                    if (der_f_c == 0)
                    {
                      return(beta_c)
                    }
                    # Check where to recurse
                    if (sign(der_f_c) == sign(der_f_a))
                    { 
                      a <- c
                      beta_a <- beta_c
                      der_f_a <- der_f_c
                    }
                    else
                    {
                      b <- c
                      beta_b <- beta_c
                      der_f_b <- der_f_c
                    }
                  }
                  # Find best of b and a in objective function
                  obj_a <- obj_f_nll(alpha, beta_a, X, y, weights, l0)
                  obj_b <- obj_f_nll(alpha, beta_b, X, y, weights, l0)
                  ### NEW : comapre NLL(b_j)+l0 < NLL(0)
                  obj_0 <- obj_f_nll(alpha, beta_0, X, y, weights, l0)
                  if ((obj_a < obj_b) && (obj_a < obj_0))
                  {
                    return (beta_a)
                  }
                  #should be else?
                  else if ((obj_0 < obj_a) && (obj_0 < obj_b))
                  {
                    return (beta_0)
                  }
                  return (beta_b)
}

update_alpha <- function(beta, X, y, weights)
                {
                  # Run logistic regression on current integer scores
                  # Calculate scores - ignores intercept
                  #??? 
                  zi <- X[,1:ncol(X)] %*% beta[1:length(beta)]

                  # Runs logistic regression and finds alpha and beta_0
                  # using glm.fit?
                  lr_mod <- glm(y ~ as.vector(zi), data = X, weights=weights, family="binomial")
                  #? using unlist?
                  new_coef <- flatten(extract.coef(lr_mod))
                  lr_summary <- summary(lr_mod)
                  intercept_value <- lr_summary$coefficients[1,1]
                  alpha <- new_coef[0]
                  beta[0] <- lr_res.intercept_ / alpha

                  return (alpha, beta)
}