# Class functions for riskMod object
source("utils.R")

get_metrics <- function(mod, X = NULL, y = NULL, weights = NULL){
  #' Get metrics from beta and gamma
  #' 
  #' Calculates deviance, accuracy, sensitivity, and specificity
  #' @param mod riskMod object
  #' @param X input matrix with dimensions n x p, must match dimensions of beta
  #' in mod (default NULL)
  #' @param y numeric vector for the response variable (binomial) of length n, 
  #' (default NULL)
  #' @return list with deviance (dev), accuracy (acc), sensitivity (sens), and 
  #' specificity (spec), 
  
  # Check if new data
  if (is.null(X)+is.null(y) == 1) stop("Must provide both X and y")
  if (is.null(X) & is.null(y)){
    X = mod$X
    y = mod$y
  }

  # Check compatibility
  if (nrow(X) != length(y)) stop("X and y must match in number of observations")
  if (ncol(X) != length(mod$beta)) stop("X is incompatible with the model")
  if (sum(! (y %in% c(0,1)))) stop("y must be 0/1 valued")
  
  # Get predicted probs and classes
  v <- mod$gamma * X %*% mod$beta
  v <- clip_exp_vals(v)
  p <- exp(v)/(1+exp(v))
  pred <- ifelse(p>=0.5, 1, 0)
  
  # Deviance
  dev <- -2*sum(y*log(p)+(1-y)*log(1-p))
  
  # Confusion matrix
  tp <- sum(pred == 1 & y == 1)
  tn <- sum(pred == 0 & y == 0)
  fp <- sum(pred == 1 & y == 0)
  fn <- sum(pred == 0 & y == 1)
  
  # Accuracy values
  acc <- (tp+tn)/(tp+tn+fp+fn)
  sens <- tp/(tp+fn)
  spec <- tn/(tn+fp)
  
  return(list(dev = dev, acc=acc, sens=sens, spec=spec))
}