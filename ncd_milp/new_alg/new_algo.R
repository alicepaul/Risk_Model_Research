library(pROC)

#df <- sim_500_5_1_0_1_1_data
#X <- as.matrix(df[,-1])


#y <- as.matrix(df[,1],ncol=1)

#debug(risk_mod_new_alg)
#w <- risk_mod_new_alg(X,y,a=-10,b=10)

# LR and ncd comparison 
# cut off is 0.5


#' Objective function for NLL+penalty
#' 
#' Calculates the objective function for gamma, beta (NLL+penalty)
#' @param p input matrix of probabilities calculated from tentative score board 
#' @param y numeric vector for the response variable (binomial)

#' @param lambda0 penalty coefficient for L0 term (default 0)
#' @return numeric objective function value
obj_fcn <- function(p, y, lambda0=0) {
  
  # Calculate partial derivative for NLL
  #v <- gamma * (X %*% beta)
  #v <- clip_exp_vals(v) # avoids numeric errors
  nll_fcn <- -sum( log(p*y - (1-p)*(1-y)), na.rm = T)
  
  # Penalty term for lambda0*||beta||_0 
  #pen_fcn <- lambda0*sum(beta[-1] != 0) 
  return (nll_fcn)
}


#' Risk Model Estimation with new algo
#' 
#' Returns the estimated optimal betas for the risk score model with 
#' a constructed score board 
#' @param X input matrix with dimension n x p, every row is an observation
#' @param y numeric vector for the response variable (binomial)
#' @param a integer lower bound for betas (default -10)
#' @param b integer upper bound for betas (default 10)
#' @return optimal beta (numeric vector) and corresponding
#' score board with its risk

# auc


risk_mod_new_alg <- function(X, y, a = -10, b = 10,beta = NULL,max_iters=100,tol= 1e-5,weights=NULL){
  
  
  # Weights
  if (is.null(weights)){
    weights <- rep(1, nrow(X))
  }

  
  # Initial beta is null then round LR coefficients using median 
  if (is.null(beta)){
    # Initial model 
    df <- data.frame(X, y)
    init_mod <- glm(y~.-1, family = "binomial", weights = weights, data = df)
    
    # Replace NA's with 0's
    coef_vals <- unname(coef(init_mod))
    coef_vals[is.na(coef_vals)] <- 0
    
    # Round so betas within range
    gamma <- min(abs(a), abs(b))/max(abs(coef_vals))
    beta <- coef_vals*gamma
    # initial starting vector for betas 
    beta <- round(beta)
  }
  

  # iteration
  iters <- 1
  while (iters < max_iters){
    
    for (j in 1:ncol(X)) {
    
    old_beta <- beta
      
    beta_result <- matrix(c(seq(a,b),
                            rep(0,times=length(seq(a,b)))),
                          ncol = 2)
    
    for (b_t in seq(a,b)) {
      
      #browser()
      # update value of the current beta evaluting 
      beta[j] <- b_t
      # calculate score for all trainning under the current beta
      s <- as.matrix(X) %*% as.matrix(beta)
      # extract all unique scores and construct score board 
      uni_s <- unique(s)
      v_list <- matrix(c(uni_s,rep(0,times=length(uni_s))),ncol = 2)
      # calculate corresponding probabilities with the score board 
      for (v in 1:length(uni_s)) {
        v_list[v,2] <- mean(y[which(s==uni_s[v])]==1)
      }
      
      # evaluate score board and test error
      
      # convert score to be probs
      y_pred_test <- sapply(s, function(s) return(v_list[,2][v_list[,1]==s]))
      y_pred_test <- unlist(y_pred_test)
      
      # get objective functions
      obj <- obj_fcn(y_pred_test,y)
      
      # update obj for the current evaluation of beta[p]
      beta_result[,2][beta_result[,1]==b_t] <- obj
    }
    
    # find the optimal beta value for beta[p] and update result list
    beta[j] <- beta_result[,1][which.min(beta_result[,2])]
    }
    
    # Check if change in beta is within tolerance to converge
    if (max(abs(old_beta - beta)) < tol){
      break
    }
    iters <- iters+1
  }
  
  
  # construct score board using all data
  s_all <- as.matrix(X) %*% as.matrix(beta)
  # extract all unique scores and construct score board 
  score <- unique(s_all)
  score_board <- matrix(c(score,rep(0,times=length(score))),ncol = 2)
  # calculate corresponding probabilities with the score board 
  for (v in 1:length(score)) {
    score_board[v,2] <- mean(y[which(s_all==score[v])]==1)
  }
  
  score_board <- score_board[order(score_board[,1]),]
  return(list(beta <- beta,
              score_board <- score_board[order(score_board[,1]),],
              iters <- iters))
}

#debug(risk_mod_new_alg)
#undebug(risk_mod_new_alg)
# <- risk_mod_new_alg(X,y,a=-10,b=10)

# Get matrix function
get_metrics <- function(y,p){
  #' Get metrics from beta and gamma
  #' 
  #' Calculates deviance, accuracy, sensitivity, and specificity
  #' @param coef coefficient of previously selected variables
  #' @return list with deviance (dev), accuracy (acc), sensitivity (sens), and 
  #' specificity (spec), 
  
  
  test_y <- y
  
  
  roc_obj <- roc(test_y,p)
  auc <- auc(roc_obj)[1]
  optimal_cutoff <- coords(roc_obj, "best", ret = "threshold")
  
  pred <- ifelse(p>=optimal_cutoff[1,], 1, 0)
  # Confusion matrix
  tp <- sum(pred == 1 & test_y == 1)
  tn <- sum(pred == 0 & test_y == 0)
  fp <- sum(pred == 1 & test_y == 0)
  fn <- sum(pred == 0 & test_y == 1)
  
  # Accuracy values
  acc <- (tp+tn)/(tp+tn+fp+fn)
  sens <- tp/(tp+fn)
  spec <- tn/(tn+fp)
  ppv <- tp/(tp+fp)
  npv <- tn/(tn+fn)
  f1 <- 2*ppv*sens / (ppv+sens)
  
  
  return(list(acc=acc, sens=sens, spec=spec, ppv=ppv, npv=npv, f1=f1, auc=auc))
}

