######## for s in test set that is not present in the previously constructed score board from train ######
## 2024.1.29: write out the body of algorithm with an example tested out and have it written as a generic function
## next step: write experiments to test on different datasets




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

risk_mod_new_alg <- function(X, y, a = -10, b = 10){
  
  #browser()
  # initial starting vector for betas 
  beta_list <- rep(a,times=ncol(X))
  
  v_list_total <- list()
  
  # train test split 
  test_index <- sample(c(TRUE, FALSE), size = nrow(X), replace = TRUE, prob = c(0.25, 0.75))
  
  # result list of optimal beta
  optimal_beta <- matrix(c( seq(1,ncol(X)),
                           rep(0,times=ncol(X))),ncol = 2)
  
  # iteration
  for (p in 1:ncol(X[!test_index,])) {
    
    beta_result <- matrix(c(seq(a,b),
                            rep(0,times=length(seq(a,b)))),
                          ncol = 2)
    
    for (beta in seq(a,b)) {
      #browser()
      # update value of the current beta evaluting 
      beta_list[p] <- beta
      # calculate score for all trainning under the current beta
      s <- as.matrix(X[!test_index,]) %*% as.matrix(beta_list)
      # extract all unique scores and construct score board 
      uni_s <- unique(s)
      v_list <- matrix(c(uni_s,rep(0,times=length(uni_s))),ncol = 2)
      # calculate corresponding probabilities with the score board 
      for (v in 1:length(uni_s)) {
        v_list[v,2] <- mean(y[!test_index][which(s==uni_s[v])]==1)
      }
      
      # evaluate score board and test error
      s_test <- as.matrix(X[test_index,]) %*% as.matrix(beta_list)
      uni_s_test <- unique(s_test)
      
      # check for rare cases ###### Currently set as 0 
      if (!identical(uni_s,uni_s_test)){
        diff <- c()
        for (vals in uni_s_test) {
          if (!(vals %in% uni_s)) {
            diff <- c(diff,vals)
          }
        }
        # update v_list
        v_list <- rbind(v_list,
                        cbind(diff,rep(0,times=length(diff))))
      }
      
      # convert score to be probs
      y_pred_test <- sapply(s_test, function(s_test) return(v_list[,2][v_list[,1]==s_test]))
      y_pred_test <- unlist(y_pred_test)
      # convert probs to be class
      y_pred_test <- rbinom(length(y_pred_test),1,y_pred_test)
      y_test <- as.numeric(as.matrix(y[test_index]))
      
      # get confusion matrix
      mat <- get_metrics(y=y_test,y_pred = y_pred_test)
      # update error for the current evalutation of beta[p]
      beta_result[,2][beta_result[,1]==beta] <- mat$acc
    }
    
    # find the optimal beta value for beta[p] and update result list
    optimal_beta[,2][optimal_beta[,1]==p] <- beta_result[,1][which.max(beta_result[,2])]
    beta_list[p] <- beta_result[,1][which.max(beta_result[,2])]
    
  }
  
  # construct score board using all data
  s_all <- as.matrix(X) %*% as.matrix(optimal_beta[,2],nrow=1)
  # extract all unique scores and construct score board 
  score <- unique(s_all)
  score_board <- matrix(c(score,rep(0,times=length(score))),ncol = 2)
  # calculate corresponding probabilities with the score board 
  for (v in 1:length(score)) {
    score_board[v,2] <- mean(y[which(s_all==score[v])]==1)
  }
  
  
  return(list(beta <- optimal_beta,
              score_board <- score_board[order(score_board[,1]),] ))
}

#debug(risk_mod_new_alg)
#undebug(risk_mod_new_alg)
#w <- risk_mod_new_alg(X,y,a=-10,b=10)

# Get matrix function
get_metrics <- function(y,y_pred){
  #' Get metrics from beta and gamma
  #' 
  #' Calculates deviance, accuracy, sensitivity, and specificity
  #' @param coef coefficient of previously selected variables
  #' @return list with deviance (dev), accuracy (acc), sensitivity (sens), and 
  #' specificity (spec), 
  
  
  test_y <- y
  pred <- y_pred
  
  #roc_obj <- roc(test_y,p)
  
  #auc <- auc(roc_obj)[1]
  
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
  
  
  return(list(acc=acc, sens=sens, spec=spec, ppv=ppv, npv=npv, f1=f1))
}

