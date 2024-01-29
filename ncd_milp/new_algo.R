######## for s in test set that is not present in the previously constructed score board from train ######
X <- simulate5_5_10_4_data[,-1]
y <- simulate5_5_10_4_data[,1]
beta_list <- rep(-5,times=ncol(X))
v_list_total <- list()

test_index <- sample(c(TRUE, FALSE), size = nrow(X), replace = TRUE, prob = c(0.25, 0.75))
optimal_beta <- matrix(c(seq(1,ncol(X)),rep(0,times=10)),ncol = 2)

for (p in 1:ncol(X[!test_index,])) {
  
  beta_result <- matrix(c(seq(-5,5),rep(0,times=11)),ncol = 2)
  
  for (beta in seq(-5,5)) {
    # construct score board
    beta_list[p] <- beta
    s <- as.matrix(X[!test_index,]) %*% as.matrix(beta_list)
    uni_s <- unique(s)
    v_list <- matrix(c(uni_s,rep(0,times=length(uni_s))),ncol = 2)
    for (v in 1:length(uni_s)) {
      v_list[v,2] <- mean(y[!test_index,][which(s==uni_s[v]),]==1)
    }
    
    # evaluate score board and test error
    s_test <- as.matrix(X[test_index,]) %*% as.matrix(beta_list)
    # convert score to be probs
    y_pred_test <- sapply(s_test, function(s_test) return(v_list[,2][v_list[,1]==s_test]))
    y_pred_test <- unlist(y_pred_test)
    # convert probs to be class
    y_pred_test <- rbinom(length(y_pred_test),1,y_pred_test)
    y_test <- as.numeric(as.matrix(y[test_index,]))
    
    # get conf# get conf# get confusion matrix
    mat <- get_metrics(y=y_test,y_pred = y_pred_test)
    beta_result[,2][beta_result[,1]==beta] <- mat$acc
  }
  
  optimal_beta[,2][optimal_beta[,1]==p] <- beta_result[,1][which.max(beta_result[,2])]
  
}

debug(get_metrics)
undebug(get_metrics)
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
