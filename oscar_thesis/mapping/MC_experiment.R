source('new_algo.R')
source('risk.R')
source('dat_sim.R')
library(tidyverse)
suppressMessages(get_metrics_mapping)
options(warn=-1)
options(message=-1)

MC_func <- function(nobs,p1,p2,eps=0,link=1,a=5,b=-5,nsim=500){
  
  t_beta <- runif(p1)
  t_beta <- c(t_beta, runif(p2,min=1,max=5))
  
  
  # create result dataset
  results_ccd <- data.frame(sim = numeric(), n = numeric(), p = numeric(),
                        lambda0 = numeric(), non_zeros = numeric(), 
                        med_abs = numeric(), max_abs = numeric(),auc=numeric(),
                        acc = numeric(), sens = numeric(), spec = numeric(),
                        sec = numeric())
  
  
  col_names <- paste0("b", 1:(p1+p2))
  # create beta result 
  beta_res_ccd <- data.frame(matrix(ncol = p1+p2, nrow = 0))
  names(beta_res_ccd) <- col_names
  

  
  results_mapping <- data.frame(sim = numeric(), n = numeric(), p = numeric(),
                        acc = numeric(), sens = numeric(), spec = numeric(), auc = numeric(),
                        sec = numeric(), iters= numeric())
  
  beta_res_mapping <- data.frame(matrix(ncol = p1+p2, nrow = 0))
  names(beta_res_mapping) <- col_names
  
  # run models with nsim
  for (i in 1:nsim) {
    
    df <- simulate_data(n=nobs,coef=t_beta,eps = eps,link = link)
    
    # ccd model:
    y <- df$y
    X <- as.matrix(df$x)
    X <- cbind(rep(1,nrow(X)), X)
    
    t1 <- Sys.time()
    #lambda0 <- cv_risk_mod(X, y, weights=weights, nfolds = 5)$lambda_min # chenge for cv
    #lambda1se <- cv_risk_mod(X, y, weights=weights, nfolds = 5)$lambda_1se
    mod <- risk_mod(X, y, weights=rep(1, nrow(X)), lambda0 = 0) # change for cv
    t2 <- Sys.time()
    
    # Get evaluation metrics
    time_secs <- t2 - t1
    res_metrics <- get_metrics(mod)
    non_zeros <- sum(mod$beta[-1] != 0)
    med_abs <- median(abs(mod$beta[-1]))
    max_abs <- max(abs(mod$beta[-1]))
    probs <- predict.risk_mod(mod,type ='response')
    roc_obj <- roc(y,probs)
    auc <- auc(roc_obj)[1]
    
    # Add row to data frame
    file_row <- data.frame(sim=i, n = nrow(X), p = ncol(X), lambda0 = 0, 
                           non_zeros = non_zeros, med_abs = med_abs, max_abs = max_abs,auc=auc,
                           acc = res_metrics$acc, sens = res_metrics$sens, 
                           spec = res_metrics$spec, sec = time_secs) # change for cv  
    results_ccd <- rbind(results_ccd, file_row)
    
    # Add beta_hat to beta_res
    beta_res_ccd <- rbind(mod$beta[-1],beta_res_ccd)
    
  }
  
  
  # run models with mapping_score
  for (j in 1:nsim) {
    
    data <- simulate_data(n=nobs,coef=t_beta,eps = eps,link = link)
    df <- as.data.frame(data$x)
    df$y <- data$y
    df <- df %>%
      select(y, everything())

    
    X <- as.matrix(df[,-1])
    y <- as.matrix(df[,1],ncol=1)
    
    
    test_index <- sample(c(TRUE, FALSE), size = nrow(X), replace = TRUE, prob = c(0.2, 0.8))
    X_train <- X[!test_index,]
    y_train <- y[!test_index,]
    
    X_test <- X[test_index,]
    y_test <- y[test_index,]
    

    # Run algorithm to get risk model
    t1 <- Sys.time()
    mod <- risk_mod_new_alg(X_train, y_train,a = a, b = b) # change for beta bounds
    t2 <- Sys.time()
    
    
    # get range of possible scores
    X_nonzero <- X[,which(mod[[1]] != 0)]
    nonzero_beta <- mod[[1]][which(mod[[1]] != 0)]
    min_pts <- rep(NA, length(nonzero_beta))
    max_pts <- rep(NA, length(nonzero_beta))
    for (i in 1:ncol(X_nonzero)) {
      temp <- nonzero_beta[i] * c(min(X_nonzero[,i]), max(X_nonzero[,i]))
      min_pts[i] <- min(temp)
      max_pts[i] <- max(temp)
    }
    
    score_range <- seq(sum(min_pts), sum(max_pts)) 
    # Create a modified score board
    score_range <- matrix(c(score_range,rep(NA,times=4 * length(score_range))),ncol=5)
    
    # Get # of obs from testing y with corresponding scores
    score_y <- X_train %*% mod[[1]] 
    score_y <- as.data.frame(table(score_y))
    
    # match possible score with existing score board 
    score_range[match(score_y[, 1], score_range[, 1]), 2] <- score_y[, 2] 
    score_range[match(mod[[2]][, 1], score_range[, 1]), 3] <- mod[[2]][, 2] 
    
    # Weighted version of score board 
    for (i in 1:nrow(score_range)) {
      if(is.na(score_range[i,2]) | score_range[i,2] < 5) {
        #construct proportion weights
        score_range[,4] <- 1/sapply(score_range[,1], function(x) (x-score_range[i,1])^2)
        score_range[,4][is.infinite(score_range[,4])] <- 0
        score_range[,5] <- score_range[,3] * score_range[,4]
        score_range[i,3] <- sum(score_range[,5],na.rm = T) / sum(score_range[,4],na.rm = T)
      }
    }
    
    score_board <- score_range[,c(1,3)]

    
    # Get evaluation metrics
    time_secs <- t2 - t1
    s_test <- X_test %*% mod[[1]]
    y_pred_probs <- sapply(s_test, function(s_test) return(score_board[,2][score_board[,1]==s_test]))
    y_pred_probs <- unlist(y_pred_probs)
    
    # convert probs to be class : threshold is determined by auc
    res_metrics <- get_metrics_mapping(y_test,y_pred_probs)
    
    
    # Add row to data frame
    file_row <- data.frame(sim=j, n = nrow(X), p = ncol(X),
                           acc = res_metrics$acc, sens = res_metrics$sens, 
                           spec = res_metrics$spec, auc = res_metrics$auc, sec = time_secs,iters=mod[[3]]) # change for cv  
    results_mapping <- rbind(results_mapping, file_row)
    
    # Add beta_hat to beta_res
    beta_res_mapping <- rbind(mod[[1]],beta_res_mapping)
    
  }
  
  names(beta_res_ccd) <- col_names
  names(beta_res_mapping) <- col_names
  
  return(list(t_beta=t_beta,
    results_ccd = results_ccd,beta_res_ccd=beta_res_ccd,results_mapping=results_mapping,beta_res_mapping=beta_res_mapping))
}



sim1000 <- MC_func(nobs=1000,p1=4,p2=5,eps=0,link=1,a=5,b=-5,nsim=1000)



