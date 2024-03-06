source('new_algo.R')
library(tidyverse)

run_experiments <- function(my_path){
  #' Risk model algorithm experiments
  #' 
  #' Iterates through all csv files in the path and runs risk mod
  #' @param my_path character path to folder of csv files (assumes
  #' files are stored as (x)_data.csv and (x)_weights.csv)
  #' @return results are saved as a csv file results_R.csv in the same folder
  
  # Files in path
  files <- list.files(my_path)
  results <- data.frame(data = character(), n = numeric(), p = numeric(),
                        acc = numeric(), sens = numeric(), spec = numeric(), auc = numeric(),
                        sec = numeric(), iters= numeric())
  
  
  
  # Iterate through files
  for (f in files){
    if (length(grep("_data.csv", f)) == 0) next
    
    # Print for ease
    print(paste0(my_path,f))

    # Read in data
    df <- read.csv(paste0(my_path,f))
    X <- as.matrix(df[,-1])
    y <- as.matrix(df[,1],ncol=1)
    

    test_index <- sample(c(TRUE, FALSE), size = nrow(X), replace = TRUE, prob = c(0.2, 0.8))
    X_train <- X[!test_index,]
    y_train <- y[!test_index,]
    
    X_test <- X[test_index,]
    y_test <- y[test_index,]
    
    # Add weights file if needed
    #weights <- rep(1, nrow(X))
    #weights_file <- paste0(substr(f,1,nchar(f)-8),"_weights.csv")
    #if (file.exists(weights_file)){
    #  weights <- read.csv(weights_file)
    #  weights <- weights[[1]]
    #}
    
    # Run algorithm to get risk model
    t1 <- Sys.time()
    mod <- risk_mod_new_alg(X_train, y_train,a = -5, b = 5) # change for beta bounds
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
    # replace missing probabilities with
    
    # Get evaluation metrics
    time_secs <- t2 - t1
    s_test <- X_test %*% mod[[1]]
    y_pred_probs <- sapply(s_test, function(s_test) return(score_board[,2][score_board[,1]==s_test]))
    y_pred_probs <- unlist(y_pred_probs)
    
    # convert probs to be class : threshold is 0.5
    res_metrics <- get_metrics_mapping(y_test,y_pred_probs)
    
    
    # Add row to data frame
    file_row <- data.frame(data=f, n = nrow(X), p = ncol(X),
                           acc = res_metrics$acc, sens = res_metrics$sens, 
                           spec = res_metrics$spec, auc = res_metrics$auc, sec = time_secs,iters=mod[[3]]) # change for cv  
    results <- rbind(results, file_row)
    
    
  }
  write.csv(results, paste0("/Users/oscar/Documents/GitHub/Risk_Model_Research/ncd_milp/sim_newalg/", "testing_2024.2.26_new.csv"), row.names=FALSE)
  
}

run_experiments("/Users/oscar/Documents/GitHub/Risk_Model_Research/ncd_milp/sim_milp/")

