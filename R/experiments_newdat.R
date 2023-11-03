source('risk.R')
library(doParallel)
registerDoParallel(cores=8)
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
                        lambda0 = numeric(), non_zeros = numeric(), 
                        med_abs = numeric(), max_abs = numeric(),
                        acc = numeric(), sens = numeric(), spec = numeric(),
                        sec = numeric())
  
  results_2 <- data.frame(data = character(), n = numeric(), p = numeric(),
                          lambda0 = numeric(), non_zeros = numeric(), 
                          med_abs = numeric(), max_abs = numeric(),
                          acc = numeric(), sens = numeric(), spec = numeric(),
                          sec = numeric())
  
  # Iterate through files
  for (f in files){
    if (length(grep("_data.csv", f)) == 0) next
    
    # Print for ease
    print(paste0(my_path,f))
    
    # Read in data
    df <- read.csv(paste0(my_path,f))
    y <- df[[1]]
    X <- as.matrix(df[,2:ncol(df)])
    X <- cbind(rep(1,nrow(X)), X) # adds intercept column
    
    # Add weights file if needed
    weights <- rep(1, nrow(X))
    weights_file <- paste0(substr(f,1,nchar(f)-8),"_weights.csv")
    if (file.exists(weights_file)){
      weights <- read.csv(weights_file)
      weights <- weights[[1]]
    }
    
    # Run algorithm to get risk model
    # Testing for lambda_min as lambda 0
    t1 <- Sys.time()
    lambda0 <- cv_risk_mod(X, y, weights=weights, nfolds = 5,parallel = T)$lambda_min # chenge for cv
    mod <- risk_mod(X, y, weights=weights, lambda0 = lambda0) # change for cv
    t2 <- Sys.time()
    
    
    # Get evaluation metrics
    time_secs <- t2 - t1
    res_metrics <- get_metrics(mod)
    non_zeros <- sum(mod$beta[-1] != 0)
    med_abs <- median(abs(mod$beta[-1]))
    max_abs <- max(abs(mod$beta[-1]))
    
    # Add row to data frame
    file_row <- data.frame(data=f, n = nrow(X), p = ncol(X), lambda0 = lambda0, 
                           non_zeros = non_zeros, med_abs = med_abs, max_abs = max_abs,
                           acc = res_metrics$acc, sens = res_metrics$sens, 
                           spec = res_metrics$spec, sec = time_secs) # change for cv  
    results <- rbind(results, file_row)
    
    # Testing for lambda_1se as lambda 0
    t1 <- Sys.time()
    lambda0 <- cv_risk_mod(X, y, weights=weights, nfolds = 5,parallel = T)$lambda_1se # chenge for cv
    mod <- risk_mod(X, y, weights=weights, lambda0 = lambda0) # change for cv
    t2 <- Sys.time()
    
    # Get evaluation metrics
    time_secs <- t2 - t1
    res_metrics <- get_metrics(mod)
    non_zeros <- sum(mod$beta[-1] != 0)
    med_abs <- median(abs(mod$beta[-1]))
    max_abs <- max(abs(mod$beta[-1]))
    
    # Add row to data frame
    file_row <- data.frame(data=f, n = nrow(X), p = ncol(X), lambda0 = lambda0, 
                           non_zeros = non_zeros, med_abs = med_abs, max_abs = max_abs,
                           acc = res_metrics$acc, sens = res_metrics$sens, 
                           spec = res_metrics$spec, sec = time_secs) # change for cv  
    results_2 <- rbind(results_2, file_row)
    
  }
  write.csv(results, paste0("/Users/oscar/Documents/GitHub/Risk_Model_Research/", "ncd_cv_R_newdat_min_para.csv"), row.names=FALSE)
  write.csv(results_2, paste0("/Users/oscar/Documents/GitHub/Risk_Model_Research/", "ncd_cv_R_newdat_1se_para.csv"), row.names=FALSE)
}

run_experiments("/Users/oscar/Documents/GitHub/Risk_Model_Research/sim_dat_new/")
