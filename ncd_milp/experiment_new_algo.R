source('new_algo.R')

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
                        acc = numeric(), sens = numeric(), spec = numeric(),
                        sec = numeric())
  
  
  
  # Iterate through files
  for (f in files){
    if (length(grep("_data.csv", f)) == 0) next
    
    # Print for ease
    print(paste0(my_path,f))
    
    # Read in data
    df <- read.csv(paste0(my_path,f))
    X <- as.matrix(df[,-1])
    y <- as.matrix(df[,1],ncol=1)
    
    set.seed(1000)
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
    
    
    # Get evaluation metrics
    time_secs <- t2 - t1
    s_test <- X_test %*% mod[[1]]
    y_pred_probs <- sapply(s_test, function(s_test) return(mod[[2]][,2][mod[[2]][,1]==s_test]))
    y_pred_probs <- unlist(y_pred_probs)
    
    # convert probs to be class
    set.seed(1000)
    y_pred_class <- rbinom(length(y_pred_probs),1,y_pred_probs) 
    res_metrics <- get_metrics(y_test,y_pred_class)
    
    
    # Add row to data frame
    file_row <- data.frame(data=f, n = nrow(X), p = ncol(X),
                           acc = res_metrics$acc, sens = res_metrics$sens, 
                           spec = res_metrics$spec, sec = time_secs) # change for cv  
    results <- rbind(results, file_row)
    
  }
  write.csv(results, paste0("/Users/oscar/Documents/GitHub/Risk_Model_Research/ncd_milp/sim_newalg/", "testing_2024.2.3.csv"), row.names=FALSE)
}

run_experiments("/Users/oscar/Documents/GitHub/Risk_Model_Research/ncd_milp/sim_newalg/")


debug(run_experiments)
undebug(run_experiments)

debug(risk_mod_new_alg)
undebug(risk_mod_new_alg)

df <- sim_1000_5_1_0_1_8_data
X <- as.matrix(df[,-1])

y <- as.matrix(df[,1],ncol=1)

w <- risk_mod_new_alg(X,y,a=-2,b=2)

