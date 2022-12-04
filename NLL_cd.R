library(dplyr)
library(glmnet)
library(ggplot2)
library(tidyverse)
library(tibble)
library(caTools)
library(pracma)
library(data.table)
# Calculates partial derivative for beta_j in logistic regression with
# l0 is L0 penalization, alpha*beta are current coefficients
# Partial derivative only on NLL
par_deriv_nll <- function(alpha, beta, X, y, weights, j)
{
                    n <- nrow(X)
                    
                    # Partial derivate for loss
                    # weights = data_weight
                    # need to transform X, y to a numeric/matrix format
                    X <- as.matrix(X)
                    y <- as.numeric(unlist(y))
                    beta <- as.numeric(unlist(beta))
                    weights <- as.numeric(unlist(weights))
                    pd_1 <- alpha * (weights %*% (y * X[,j]))
                    # not sure if the beta here need include coef of weights, 
                    # if not use 2:length-1
                    pr <- exp(alpha * (X %*% beta[2:(length(beta)-1)]))
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
              X <- as.matrix(X)
              y <- as.numeric(unlist(y))
              beta <- as.numeric(unlist(beta))
              weights <- as.numeric(unlist(weights))
              # Calculate probs
              v <- alpha * (X %*% beta[2:(length(beta)-1)])
            
              obj_1 <- weights %*% (y * v)
              obj_2 <- weights %*% log(1 + exp(v))
              minimize_j_nll <- (-1.0/n) * (obj_1-obj_2) + l0 * sum((beta!=0))
              return (minimize_j_nll)
}
bisec_search <- function(alpha, beta, X, y, weights, l0, j, a=-10, b=10, TOL=1.0)
{ 
                  beta_a <- copy(beta)
                  beta_a[[j]] <- a
                  beta_b <- copy(beta)
                  beta_b[[j]] <- b
                  #NLL(0)
                  beta_0 <- rep(0, length(beta_a)) 
                  der_f_a <- par_deriv_nll(alpha, beta_a, X, y, weights, j)
                  der_f_b <- par_deriv_nll(alpha, beta_b, X, y, weights, j)
                  # Check that 0 derivative in range
                  search <- TRUE
                  if ((der_f_a > 0) | (der_f_b < 0))
                  {
                    search <- FALSE
                  }
                  while ((b-a > 1) & search)
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
                  # comapre NLL(b_j)+l0 < NLL(0)
                  obj_0 <- obj_f_nll(alpha, beta_0, X, y, weights, l0)
                  if ((obj_a < obj_b) & (obj_a < obj_0)){
                    return (beta_a)
                  } else {
                    return (beta_0)}
                  return (beta_b)
}

update_alpha <- function(beta, X, y, weights)
{
                  X <- as.matrix(X)
                  y <- as.numeric(unlist(y))
                  beta <- as.numeric(unlist(beta))
                  weights <- as.numeric(unlist(weights)) 
                  # Run logistic regression on current integer scores
                  zi <- X %*% beta[2:(length(beta)-1)]
                  # Runs logistic regression and finds alpha and beta_0
                  lr_mod <- glm(y ~ as.vector(zi), weights=weights, family="binomial")
                  coef_all <- unname(coef(lr_mod))
                  intercept_value <- coef_all[1]
                  alpha <- coef_all[2]
                  beta[1] <- intercept_value / alpha
                  return (list(alpha, beta))
}

coord_desc_nll<- function(X,y, weights, alpha, beta, l0 = 0.0, max_iter = 100, tol= 1e-5)
{
                  X <- as.matrix(X)
                  y <- as.numeric(unlist(y))
                  beta <- as.numeric(unlist(beta))
                  weights <- as.numeric(unlist(weights)) 
                  n <- nrow(X)
                  #weights <- sample_weights_train[[1]]
                  ytemp <- y_train[[1]]
   
                  y <- rep(0, n)
                  for (i in (1:n))
                  {
                    if (ytemp[i] == 1)
                    {
                      y[i] = 1
                    }
                  }
                  p <- ncol(X)
                  iters <- 0
                  #this while loop takes longer than expected
                  while (iters < max_iter)
                  {
                    old_beta <- copy(beta)
                  }
                  # Coodinate descent for each j
                  for (j in (1:p))
                  {
                    beta <- bisec_search(alpha, beta, X, y, weights, l0, j)
                    alpha <- update_alpha(beta, X, y, weights)[1]
                    # during testing, when l0=1e-6, alpha will return NA, use 0 fill NA
                    alpha[is.na(alpha)] <- 0
                    beta <- update_alpha(beta, X, y, weights)[2]
                    # same thing happened here for beta
                    beta[is.na(beta)] <- 0
                  }
                  # Check if change in beta is within tolerance to converge
                  if (max(abs(old_beta - beta)) < tol)
                  {
                    break
                  }
                  iters <- iters+1
                  return(list(alpha, beta))
    
  
  
}
                
  
  
load_data_from_csv <- function(dataset_csv_file, sample_weights_csv_file=NULL)
{
                  dataset_csv_file <- dataset_csv_file
                  if (file.exists(dataset_csv_file) == FALSE)
                  {
                    stop('could not find dataset_csv_file: %s', dataset_csv_file)
                  }
                  
  
                  df <- read.csv(dataset_csv_file, sep = ',')
  
                  #raw_data <- df.to_numpy()
                  data_headers <- list(colnames(df))
                  N <- nrow(df)
                  
                  # setup Y vector and Y_name
                  Y_col_idx <- 1
                  Y <- df[, Y_col_idx, drop=FALSE]
                  Y_name <- data_headers[[Y_col_idx]][1]
                  # Different from python code, they coded to -1
                  #Y[Y == 0] <- 0
                  
                  
                  X_features <- list()
                  # set up x
                  for (j in(colnames(df)))
                  {
                    if ( (j != Y_col_idx))
                    {
                      X_features <- append(X_features,j)
                    }
                  }
                  X_features <- X_features[-1]
                  
                  X <- df[unlist(X_features)]
                  variable_names <- X_features
                  
                  if (file.exists(sample_weights_csv_file) == FALSE)
                  {
                    sample_weights <-rep(1,nrow(X))
                  }else
                  {
                    sample_weights_csv_file <- sample_weights_csv_file
                  }
                    if (file.exists(sample_weights_csv_file) == FALSE)
                    {
                      stop('could not find weight_csv_file: %s', sample_weights_csv_file)
                    }
                      
                    sample_weights <- read.csv(sample_weights_csv_file, header=FALSE, sep=',')
                  
                    data <- list(
                      'X'= X,
                      'Y'= Y,
                      #'variable_names'= variable_names,
                      #'outcome_name'= Y_name,
                      'sample_weights'= sample_weights
                    )
                    #return(list(X=X, Y=Y, variable_names=variable_names, outcome_name=outcome_name,
                                #sample_weights=sample_weights))
                    return(list(data=data, X=X,Y=Y,sample_weights=sample_weights))
}

round_coef<- function(coef, alpha = 1.0)
{
  coef_new <- coef
  coef_new <- coef_new/as.numeric(alpha)
  coef_new <- round(coef_new)
  coef_new[1] <- coef[1]/alpha
  return(coef_new)
}
  


data_breast <- load_data_from_csv('/Users/zhaotongtong/Desktop/Risk_Model_Research/data/breastcancer_data.csv',
                          '/Users/zhaotongtong/Desktop/Risk_Model_Research/data/breastcancer_weights.csv')


#data<- read.csv('/Users/zhaotongtong/Desktop/Risk_Model_Research/data/bank_data.csv', sep=',')

#data <- read.csv('/Users/zhaotongtong/Desktop/Risk_Model_Research/data/tbrisk_data.csv', sep=',')


# split coded into a function?
set.seed(42)
train_ids <- sample(1:nrow(data_breast$X), floor(0.75*nrow(data_breast$X)), replace=FALSE)
data <- as.data.frame(data_breast$data)
train <- data[train_ids, ]
test <- data[-train_ids,]
X_train <- train[,1:(ncol(train)-2)]
y_train <- train[,(ncol(train)-1), drop=FALSE]
X_test <- test[,1:(ncol(train)-2)]
y_test <- test[,(ncol(test)-1), drop=FALSE]
sample_weights_train <- train[,ncol(train), drop=FALSE]
sample_weights_test <- test[,ncol(test), drop=FALSE]

# define lambda0
lambda0 = logspace(-6, -1, 7)

# start computing
for (i in lambda0){
  print(i)
  logistic_model <- glm(train$Benign~.,
                        data = train,
                        weights = train$V1,
                        family = "binomial")
  coef_lr <- coef(logistic_model)
  coef_lr[is.na(coef_lr)] <- 0
  coef_lr <- as.numeric(coef_lr)
  alpha <- max(abs(coef_lr[2:length(coef_lr)]))/10.0
  coef_ncd <- round_coef(coef_lr, alpha=alpha)
  res <- coord_desc_nll(X_train, y_train, sample_weights_train, 1.0/alpha, coef_ncd, i) 
  alpha_ncd <-res[1]
  coef_ncd <- res[2]
}
