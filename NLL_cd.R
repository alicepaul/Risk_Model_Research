library(dplyr)
library(glmnet)
library(ggplot2)
library(tidyverse)
library(tibble)
library(caTools)
library(pracma)
# Calculates partial derivative for beta_j in logistic regression with
# l0 is L0 penalization, alpha*beta are current coefficients
# Partial derivative only on NLL
par_deriv_nll <- function(alpha, beta, X, y, weights, j)
{
                    n <- nrow(X)
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
                  ### NEW : comapre NLL(b_j)+l0 < NLL(0)
                  obj_0 <- obj_f_nll(alpha, beta_0, X, y, weights, l0)
                  if ((obj_a < obj_b) & (obj_a < obj_0))
                  {
                    return (beta_a)
                  }
                  #should be else?
                  else #((obj_0 < obj_a) && (obj_0 < obj_b))
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
                  zi <- X[,2:ncol(X)] %*% beta[2:length(beta)]

                  # Runs logistic regression and finds alpha and beta_0
                  # using glm.fit?
                  lr_mod <- glm(y ~ as.vector(zi), weights=weights, family="binomial")
                  coef_all <- unname(coef(lr_mod))
                  #lr_summary <- summary(lr_mod)
                  intercept_value <- coef_all[1]
                  alpha <- coef_all[2]
                  beta[0] <- intercept_value / alpha

                  return (alpha, beta)
}

coord_desc_nll<- function(data, alpha, beta, l0 = 0.0, max_iter = 100, tol= 1e-5)
{
                  X <- data$X
                  #n <- nrow(X)
                  weights <- data$sample_weights.flatten()
                  ytemp <- data$y
                  y <- rep(0, length(X))
                  for (i in n)
                  {
                    if (ytemp[i] == 1)
                    {
                      y[i] = 1
                    }
                  }
                  p <- ncol(X)
                  iters <- 0
                  while (iters < max_iter)
                  {
                    old_beta <- copy(beta)
                  }
                  # Coodinate descent for each j
                  for (j in (1:p))
                  {
                    beta <- bisec_search(alpha, beta, X, y, weights, l0, j)
                    alpha <- update_alpha(beta, X, y, weights)[1]
                    beta <- update_alpha(beta, X, y, weights)[2]
                  }
                  # Check if change in beta is within tolerance to converge
                  if (max(abs(old_beta - beta)) < tol)
                  {
                    break
                  }
                  iters <- iters+1
                  
                  return(alpha, beta)
    
  
  
}
                
  
  
load_data_from_csv <- function(dataset_csv_file, sample_weights_csv_file)
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
                  Y <- df[, Y_col_idx]
                  Y_name <- data_headers[[Y_col_idx]][1]
                  Y[Y == 0] <- -1
                  
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
                  #for (j in X_features)
                  #{
                    #variable_names <- append(variable_names,data_headers[j])
                  #}
                  
                  # insert a column of ones to X for the intercept
                  intercept <- rep(1,nrow(X))
                  X <- as.data.frame(cbind(intercept,X))
                  variable_names <- append('intercept', variable_names)
                  
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
                      'variable_names'= variable_names,
                      'outcome_name'= Y_name,
                      'sample_weights'= sample_weights
                    )
                    return(data)
}




#data <- load_data_from_csv('/Users/zhaotongtong/Desktop/Risk_Model_Research/data/breastcancer_data.csv',
                          # '/Users/zhaotongtong/Desktop/Risk_Model_Research/data/breastcancer_weights.csv')

data_bank <- read.csv('/Users/zhaotongtong/Desktop/Risk_Model_Research/data/bank_data.csv', sep=',')

# set Y
Y_col_idx <- 1
Y <- data_bank[, Y_col_idx, drop=FALSE]
data_headers <- list(colnames(data_bank))
Y_name <- data_headers[[Y_col_idx]][1]
Y[Y == 0] <- 0

#set X
X_features <- list()
# set up x
for (j in(colnames(data_bank)))
{
  if ( (j != Y_col_idx))
  {
    X_features <- append(X_features,j)
  }
}
X_features <- X_features[-1]
X <- data_bank[unlist(X_features)]
variable_names <- X_features

#set intercept
intercept <- rep(1,nrow(X))
X <- as.data.frame(cbind(intercept,X))
variable_names <- append('intercept', variable_names)

# set weight file
sample_weights <-rep(1,nrow(X))

# concate X, Y, weight
data_bank <- cbind(X,Y,sample_weights)

# split
set.seed(42)
sample <- sample.split(data_bank$sign_up, SplitRatio = 0.75)
train  <- subset(data_bank, sample == TRUE)
test   <- subset(data_bank, sample == FALSE)

X_train <- train[unlist(X_features)]
y_train <- train[,ncol(train)-1, drop=FALSE]
X_test <- test[unlist(variable_names)]
y_test <- test[,ncol(test)-1, drop=FALSE]
sample_weights_train <- train[,ncol(train), drop=FALSE]
sample_weights_test <- test[,ncol(test), drop=FALSE]

variable_name_train <- variable_names
variable_name_test <- variable_names

# define lambda0
lambda0 = logspace(-6, -1, 7)

Y <- as.factor(train$Y)
y_train_1 <- as.factor(y_train)
# start computing
for (i in range(lambda0))
{
  logistic_model <- glm(unlist(y_train) ~ ., 
                        data =cbind(y_train, X_train), weights = unlist(sample_weights_train),
                        family = "binomial")
  coef_lr <- coef(summary(logistic_model))
  
}