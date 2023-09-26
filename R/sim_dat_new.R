# Simulate_data_new 
# Adopted form https://github.com/alicepaul/var_selection/blob/main/gen_syn_data.py
# Included more randomness and higher degrees of freedom to the data_generation process for further testing of the ncd algorithm

library(tidyverse)
library(MASS) # for multinomal 

sim_dat = function(n, p1,p2,p3,rho=0.5, snr=5){
  #' Data Simulation function
  #' The data matrix x is sampled from a multivariate gaussian with exponential correlation between columns.
  #' The response y = xb + epsilon, where b is a vector with 'supp_size', defined as randomly chosen entries equal to 1 and the rest equal to 0.
  #' The error term epsilon is sampled from an normal distribution (independent of x).
  #' 
  #' Inputs:
  #' @param n number of sample to generate in each dataset
  #' @param p1 number of variables with a coefficient between 0 and 1
  #' @param p2 number of variables with a coefficient between 1 and 5
  #' @param p3 number of variables with a coefficient of 0
  #' @param rho correlation coefficients of the X matrix. Default is 0.5
  #' @param snr signal to noise ratio. Default is 5
  #' files are stored as (x)_data.csv and (x)_weights.csv)
  #' @return a list containing X,Y,corfficients,and intercept matrix
  
  # define total number of columns
  p=p1+p2+p3
  
  # Define support size
  supp_size = as.integer(0.1*p)
  support_mat = matrix(0,nrow = n ,ncol = supp_size)
  
  # Generate X matrix from multivariate Gaussian with exponential correlation 
  cov_mat = matrix(0,nrow = p,ncol = p)
  for (row in 1:p) {
    for (col in 1:p) {
      cov_mat[row, col] = rho ** abs(row-col)
    }
  }
  
  x = mvrnorm(n=n,mu=rep(0,p),Sigma = cov_mat)
  
  x_centered = x - colMeans(x)
  x_normalized =  x_centered / norm(x_centered, type = "2")
  
  b <- runif(p1)
  b <- c(b, runif(p2,min=1,max=5), rep(0,p3))
  
  mu <- x %*% b
  intercept <- -mean(mu)
  
  
  #unshuffled_support = c()
  #for (i in 1:p) {
  #  if (i %% (p/supp_size) == 0) {
  #    unshuffled_support <- c(unshuffled_support, i)
  #  }
  #}

  #b = matrix(0,nrow = p,ncol = 1)
  #b[unshuffled_support] = 1
  
  # get mu = x * b
  #mu = x %*% b
  
  # Calculate var_xb
  var_xb <- var(mu, na.rm = TRUE)
  
  # Calculate sd_epsilon
  sd_epsilon <- sqrt(var_xb / snr)
  
  # Generate random epsilon values with the specified standard deviation
  epsilon <- rnorm(n, mean = 0, sd = sd_epsilon)
  
  mu_star = mu +  matrix(epsilon, ncol = 1)
  probs <- exp(mu_star)/(1+exp(mu_star))
  y <- rbinom(n, 1, prob = probs)

  
  return(list(x=x_normalized,y=y,coef=b,intercept = intercept ))
}



gen_data <- function(n, p1, p2, p3, rho,snr,filename){
  # Generates data and saves two files - one with data, one with coefficient vals
  data <- sim_dat(n, p1, p2, p3, rho,snr)

  # data file
  df <- as.data.frame(data$x)
  df$y <- data$y
  df <- df %>%
    dplyr::select(y, everything()) # Define package, easily get conflicted 
  write.csv(df, paste0("sim_dat_new/",filename,"_data.csv"), row.names=FALSE)
  
  # coefficients file
  coef_df <- data.frame(names = c("Intercept",names(df)[-1]), # changed to make variable names correct
                        vals = c(data$intercept, data$coef))
  write.csv(coef_df, paste0("sim_dat_new/",filename,"_coef.csv"), row.names=FALSE)
}


set.seed(5)
p1 = c(0, 5, 10)
p2 = c(10, 5, 10)
p3 = c(1,2,3)
for (j in 1:length(p1)){
  for (rho in c(0, 0.3, 0.5)){
    for (i in 1:5){
      gen_data(1000, p1[j], p2[j], p3[j],rho,snr=5,
               paste0("simulate",p1[j],"_",p2[j],"_",rho,"_",i) )
    }
  }
}
