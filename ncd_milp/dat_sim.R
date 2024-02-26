
## This script is the data simulation updated on 01.28.2024
## It has more options of link function from X to Y. Both X,Y are binary

library(dplyr)


#' Simulate Binary data with a link function 
#' 
#' @param n number of observations
#' @param p1: number of variables with a coefficient between 0 and 1
#' @param p2: number of variables with a coefficient between 1 and 5
#' @param eps: Epslon that brings some noise to the data  
#' @param link: link functions; 1:logit link ; 2:probit link ; 3:Complementary Log-Log (cloglog) Link
#' @return a list of data with X,Y,coefficients, and intercept 
simulate_data <- function(n, p1, p2, eps=0, link=1){
  # n: number of observations
  # p1: number of variables with a coefficient between 0 and 1
  # p2: number of variables with a coefficient between 1 and 5
  
  # covariates
  x <- matrix(0, nrow=n, ncol=(p1+p2))
  for (i in 1:(p1+p2)){
    x[,i] <- rbinom(n, 1, runif(1))
  }
  
  # coefficients
  coef <- runif(p1)
  coef <- c(coef, runif(p2,min=1,max=5))
  vals <- x %*% coef
  intercept <- -mean(vals)
  
  # outcome (based on link functions)
  vals <- vals+intercept+rnorm(n,0,eps*sd(vals))
  
  if (link==1){
    # logit link 
    probs <- exp(vals)/(1+exp(vals))
    y <- rbinom(n, 1, prob = probs)
  } else if(link==2){
    # probit link
    probs <- pnorm(vals)
    y <- rbinom(n, 1, prob = probs)
  } else {
    # Complementary Log-Log (cloglog) Link
    probs <- 1 - exp(-exp(vals))
    
    # Simulating Y
    y <- rbinom(n, 1, prob = probs)
  } 
  
  
  return(list(x=x,y=y,coef=coef,intercept=intercept))
}

gen_data <- function(n, p1, p2, eps, link,filename){
  # Generates data and saves two files - one with data, one with coefficient vals
  data <- simulate_data(n, p1, p2, eps, link)
  
  # data file
  df <- as.data.frame(data$x)
  df$y <- data$y
  df <- df %>%
    select(y, everything())
  write.csv(df, paste0("/Users/oscar/Documents/GitHub/Risk_Model_Research/ncd_milp/sim_milp_v1/",filename,"_data.csv"), row.names=FALSE)
  
  # coefficients file
  coef_df <- data.frame(names = c("Intercept",names(df)[1:(p1+p2)]),
                        vals = c(data$intercept, data$coef))
  write.csv(coef_df, paste0("/Users/oscar/Documents/GitHub/Risk_Model_Research/ncd_milp/sim_milp/",filename,"_coef.csv"), row.names=FALSE)
  
}

# Sim for new alg
set.seed(5)
n = c(500,1000)
p1 = c(5,6,7,8,9)
p2 = c(1,2,3,4,5)


for (j in 1:length(n)){
  for (k in 1:length(p1)) {
    for (l in 1:length(p2)) {
      for (eps in c(0, 0.3, 0.5, 1.0)){
        for (link in c(1,2,3)) {
          for (i in 1:10){
            gen_data(n[j], p1[k], p2[k], eps, link,
                     paste0("sim",'_',n[j],'_',p1[k],"_",p2[l],"_",as.integer(10*eps),"_",link,'_',i))
          }
        }
      }
    }
  }
}


# Sim for milp
set.seed(10)
n = 50
p1 = 3
p2 = 2
eps = 0.7


for (link in c(1,2,3)) {
    for (i in 1:10){
            gen_data(n, p1, p2, eps, link,
                     paste0("sim",'_',n,'_',p1,"_",p2,"_",as.integer(10*eps),"_",link,'_',i))
          }
        }



