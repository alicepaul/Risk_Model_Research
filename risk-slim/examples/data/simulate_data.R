
library(dplyr)

simulate_data <- function(n, p1, p2){
  # n: number of observations
  # p1: number of variables with a coefficient between 0 and 1
  # p2: number of variables with a coefficient between 0 and 5
  
  # covariates
  x <- matrix(0, nrow=n, ncol=(p1+p2))
  for (i in 1:(p1+p2)){
    x[,i] <- rbinom(n, 1, runif(1))
  }
  
  # coefficients
  coef <- runif(p1)
  coef <- c(coef, runif(p2,min=0,max=5))
  vals <- x %*% coef
  intercept <- -mean(vals)
  
  # outcome
  vals <- vals+intercept
  probs <- exp(vals)/(1+exp(vals))
  y <- rbinom(n, 1, prob = probs)
  
  return(list(x=x,y=y,coef=coef, intercept =intercept))
}

gen_data <- function(n, p1, p2, filename){
  # Generates data and saves two files - one with data, one with coefficient vals
  data <- simulate_data(n, p1, p2)
  
  # data file
  df <- as.data.frame(data$x)
  df$y <- data$y
  df <- df %>%
    select(y, everything())
  write.csv(df, paste0("/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/data/",filename,"_data.csv"), row.names=FALSE)
  
  # coefficients file
  coef_df <- data.frame(names = c("Intercept",names(df)[1:(p1+p2)]),
                        vals = c(data$intercept, data$coef))
  write.csv(coef_df, paste0("/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/data/",filename,"_coef.csv"), row.names=FALSE)
  
}

# Example of generating data and saving
for (i in 1:10){
  gen_data(1000, 50, 50, paste0("simulate50_", i))
}

