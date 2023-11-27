library(ompr) # Package for establish OMPR Model
library(ompr.roi) # Solver package
library(ROI.plugin.glpk) # Specific solver used
library(tidyverse)
source('risk.R')


# MILP Example
max_capacity <- 5
n <- 10
set.seed(1234)
weights <- runif(n, max = max_capacity)
MIPModel()  %>% 
  add_variable(x[i], i = 1:n, type = "binary") %>%
  set_objective(sum_over(weights[i] * x[i], i = 1:n), "max") %>%
  add_constraint(sum_over(weights[i] * x[i], i = 1:n) <= max_capacity) %>%
  solve_model(with_ROI(solver = "glpk")) %>%
  get_solution(x[i]) %>%
  filter(value > 0)



# MILP Version 2

# Testing run with smaller sample sized datasets 
# Small data stored at small_data
files <- list.files("/Users/oscar/Documents/GitHub/Risk_Model_Research/MILP/data")
df <- read.csv(paste0("/Users/oscar/Documents/GitHub/Risk_Model_Research/MILP/data/",files[10]))

# Set data matrix
# works out fine for 1:5, errors when trying yo increase sample size
df <- df[10:20,1:5]
y <- df[[1]]
X <- as.matrix(df[,2:ncol(df)])


# Warm start solution
n = nrow(X)
p = ncol(X)
M = 1000 #

Lambda0 = 0
# Score Pool
SK_pool <- c((-5*p) : (5*p)) # Determined by lb and ub of beta

# Probability pool
PI <- seq(0,1,length=100) # Equally spaced between 0,1, length = 100 
# Delete 0 and 1 so that obj can take log
PI <- PI[-c(1,100)]



t1 = Sys.time()
# Model description 
MILP_V2 <- MIPModel() %>% 
  # Integer coefficients for attributes
  add_variable(beta[j], j=1:p, type = 'integer', up=5,lb=-5) %>% 
  # Predicted Risk Score from the integer coefficients 
  add_variable(s[i], i= 1:n, type = 'integer',up=10,lb=-10) %>% 
  # Exchanged this line on 11.8.2023
  add_constraint(sum_over(beta[j]*X[i,j], j=1:p) - s[i] == 0,i=1:n) %>%  
  # Indicator of S_i = k, k: one potential score from the score pool
  add_variable(z_ik[i,k], i=1:n, k=1:length(SK_pool), type = 'binary') %>% 
  add_constraint(sum_over(z_ik[i,k], k = 1:length(SK_pool)) == 1, i=1:n) %>%
    # old
  #add_constraint( s[i] - SK_pool[k] <= M * (1-z_ik[i,k]) , i=1:n,k=1:length(SK_pool)) %>% 
  #add_constraint( s[i] - SK_pool[k] >= -M * (1-z_ik[i,k]), i=1:n,k=1:length(SK_pool)) %>%
    # new
  add_constraint( (s[i] - SK_pool[k]) - (M * (1-z_ik[i,k])) <= 0 , i=1:n,k=1:length(SK_pool)) %>% 
  add_constraint((s[i] - SK_pool[k]) - (-M * (1-z_ik[i,k])) >= 0, i=1:n,k=1:length(SK_pool)) %>%
  # Indicator of score k assigned to prob pi_l
  add_variable(z_kl[k,l], k=1:length(SK_pool), l=1:length(PI), type = 'binary') 

  # higher K is associated with higher probabilities
  # Add constraints in nested loops
  #for (k in 2:length(SK_pool)) { # Start from 2 since we are using k-1
  #  for (l in 1:(length(PI)-1)) {
  #  for (lh in (l+1):length(PI)) {
  #      MILP_V2 <- MILP_V2 %>%
  #        add_constraint(z_kl[k, l] <= 1 - z_kl[k-1, lh])
  #    }
  #  }
  #} 

  MILP_V2 <- MILP_V2 %>% 
  # each k is assigned to exactly 1 probs 
  add_constraint(sum_over(z_kl[k,l], l=1:length(PI)) == 1,k=1:length(SK_pool)) %>% 
  
  # Indicator of point i assigned with prob pi_l
  add_variable(p_il[i,l], i=1:n, l=1:length(PI),type = 'binary') %>% 
    # old 
  #add_constraint(p_il[i,l] >= z_kl[k,l] + z_ik[i,k] - 1, i=1:n, k=1:length(SK_pool), l=1:length(PI)) %>% 
    # new
  add_constraint(p_il[i,l] - z_kl[k,l] - z_ik[i,k] >= -1, i=1:n, k=1:length(SK_pool), l=1:length(PI)) %>% 
  # Each i have exactly one probs
  add_constraint(sum_over(p_il[i,l], l=1:length(PI)) == 1, i=1:n) %>% 
  #Penalty Expression
  #add_variable(lambda[j],j=1:p,type = 'binary') %>% 
  #add_constraint(beta[j] <= M*lambda[j], j=1:p) %>% 
  #add_constraint(beta[j] <= -M*lambda[j], j=1:p) %>% 
  # Objective Function
  set_objective(  - (sum_over(y[i] * log(PI[l]) * p_il[i,l] + (1-y[i]) * log(1-PI[l]) * p_il[i,l],i=1:n,l=1:length(PI)))
                  ,sense = "min") %>% 
  solve_model(with_ROI(solver = "glpk"))

# Lambda in obj
# + Lambda0 * sum_expr(lambda[j],j=1:p

t2 = Sys.time()

time.diff <- t2 - t1

# Extract Coefficients
beta <- MILP_V2 %>% get_solution(beta[j])
s <- MILP_V2 %>% get_solution(s[i])

s_check <- unlist(lapply(1:10, function(i) t(as.matrix(beta[,3])) %*% as.matrix(X[i,])))

SK_pool[11]

#
zik <- MILP_V2 %>% get_solution(z_ik[i,k])
zik %>% filter(value==1)
zik %>% group_by(i) %>% summarise(w <- sum(value))
# 
zkl <- MILP_V2 %>% get_solution(z_kl[k,l]) 
zkl %>% group_by(k) %>% summarise(w <- sum(value))
zkl %>% filter(value==1)
# 
pil <- MILP_V2 %>% get_solution(p_il[i,l]) 
pil %>% group_by(i) %>% summarise(sum_p = sum(value))
pil %>% filter(value==1)



# Lambda
# lambda_list <- MILP_V2 %>% get_solution(lambda[j]) 

MILP_V2$objective_value




# Results from risk.mod
df <- df[1:10,1:5]
y <- df[[1]]
X <- as.matrix(df[,2:ncol(df)])
X <- cbind(rep(1,nrow(X)), X) 


risk_df <- risk_mod(X, y, gamma = NULL, beta = NULL, weights = NULL, 
                    lambda0 = 0, a = -5, b = 5, max_iters = 100, tol= 1e-5)
