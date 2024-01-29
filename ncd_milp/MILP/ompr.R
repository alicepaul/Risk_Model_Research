library(ompr) # Package for establish OMPR Model
library(ompr.roi) # Solver package
library(ROI.plugin.glpk) # Specific solver used
library(tidyverse)


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
df <- df[5:10,]
y <- df[[1]]
X <- as.matrix(df[,2:ncol(df)])


# Warm start solution
n = nrow(X)
p = ncol(X)
M = 999999999 #

Lambda0 = 0
# Score Pool
SK_pool = c((-5*p) : (5*p)) # Determined by lb and ub of beta

# Probability pool
PI = seq(0,1,length=100) # Equally spaced between 0,1, length = 100 
# Delete 0 and 1 so that obj can take log
PI <- PI[-c(1,100)]

t1 = Sys.time()
# Model description 
MILP_V2 <- MIPModel() %>% 
  # Integer coefficients for attributes
  add_variable(beta[j], j=1:p, type = 'integer') %>% 
  # Predicted Risk Score from the integer coefficients 
  add_variable(s[i], i= 1:n, type = 'integer') %>% 
  # Exchanged this line on 11.8.2023
  add_constraint(sum_over(beta[j]*X[i,j], j=1:p) - s[i] == 0, i=1:n) %>%  
  # Indicator of S_i = k, k: one potential score from the score pool
  add_variable(z_ik[i,k], i=1:n, k=1:length(SK_pool), type = 'binary') %>% 
  add_constraint(sum_over(z_ik[i,k], k = 1:length(SK_pool)) == 1, i=1:n) %>%
  add_constraint(s[i] - SK_pool[k] <= M * (1-z_ik[i,k]), i=1:n,k=1:length(SK_pool)) %>% 
  add_constraint(s[i] - SK_pool[k] >= -M * (1-z_ik[i,k]), i=1:n,k=1:length(SK_pool)) %>%
  # Indicator of score k assigned to prob pi_l
  add_variable(z_kl[k,l], k=1:length(SK_pool), l=1:length(PI), type = 'binary') %>% 
  # each k is asigened to exactly 1 probs 
  add_constraint(sum_expr(z_kl[k,l], l=1:length(PI)) == 1,k=1:length(SK_pool)) %>%
  # higher k is associated with higher probs 
  #add_constraint(z_kl[k,l] <= 1 - z_kl[k-1,l_star], l=1:length(PI),l_star=1:length(PI),k=1:length(K)) %>% 
  # Indicator of point i assigned with probs pi_l
  add_variable(p_il[i,l], i=1:n, l=1:length(PI),type = 'binary') %>% 
  add_constraint(p_il[i,l] >= z_kl[k,l] + z_ik[i,k] - 1, i=1:n,k=1:length(SK_pool),l=1:length(PI)) %>% 
  # Each i have exactly one probs
  add_constraint(sum_expr(p_il[i,l], l=1:length(PI)) == 1, i=1:n) %>% 
  # Penalty Expression
  #add_variable(lambda[j],j=1:p,type = 'binary') %>% 
  #add_constraint(beta[j] <= M*lambda[j], j=1:p) %>% 
  #add_constraint(beta[j] <= -M*lambda[j], j=1:p) %>% 
  # Objective Function
  set_objective( - (sum_over(y[i] * log(PI[l]) * p_il[i,l] + (1-y[i]) * log(1-PI[l]) * p_il[i,l], i=1:n,l=1:length(PI)) 
                    ), sense = "min") %>% 
  solve_model(with_ROI(solver = "glpk"))

# Lambda in obj
# + Lambda0 * sum_expr(lambda[j],j=1:p
t2 = Sys.time()

time.diff <- t2 - t1

# Extract Coefficients
beta <- MILP_V2 %>% get_solution(beta[j])
s <- MILP_V2 %>% get_solution(s[i])
# First K for every i k == 50
zik <- MILP_V2 %>% get_solution(z_ik[i,k])
zik %>% group_by(i) %>% summarise(w <- sum(value))
# First l for every k
zkl <- MILP_V2 %>% get_solution(z_kl[k,l]) 
zkl %>% group_by(k) %>% summarise(w <- sum(value))
zkl %>% filter(k==1,value==1)
# first l for every i
pil <- MILP_V2 %>% get_solution(p_il[i,l]) 
pil %>% filter(value==1)

MILP_V2$objective_value

# Based on the MILP, Beta = 0,0,-50,0,0,0,0,0,0,0 ; S = -50,-50,-50,-50,-50 ; optimal k <- 1, l <- 59
- (log(PI[17]) * sum(y==1) + log(1-PI[17]) * sum(y!=1))
