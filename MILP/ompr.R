library(ompr) # Package for establish OMPR Model
library(ompr.roi) # Solver package
library(ROI.plugin.glpk) # Specific solver used
library(tidyverse)

# Read in Data for illustration 
files <- list.files("/Users/oscar/Documents/GitHub/Risk_Model_Research/MILP/data")
df <- read.csv(paste0("/Users/oscar/Documents/GitHub/Risk_Model_Research/MILP/data/",files[1]))

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

# Set dat matrix
y <- df[[1]]
X <- as.matrix(df[,2:ncol(df)])


# MILP Version 2
n = nrow(X)
p = ncol(X)
M = Inf

Lambda0 = 0
# Score Pool
K = c(0,seq(1:10))
# Probability pool
PI = sort(runif(100))
MIPModel() %>% 
  # Integer coefficients for attributes
  add_variable(beta[j], j=1:p, type = 'integer') %>% 
  # Predicted Risk Score from the integer coefficients : ??? should we put constraints 
  add_variable(s[i], i= 1:n, type = 'integer') %>% 
  add_constraint(sum_expr(beta[j]*X[i,j], j=1:p) == s[i], i=1:n) %>% 
  # True Risk Score, set bounds 
  # add_variable(k[a],a=1:n,type = 'integer',lb=0,up=10) %>% 
  # Indicator of S_i = k, k: one potential score from the score pool
  add_variable(z_ik[i,k], i=1:n, k=1:length(K), type = 'binary') %>% 
  add_constraint(sum_expr(z_ik[i,k], k = 1:length(K)) == 1, i=1:n) %>%
  add_constraint(s[i] - K[k] <= M * (1-z_ik[i,k]), i=1:n,k=1:length(K)) %>% 
  add_constraint(s[i] - K[k] >= -M * (1-z_ik[i,k]), i=1:n,k=1:length(K)) %>%
  # Indicator of score k assigned to prob pi_l
  add_variable(z_kl[k,l], k=1:length(K), l=1:length(PI), type = 'binary') %>% 
  # each k is asigened to exactly 1 probs 
  add_constraint(sum_expr(z_kl[k,l], l=1:length(PI)) == 1,k=1:length(K)) %>%
  # higher k is associated with higher probs ???
  #add_constraint(z_kl[k,l] <= 1 - z_kl[k-1,l_star], l=1:length(PI),l_star=1:length(PI),k=1:length(K)) %>% 
  # Indicator of point i assigned with probs pi_l
  add_variable(pr[i,l], i=1:n, l=1:length(PI),type = 'binary') %>% 
  add_constraint(pr[i,l] >= z_kl[k,l] + z_ik[i,k] - 1, i=1:n,k=1:length(K),l=1:length(PI)) %>% 
  # Each i have exactly one probs
  add_constraint(sum_expr(pr[i,l], l=1:length(PI)) == 1, i=1:n) %>% 
  # Penalty Expression
  add_variable(lambda[j],j=1:p,type = 'binary') %>% 
  add_constraint(beta[j] <= M*lambda[j], j=1:p) %>% 
  add_constraint(beta[j] <= -M*lambda[j], j=1:p) %>% 
  # Objective Function
  set_objective(- sum_expr(y[i] * log(PI[l]) * pr[i,l] + (1-y[i]) * log(1-PI[l]) * pr[i,l], i=1:n,l=1:length(PI))
                + Lambda0 * sum_expr(lambda[j],j=1:p) , sense = "min")



%>%
  solve_model(with_ROI(solver = "glpk")) %>%
  get_solution()

  
  
  
  
  
  
  
  