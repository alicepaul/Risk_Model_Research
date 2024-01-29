library(ompr)
library(ompr.roi)
library(ROI.plugin.glpk) # Example solver
library(tidyverse)

# Load your data
files <- list.files("/Users/oscar/Documents/GitHub/Risk_Model_Research/MILP/data")
df <- read.csv(paste0("/Users/oscar/Documents/GitHub/Risk_Model_Research/MILP/data/",files[10]))

# Preprocess the data
df <- df[1:20, 1:5]
y <- df[, 1]
X <- as.matrix(df[, -1])

# Parameters
n <- nrow(X)
p <- ncol(X)
M <- 1000
SK_pool <- seq(-5 * p, 5 * p, 1)
PI <- seq(0.01, 0.99, length.out = 100)

# Assuming X, y, n, p, M, SK_pool, and PI are already defined

model <- MIPModel() %>%
  # Variable definitions
  add_variable(beta[j], j = 1:p, type = "integer", lb = -5, ub = 5) %>%
  add_variable(s[i], i = 1:n, type = "integer", lb = -10, ub = 10) %>%
  add_variable(z_ik[i, k], i = 1:n, k = 1:length(SK_pool), type = "binary") %>%
  add_variable(p_ik[i, k], i = 1:n, k = 1:length(SK_pool), type = "continuous", lb = 0.0001, ub = 0.9999) %>%
  
  # Constraints
  # Score constraint
  add_constraint(sum_expr(beta[j] * X[i, j], j = 1:p) == s[i], i = 1:n) %>%
  
  # Sum of z_ik for each i
  add_constraint(sum_expr(z_ik[i, k], k = 1:length(SK_pool)) == 1, i = 1:n) %>%
  
  # Linking s[i] to z_ik
  add_constraint(s[i] - SK_pool[k] - M * (1 - z_ik[i, k]) <= 0, i = 1:n, k = 1:length(SK_pool)) %>%
  add_constraint(s[i] - SK_pool[k] + M * (1 - z_ik[i, k]) >= 0, i = 1:n, k = 1:length(SK_pool)) %>%
  
  # Linking p_ik to z_ik
  add_constraint(p_ik[i, k] - PI[l] <= M * (1 - z_ik[i, k]), i = 1:n, k = 1:length(SK_pool), l = 1:length(PI)) %>%
  add_constraint(p_ik[i, k] - PI[l] >= -M * (1 - z_ik[i, k]), i = 1:n, k = 1:length(SK_pool), l = 1:length(PI)) %>%
  
  # Objective function (define as per your requirement)
  set_objective(-sum_expr(y[i] * log(p_ik[i, k]) - (1 - y[i]) * log(1 - p_ik[i, k]), i = 1:n, k = 1:length(SK_pool), "min"))

# Solve the model
result <- solve_model(model, with_ROI(solver = "glpk", verbose = TRUE))

# Extracting solutions (adjust as necessary)
solution_beta <- get_solution(result, beta[j], j = 1:p)
solution_s <- get_solution(result, s[i], i = 1:n)
# ... (Extract other variables as needed)
