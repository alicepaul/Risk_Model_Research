library(tidyverse)
set.seed(1)
sim_logistic_data <- function(sample_size = 10, beta_0 = 1, beta_1 = -4){
  x = rnorm(n = sample_size)
  eta = beta_0 + beta_1 * x
  p = 1 / (1 + exp(-eta))
  y = rbinom(n = sample_size, size = 1, prob = p)
  data.frame(y, x)
}

example_data <- sim_logistic_data(sample_size = 10, beta_0 = 1, beta_1 = -4)
fit_glm <- glm(y ~ x, data = example_data, family = binomial)
