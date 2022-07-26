library(tidyverse)
set.seed(1)
sim_logistic_data <- function(sample_size = 10, beta_0 = 1, beta_1 = -4, beta_2 = -3){
  x1 = rnorm(n = sample_size)
  x2 = rnorm(n = sample_size)
  eta = beta_0 + beta_1 * x1 + beta_2 * x2
  p = 1 / (1 + exp(-eta))
  y = rbinom(n = sample_size, size = 1, prob = p)
  data.frame(y, x1, x2)
}

example_data <- sim_logistic_data(sample_size = 10, beta_0 = 1, beta_1 = -4, beta_2 = -3)
fit_glm <- glm(y ~ x1+x2, data = example_data, family = binomial)
