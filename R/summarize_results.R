# setwd("/Users/hannaheglinton/Documents/GitHub/Risk_Model_Research/R")

results <- read.csv("../sim_data/results_ncd_fr_HE.csv")

mean_performance <- results %>%
  group_by(n, p, method) %>%
  summarize(mean_acc = mean(acc), mean_sens = mean(sens), mean_spec = mean(spec),
            mean_auc = mean(auc), mean_time = mean(time), mean_nonz = mean(non.zeros))

perc_best <- results %>%
  filter(method != "LR") %>%
  arrange(desc(auc)) %>%
  group_by(data) %>%
  slice(1) %>%
  group_by(n, p) %>%
  summarize(FR = round(100*mean(method == "FR"),0), 
            NLLCD = round(100*mean(method == "NLLCD"),0),
            Round = round(100*mean(method == "Round"),0)) %>%
  pivot_longer(3:5, names_to = "method", values_to = "perc_best")


summary <- left_join(mean_performance, perc_best) %>%
  select(n, p, method, perc_best, mean_acc, mean_sens, mean_spec, mean_auc,
         mean_time, mean_nonz)


write.csv(summary, "../sim_data/results_summary_0930.csv")
