## This script is the function for get the true beta from simulated coef_csv from dat_sim script 

library(dplyr)

get_T_beta <- function(data_name,my_path){
  
  # Files in path
  files <- list.files(my_path)
  
  # get coef data file name
  coef_dat_name <- gsub("_data.csv", "", data_name)
  coef_dat_name <- paste0(coef_dat_name,"_coef.csv")
  
  # get coef data file
  coef_dat <- read_csv( paste0(my_path,'/',coef_dat_name) )
  #coef_dat$names <- c("Intercept","V1","V2","V3","V4","V5","V6")
  
  # return true beta
  return(coef_dat$vals[-1])
}