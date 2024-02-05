if (!identical(uni_s,uni_s_test)){
  diff <- c()
  for (vals in uni_s_test) {
    if (!(vals %in% uni_s)) {
      diff <- c(diff,vals)
    }
  }
  # update v_list
  v_list <- rbind(v_list,
                  cbind(diff,rep(0,times=length(diff))))
}