data <- as.data.frame(data_breast$data)

X <- data_breast$data$X

Y <- data_breast[2]
variable_names <- data_breast[3]
outcome_name <- data_breast[4]
sample_weights <- sample_weights[5]

# set Y
Y_col_idx <- 1
Y <- data[, Y_col_idx, drop=FALSE]
data_headers <- list(colnames(data))
Y_name <- data_headers[[Y_col_idx]][1]
Y[Y == 0] <- 0

#set X
X_features <- list()
# set up x
for (j in(colnames(data)))
{
  if ( (j != Y_col_idx))
  {
    X_features <- append(X_features,j)
  }
}
X_features <- X_features[-1]
X <- data[unlist(X_features)]
variable_names <- X_features

#DEL: set intercept
#intercept <- rep(1,nrow(X))
#X <- as.data.frame(cbind(intercept,X))
#variable_names <- append('intercept', variable_names)

# set weight file
sample_weights <-rep(1,nrow(X))

# concate X, Y, weight
data <- cbind(X,Y,sample_weights)
