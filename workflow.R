library(tensorflow)
library(reticulate)

main <- py_run_file("load_tfrecords.py")
X <- main$X
Y <- main$Y

train_ind <- sample(1:nrow(X), 7000)
X_train <- X[train_ind, ]
Y_train <- Y[train_ind, ]
X_test <- X[-train_ind, ]
Y_test <- Y[-train_ind, ]

train_forest <- grow_forest(ntrees = 4, X_train, Y_train, max_leaf = 100)

Y_test_hat <- forest_predict(train_forest, X = X_test, Y = Y_test)

dim(Y_test_hat)
dim(Y_test)

mean(sapply(1:nrow(Y_test), function(a) {
  mean(Y_test[a, rank_op(Y_test_hat[a, ], k = 20)])
}))

mean(sapply(1:nrow(Y_test), function(a) {
  nDCG(rank_op(Y_test_hat[a, ], k = 20), Y_test[a, ], k = 20)
}))
