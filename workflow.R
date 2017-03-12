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
