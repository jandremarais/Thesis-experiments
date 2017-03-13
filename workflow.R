library(tensorflow)
library(reticulate)

#system("gsutil ls gs://us.data.yt8m.org/1/video_level/test > video_test_files.txt")
#system("gsutil ls gs://us.data.yt8m.org/1/video_level/train > video_train_files.txt")
#system("gsutil ls gs://us.data.yt8m.org/1/video_level/validate > video_validate_files.txt")

system("gsutil ls gs://youtube8m-ml/1/video_level/test > video_test_files.txt")
system("gsutil ls gs://youtube8m-ml/1/video_level/train > video_train_files.txt")
system("gsutil ls gs://youtube8m-ml/1/video_level/validate > video_validate_files.txt")

#video_test_files <- read.table("video_test_files.txt", stringsAsFactors = FALSE)
#video_train_files <- read.table("video_train_files.txt", stringsAsFactors = FALSE)
#video_validate_files <- read.table("video_validate_files.txt", stringsAsFactors = FALSE)

main <- py_run_file("load_tfrecords.py")
X <- main$X
Y <- main$Y

train_ind <- sample(1:nrow(X), 2000)
X_train <- X[train_ind, ]
Y_train <- Y[train_ind, ]
X_test <- X[-train_ind, ]
Y_test <- Y[-train_ind, ]

train_forest <- grow_forest(ntrees = 6, X, Y, max_leaf = 500)

files <- read.table("video_test_files.txt", stringsAsFactors = FALSE)
write.table(files[2, ], file = "to_imp.txt", row.names = FALSE, col.names = FALSE, quote = FALSE)

main <- py_run_file("import_test.py")
X_test <- main$X

Y_test_hat <- forest_predict(train_forest, X = X_test, Y = Y_test)

dim(Y_test_hat)
dim(Y_test)

mean(sapply(1:nrow(Y_test), function(a) {
  mean(Y_test[a, rank_op(Y_test_hat[a, ], k = 20)])
}))

mean(sapply(1:nrow(Y_test), function(a) {
  nDCG(rank_op(Y_test_hat[a, ], k = 20), Y_test[a, ], k = 20)
}))
