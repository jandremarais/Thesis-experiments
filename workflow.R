library(tensorflow)
library(reticulate)
library(LiblineaR)
library(feather)

# create text files of list of filenames
system("gsutil ls gs://youtube8m-ml/1/video_level/test > video_test_files.txt")
system("gsutil ls gs://youtube8m-ml/1/video_level/train > video_train_files.txt")
system("gsutil ls gs://youtube8m-ml/1/video_level/validate > video_validate_files.txt")

# training step
train_files <- read.table("video_train_files.txt", stringsAsFactors = FALSE)
write.table(train_files[2:10, ], "train_to_imp.txt", row.names = FALSE, col.names = FALSE, quote = FALSE)

list2ind <- function(y, L = 4716) {
  Y <- matrix(0, ncol = L, nrow = length(y))
  for(i in 1:length(y)) {
    Y[i, y[[i]]+1] <- 1
  }
  Y
}

main <- py_run_file("load_tfrecords.py")
X_train <- main$X
#Y_train <- main$Y
y <- main$y
y <- lapply(y, unlist)
Y_train <- list2ind(y = y)

train_forest <- grow_forest(ntrees = 3, X_train, Y_train, max_leaf = 50, par = FALSE)
saveRDS(train_forest, "forest2-10_50.rds")
train_forest <- readRDS("train_forest.rds")

Y_hat_train <- forest_predict(train_forest, X = X_train, par = FALSE)

write_feather(as.data.frame(Y_hat_train), 'yhat.feather')
write_feather(as.data.frame(Y_train), 'y.feather')

py_run_file("gap.py")$value

my_gap <- function(pred, act, k = 20) {
  ord_mat <- t(apply(pred, 1, function(a) rank_op(a, k)))
  
  pred <- t(sapply(1:nrow(ord_mat), function(a) pred[a, ord_mat[a, ]]))
  act <- t(sapply(1:nrow(ord_mat), function(a) act[a, ord_mat[a, ]]))
  print(list(pred, act))
  
  ord_ind <- order(pred, decreasing = TRUE)
  print(ord_ind)
  sum((cumsum(act[ord_ind])/(1:length(ord_ind)))[act[ord_ind] == 1])/sum(act)
}

my_gap(Y_hat_train, Y_train)

# validation step
validate_files <- read.table("video_validate_files.txt", stringsAsFactors = FALSE)
write.table(validate_files[2:10, ], "train_to_imp.txt", row.names = FALSE, col.names = FALSE, quote = FALSE)

main <- py_run_file("load_tfrecords.py")
X_validate <- main$X
#Y_validate <- main$Y
y <- main$y
y <- lapply(y, unlist)
Y_validate <- list2ind(y = y)

Y_hat_validate <- forest_predict(train_forest[c(1,2,5)], X = X_validate)

#write.table(Y_hat_validate, "yhat.csv", sep = ",", row.names = FALSE, col.names = FALSE)
#write.table(Y_validate, "y.csv", sep = ",", row.names = FALSE, col.names = FALSE)

write_feather(as.data.frame(Y_hat_validate), 'yhat.feather')
write_feather(as.data.frame(Y_validate), 'y.feather')

Y_hat_validate <- read_feather('yhat.feather')
Y_validate <- read_feather('y.feather')

py_run_file("gap.py")$value

my_gap(Y_hat_validate, Y_validate)

# testing step

kaggle_sub <- NULL
files <- read.table("video_test_files.txt", stringsAsFactors = FALSE)
for(i in 2:nrow(files)) {
  write.table(files[i, ], file = "to_imp.txt", row.names = FALSE, col.names = FALSE, quote = FALSE)
  
  main <- py_run_file("import_test.py")
  X_test <- main$X
  
  Y_test_hat <- forest_predict(train_forest, X = X_test)
  
  lcp <- apply(Y_test_hat, 1, function(a) {
    ind <- rank_op(k = 20, a)
    paste(ind, round(a[ind], 6), collapse = " ")
  })
  
  kaggle_temp <- data.frame(VideoId = main$vid_ids, LabelConfidencePairs = lcp)
  
  kaggle_sub <- rbind(kaggle_sub, kaggle_temp)
  print(i)
}

write.csv(kaggle_sub, "fxml_sub.csv", quote = FALSE, row.names = FALSE)


mean(sapply(1:nrow(Y_test), function(a) {
  mean(Y_test[a, rank_op(Y_test_hat[a, ], k = 20)])
}))

mean(sapply(1:nrow(Y_test), function(a) {
  nDCG(rank_op(Y_test_hat[a, ], k = 20), Y_test[a, ], k = 20)
}))
