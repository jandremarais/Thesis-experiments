library(tensorflow)
library(reticulate)
library(LiblineaR)

# create text files of list of filenames
system("gsutil ls gs://youtube8m-ml/1/video_level/test > video_test_files.txt")
system("gsutil ls gs://youtube8m-ml/1/video_level/train > video_train_files.txt")
system("gsutil ls gs://youtube8m-ml/1/video_level/validate > video_validate_files.txt")

#video_test_files <- read.table("video_test_files.txt", stringsAsFactors = FALSE)
#video_train_files <- read.table("video_train_files.txt", stringsAsFactors = FALSE)
#video_validate_files <- read.table("video_validate_files.txt", stringsAsFactors = FALSE)

# training step
train_files <- read.table("video_train_files.txt", stringsAsFactors = FALSE)
write.table(train_files[2:30, ], "train_to_imp.txt", row.names = FALSE, col.names = FALSE, quote = FALSE)

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

train_forest <- grow_forest(ntrees = 5, X_train, Y_train, max_leaf = 100, par = FALSE)
saveRDS(train_forest, "forest2-30_100.rds")
train_forest <- readRDS("train_forest.rds")

Y_hat_train <- forest_predict(train_forest, X = X_train, par = FALSE)

library(feather)
write_feather(as.data.frame(Y_hat_train), 'yhat.feather')
write_feather(as.data.frame(Y_train), 'y.feather')

#write.table(Y_hat_train, "yhat.csv", sep = ",", row.names = FALSE, col.names = FALSE)
#write.table(Y_train, "y.csv", sep = ",", row.names = FALSE, col.names = FALSE)

py_run_file("gap.py")$value

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

py_run_file("gap.py")$value

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
