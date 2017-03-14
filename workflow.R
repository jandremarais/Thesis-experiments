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
write.table(train_files[2:5, ], "train_to_imp.txt", row.names = FALSE, col.names = FALSE, quote = FALSE)

main <- py_run_file("load_tfrecords.py")
X_train <- main$X
Y_train <- main$Y

train_forest <- grow_forest(ntrees = 1, X_train, Y_train, max_leaf = 10)
saveRDS(train_forest, "train_forest.rds")
train_forest <- readRDS("train_forest.rds")

Y_hat_train_ <- forest_predict(train_forest, X = X_train)

write.table(Y_hat_train, "yhat.csv", sep = ",", row.names = FALSE, col.names = FALSE)
write.table(Y_train, "y.csv", sep = ",", row.names = FALSE, col.names = FALSE)

py_run_file("gap.py")$value

# validation step
validate_files <- read.table("video_validate_files.txt", stringsAsFactors = FALSE)
write.table(validate_files[2:5, ], "train_to_imp.txt", row.names = FALSE, col.names = FALSE, quote = FALSE)

main <- py_run_file("load_tfrecords.py")
X_validate <- main$X
Y_validate <- main$Y

Y_hat_validate <- forest_predict(train_forest, X = X_validate)

write.table(Y_hat_validate, "yhat.csv", sep = ",", row.names = FALSE, col.names = FALSE)
write.table(Y_validate, "y.csv", sep = ",", row.names = FALSE, col.names = FALSE)

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
