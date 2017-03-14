library(tensorflow)
library(reticulate)
library(LiblineaR)

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

saveRDS(train_forest, "train_forest.rds")
train_forest <- readRDS("train_forest.rds")

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
