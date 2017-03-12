Sys.setenv("GCS_AUTH_FILE" = "/home/rstudio/Youtube-Kaggle-056b5e550d89.json")
library(googleCloudStorageR)

gcs_auth()

gcs_get_bucket("maraisjandre9_yt8m_train_bucket")

bucket_info <- gcs_get_bucket("youtube8m-ml")

gcs_get_bucket("us.data.yt8m.org/1/")

gcs_list_buckets("youtube-kaggle-158911")

objects <- gcs_list_objects("us.data.yt8m.org", detail = "summary")
nrow(objects)
objects[1000, ]
?gce