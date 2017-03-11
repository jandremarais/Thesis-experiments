library(tensorflow)
library(reticulate)

setwd("Downloads")
main <- py_run_file("temp.py")
X <- main$X
Y <- main$Y

nDCG <- function(r, y, k) {
  # function to calculate the normalized discountedcumulative gain @ k
  # given ground truth y and ranking indices r
  if(k != length(r)) stop("k of ranking does not match k")
  num_rel <- sum(y)
  Ik <- 1/sum(rep(1, min(num_rel, k))/log(1+(1:min(num_rel, k))))
  Ik * sum(y[r]/log(1+(1:k)))
}

rank_op <- function(y, k) {
  # function mimicing the rank operator
  # takes vector y as input and outputs the indices of the k largest elements of y ranked in descending order
  # ties broken randomly
  
  # can actually just use the rank/order function, "-" for decreasing
  #rank(-y, ties.method = "random")[1:k]
  order(y, decreasing = TRUE)[1:k]
}

y_gt <- c(1,1,0,0,0,1,0)
y_pred <- c(0.1,0.2,0.85,0.6,0.9,0.5,0.99)

nDCG(r = rank_op(y_pred, k = 3), y = y_gt, k = 4)

# equation 5
## objectve function

D <- ncol(X)
L <- ncol(Y)

# algo 2, split-node

Ik <- function(k, y) {
  1/sum(1/log(1 + (1:min(k, sum(y)))))
}

node <- list()
node$id <- 1:nrow(X)

Id <- node$id # index of observations in node
delta <- list(sample(c(-1, 1), length(Id), replace = TRUE))
w <- list(rep(0, D))
t <- 1
tw <- 1
W <- list(0)

while(delta[[W[[tw]]]] != delta[[W[[tw - 1]]]]) {
  r_plus <- rank_op(
    k = L,
    
  )
}


  
