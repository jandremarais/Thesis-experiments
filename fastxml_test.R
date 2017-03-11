library(tensorflow)
library(reticulate)
library(LiblineaR)

main <- py_run_file("load_tfrecords.py")
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
  order(y, sample(1:length(y)), decreasing = TRUE)[1:k]
}

y_pred[rank_op(y_pred, k = 3)]
rank_op(y_pred, k = 3)

y_gt <- c(1,1,0,0,0,1,0)
y_pred <- c(0.1,0.2,0.85,0.85,0.9,0.5,0.99)

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
W <- list(1)

C_plus <- 1
C_minus <- 1
C_rank <- 1

r_plus <- list()
r_minus <- list()

while(if(tw > 1) any(delta[[W[[tw]]]] != delta[[W[[tw - 1]]]]) else TRUE) {
  r_plus[[t]] <- rank_op(
    k = L,
    y = apply(t(apply(Y, 1, function(a) Ik(L, a) * a))[delta[[t]] == 1, ], 2, sum)
  )
  r_minus[[t]] <- rank_op(
    k = L,
    y = apply(t(apply(Y, 1, function(a) Ik(L, a) * a))[delta[[t]] == -1, ], 2, sum)
  )
  
  v_plus <- C_plus * log(1 + exp(-c(t(w[[t]]) %*% t(X)))) -
    C_rank * apply(Y, 1, Ik, k = L) * apply(Y, 1, function(a) sum(a[r_plus[[t]]]/log(1 + (1:L))))
  v_minus <- C_minus * log(1 + exp(c(t(w[[t]]) %*% t(X)))) -
    C_rank * apply(Y, 1, Ik, k = L) * apply(Y, 1, function(a) sum(a[r_minus[[t]]]/log(1 + (1:L))))
  
  delta[[t + 1]] <- delta[[t]]
  delta[[t + 1]][v_plus != v_minus] <- sign(v_minus - v_plus)[v_plus != v_minus]
  
  if(all(delta[[t + 1]] == delta[[t]])) {
    w[[t + 1]] <- c(LiblineaR(data = X, target = delta[[t]], type = 6, cost = 1, bias = 0)$W)
    W[[tw + 1]] <- t
    tw <- tw + 1
  } else {
    w[[t + 1]] <- w[[t]]
  }
  
  t <- t + 1
}


  
