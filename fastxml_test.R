library(LiblineaR)
library(tensorflow)
library(reticulate)

main <- py_run_file("load_tfrecords.py")
X <- main$X
Y <- main$Y

rank_op <- function(y, k) {
  # function mimicing the rank operator
  # takes vector y as input and outputs the indices of the k largest elements of y ranked in descending order
  # ties broken randomly
  
  # can actually just use the rank/order function, "-" for decreasing
  #rank(-y, ties.method = "random")[1:k]
  order(y, sample(1:length(y)), decreasing = TRUE)[1:k]
}

# equation 5
## objectve function

D <- ncol(X)
L <- ncol(Y)

# algo 2, split-node

Ik <- function(k, y) {
  1/sum(1/log(1 + (1:min(k, sum(y)))))
}

nDCG <- function(r, y, k) {
  # function to calculate the normalized discountedcumulative gain @ k
  # given ground truth y and ranking indices r
  if(k != length(r)) stop("k of ranking does not match k")
  num_rel <- sum(y)
  Ik(k, y) * sum(y[r]/log(1+(1:k)))
}

# function taking indices of observation in node and outputs the indices of the positive and negative partitions
split_node <- function(id, X, Y) {
  X <- X[id, ]
  Y <- Y[id, ]
  delta <- list(sample(c(-1, 1), length(id), replace = TRUE))
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
  
  list(negative = id[delta[[t]] == -1], 
       positive = id[delta[[t]] == 1],
       separator = w[[t]])
}


library(data.tree)

grow_tree <- function(X, Y, max_leaf = 100) {
  fxml_tree <- Node$new("initial", id = 1:nrow(X))
  done <- FALSE
  while(!done) {
    fxml_tree$Do(function(node) {
      if(length(node$id) >= max_leaf) {
        temp <- split_node(node$id, X = X, Y = Y)
        node$w <- temp$separator
        node$AddChild("neg", id = temp$negative)
        node$AddChild("pos", id = temp$positive)
      }
    }, filterFun = isLeaf)
    
    done <- all(sapply(fxml_tree$Get("id", filterFun = isLeaf), length) < max_leaf)
  }
  
  fxml_tree$Do(function(node) {
    node$P <- apply(Y[node$id, ], 2, sum)/length(node$id)
  }, filterFun = isLeaf)
  
  leaf_nodes <- fxml_tree$Do(function(node) node, filterFun = isLeaf)
  
  Y_pred <- matrix(NA, nrow = nrow(Y), ncol = ncol(Y))
  
  # can probably make this faster
  for(i in 1:length(leaf_nodes)) {
    Y_pred[leaf_nodes[[i]]$id, ] <- rep(leaf_nodes[[i]]$P, each = length(leaf_nodes[[i]]$id))
  }
  
  list(tree = fxml_tree, predictions = Y_pred)
}

#temp <- sample(1:nrow(X), 1000)
#temp_tree <- grow_tree(X = X[temp, ], Y = Y[temp, ], max_leaf = 100)
#temp_tree$predictions[1:5, 1:5]

tree_predict <- function(X, Y, tree) {
  pred_tree <- Clone(tree)
  pred_tree$Do(function(node) node$RemoveAttribute("id"))
  pred_tree$root$Do(function(node) node$id <- 1:nrow(X), filterFun = isRoot)
  
  pred_tree$Do(function(node) {
    if(!is.null(node$w)) {
      part_score <- t(node$w) %*% t(X[node$id,])
      node$pos$id <- node$id[part_score > 0]
      node$neg$id <- node$id[part_score <= 0]
    }
  })
  
  leaf_nodes <- pred_tree$Do(function(node) node, filterFun = isLeaf)
  
  Y_pred <- matrix(NA, nrow = nrow(Y), ncol = ncol(Y))
  
  # can probably make this faster
  for(i in 1:length(leaf_nodes)) {
    Y_pred[leaf_nodes[[i]]$id, ] <- rep(leaf_nodes[[i]]$P, each = length(leaf_nodes[[i]]$id))
  }
  list(predictions = Y_pred)
}

#tree_predict(X[temp, ], Y[temp, ], temp_tree$tree)$predictions[1:5, 1:5]
#temp_tree$predictions[1:5, 1:5]

library(parallel)
nCores <- detectCores()

grow_forest <- function(ntrees, X, Y, max_leaf) {
  require(parallel)
  nCores <- detectCores()
  trees <- mclapply(1:ntrees, function(a) {
    grow_tree(X, Y, max_leaf = max_leaf)
  }, mc.cores = nCores)
}

#temp_forest <- grow_forest(ntrees = 4, X[temp, ], Y[temp, ], max_leaf = 100)

forest_predict <- function(forest, X, Y) {
  require(parallel)
  nCores <- detectCores()
  tree_preds <- mclapply(forest, function(a) {
    tree_predict(X, Y, tree = a$tree)$predictions
  }, mc.cores = nCores)
  Reduce("+", tree_preds)/length(tree_preds)
}

#temp_pred <- forest_predict(temp_forest, X[temp, ], Y[temp, ])
#temp_pred[1:5, 1:5]

#mean(sapply(1:nrow(Y), function(a) mean(Y[a, rank_op(Y_pred[a, ], k = 20)])))

#mean(sapply(1:nrow(Y), function(a) nDCG(rank_op(Y_pred[a, ], k = 20), Y[a, ], k = 20)))

