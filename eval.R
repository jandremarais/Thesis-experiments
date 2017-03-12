solText  <- 
  "id,col1,col2,Indicator
1,1,1,Public
2,2,2,Public
3,3,3,Public
4,4,4,Public
5,5,5,Private
6,6,6,Private"

subText <- "id,col1,col2
1,1.11,1.22
2,2.11,2.22
3,3.11,3.22
4,4.11,4.22
5,7.444,7.333
6,6.444,6.333"

sol  <- read.csv(text = solText)
sub  <- read.csv(text = subText)
sol
sub

CalculateAveragePrecision  <- function(expectedColumn, submittedColumn) {
  df  <- data.frame(expectedBySubmitted = expectedColumn, submitted = submittedColumn)
  #print(df)
  df  <- df[order(df$submitted, decreasing = T),]
  df[, "expectedByExpected"] = sort(expectedColumn, decreasing = T)
  
  totalNumerator = 0.0;
  runningNumeratorExpected = 0.0;
  runningNumeratorActual = 0.0;
  
  #print(df)
  
  for (i in 1:nrow(df)) {
    runningNumeratorExpected = runningNumeratorExpected + df$expectedByExpected[i]
    runningNumeratorActual = runningNumeratorActual + df$expectedBySubmitted[i]
    division = runningNumeratorActual/runningNumeratorExpected;
    totalNumerator = totalNumerator + division;
  }
  result = totalNumerator / nrow(df)
  result
}


CalculateAveragePrecision(sol$col1, sub$col1)
CalculateAveragePrecision(Y_test_hat[, 2], Y_test[, 2])

MCAP  <- function(sol, sub, predictionColumns) {
  apply(matrix(predictionColumns), 1, function(col) CalculateAveragePrecision(sol,sub))
}

MCAP(sol, sub, 2:3)

MCAP(Y_test, Y_test_hat, 1:ncol(Y))

mean(sapply(1:20, function(a) CalculateAveragePrecision(Y_test_hat[, a], Y_test[, a])))

temp_sys <- system("gsutil ls gs://us.data.yt8m.org/1/video_level/test")

26*2 * (26 + 10) *2

