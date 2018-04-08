#install.packages("JOUSBoost")
library(JOUSBoost)

# Generate data from the circle model
set.seed(111)
dat = circle_data(n = 500)
train_index = sample(1:500, 400)

ada = adaboost(dat$X[train_index,], dat$y[train_index], tree_depth = 2,
               n_rounds = 200, verbose = TRUE)
print(ada)
yhat_ada = predict(ada, dat$X[-train_index,])

# calculate misclassification rate
mean(dat$y[-train_index] != yhat_ada)
