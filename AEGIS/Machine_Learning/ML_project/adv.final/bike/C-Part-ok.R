
#C-TREE:

setwd("C:/Users/Owner/Desktop/ML-Final projects-Advance/Bike data")
train=read.csv("train.csv")
test=read.csv("test.csv")
#test$registered=0
#test$casual=0
#test$count=0
#train <- read.csv("../input/train.csv")
#test <- read.csv("../input/test.csv")
str(train)
train_factor <- train
train_factor$weather <- factor(train$weather)
train_factor$holiday <- factor(train$holiday)
train_factor$workingday <- factor(train$workingday)
train_factor$season <- factor(train$season)
test_factor <- test
test_factor$weather <- factor(test$weather)
test_factor$holiday <- factor(test$holiday)
test_factor$workingday <- factor(test$workingday)
test_factor$season <- factor(test$season)

train_factor$time <- substring(train$datetime,12,20)
test_factor$time <- substring(test$datetime,12,20)
train_factor$time <- factor(train_factor$time)
test_factor$time <- factor(test_factor$time)
train_factor$day <- weekdays(as.Date(train_factor$datetime))
train_factor$day <- as.factor(train_factor$day)
test_factor$day <- weekdays(as.Date(test_factor$datetime))
test_factor$day <- as.factor(test_factor$day)
aggregate(train_factor[,"count"],list(train_factor$day),mean)
train_factor$sunday[train_factor$day == "Sunday"] <- "1"
train_factor$sunday[train_factor$day != "1"] <- "0"
test_factor$sunday[test_factor$day == "Sunday"] <- "1"
test_factor$sunday[test_factor$day != "1"] <- "0"
train_factor$sunday <- as.factor(train_factor$sunday)
test_factor$sunday <- as.factor(test_factor$sunday)
train_factor$hour<- as.numeric(substr(train_factor$time,1,2))
test_factor$hour<- as.numeric(substr(test_factor$time,1,2))
train_factor$daypart <- "4"
test_factor$daypart <- "4"
train_factor$daypart[(train_factor$hour < 10) & (train_factor$hour > 3)] <- 
  
  1

test_factor$daypart[(test_factor$hour < 10) & (test_factor$hour > 3)] <- 1
train_factor$daypart[(train_factor$hour < 16) & (train_factor$hour > 9)] <- 
  
  2
test_factor$daypart[(test_factor$hour < 16) & (test_factor$hour > 9)] <- 2
train_factor$daypart[(train_factor$hour < 22) & (train_factor$hour > 15)] <- 
  
  3
test_factor$daypart[(test_factor$hour < 22) & (test_factor$hour > 15)] <- 3
train_factor$daypart <- as.factor(train_factor$daypart)
test_factor$daypart <- as.factor(test_factor$daypart)
train_factor$hour <- as.factor(train_factor$hour)
test_factor$hour <- as.factor(test_factor$hour)
library('party')
library(rpart.plot)
library(rattle)
library(rpart)
formula <- count ~ season + holiday + workingday + weather + temp + hour + 
  
  daypart + sunday + sunday
fit.ctree <- ctree(formula, data=train_factor)
predict.ctree <- predict(fit.ctree, test_factor)
submit.ctree <- data.frame(datetime = test$datetime, count=predict.ctree)
plot(predict.ctree, type="simple")
write.csv(submit.ctree, file="submit_ctree_v1.csv",row.names=FALSE)
