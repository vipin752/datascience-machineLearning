#######
library(ISLR)
head(College)
df<-College

######
library(ggplot2)
ggplot(df,aes(Room.Board,Grad.Rate)) + geom_point(aes(color=Private))

#####
ggplot(df,aes(F.Undergrad)) + geom_histogram(aes(fill=Private),color='black',bins=50)

#####
ggplot(df,aes(Grad.Rate)) + geom_histogram(aes(fill=Private),color='black',bins=50)

#####
subset(df,Grad.Rate > 100)
df['Cazenovia College','Grad.Rate'] <- 100

#####
library(caTools)

set.seed(101) 

sample = sample.split(df$Private, SplitRatio = .70)
train = subset(df, sample == TRUE)
test = subset(df, sample == FALSE)

#####
library(rpart)
tree <- rpart(Private ~.,method='class',data = train)

#####
tree.preds <- predict(tree,test)

#####
head(tree.preds)

#####
tree.preds <- as.data.frame(tree.preds)
# Lots of ways to do this
joiner <- function(x){
  if (x>=0.5){
    return('Yes')
  }else{
    return("No")
  }
}

tree.preds$Private <- sapply(tree.preds$Yes,joiner)

head(tree.preds)
#####
table(tree.preds$Private,test$Private)

#####
library(rpart.plot)
prp(tree)

#####
library(randomForest)
rf.model <- randomForest(Private ~ .,data = train,importance = TRUE)

#####
rf.model$confusion

#####
rf.model$importance

#####
p <- predict(rf.model,test)
table(p,test$Private)
