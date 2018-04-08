rm(list=ls())

######
data=scale(iris[1:4])
data=cbind(data,iris[5])

####
set.seed(101)
library(caTools)
sample=sample.split(data$Species,SplitRatio = 0.7)
train=subset(data,sample== T)
test=subset(data,sample==F)

######
library(class)
predicted_result=knn(train[1:4],test[1:4],train$Species,k=5)
err=mean(test$Species != predicted_result)
err

######
predicted_result=NULL
err=NULL
for(i in 1:10){
  set.seed(101)
  predicted_result=knn(train[1:4],test[1:4],train$Species,k=i)
  err[i]=mean(test$Species != predicted_result)
  
}
err_rate=data.frame(err,x=1:10)
plot(err_rate$x,err_rate$err)

########
library(ggplot2)
ggplot(err_rate,aes(x=1:10,y=err))+geom_point()+geom_line(lty="dotted")
