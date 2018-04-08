#TODO: Show impact of an outlier on tree growth

setwd("D:/content/Aegis/MachineLearning/src")
playTennis<-read.table("playtennis.csv",header=T,sep=",")

require("C50")
head(playTennis)

#If there is any ordering in the data then shuffle the dataset
playTennisRand <- playTennis[order(runif(nrow(playTennis))),]

#Features (attributes)
X <- playTennisRand[,-c(1,6)]
#Target variable
y <- playTennisRand[,6]
dt1 <- C5.0(X, y)

#No. of leaf nodes = tree size in the output
dt1

#Decision tree: as a disjunction of conjunctions
#With accuracy on training data
summary(dt1)

#leaf color, one for each class
plot(dt1)

