################

url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data"
crx <- read.table( file=url, header=FALSE, sep="," )

##write.table( crx, "crx.dat", quote=FALSE, sep="," )
head( crx, 6 )

crx <- crx[ sample( nrow( crx ) ), ]
X <- crx[,1:15]
y <- crx[,16]

trainX <- X[1:600,]
trainy <- y[1:600]
testX <- X[601:690,]
testy <- y[601:690]


##install.packages("C50")
library(C50)
model <- C50::C5.0( trainX, trainy )
summary(model)
plot(model)
model <-  C50::C5.0( trainX, trainy, trials=50 )
p <- predict( model, testX, type="class" )
sum( p == testy ) / length( p )
p <- predict( model, testX, type="prob" )


#####
##install.packages("rpart")
library(rpart)

#######
df=kyphosis
tree=rpart(Kyphosis~.,method = 'class',data=kyphosis)
printcp(tree)
plot(tree,uniform = T,main="my tree")
text(tree,use.n = T,all = T)

#########
##install.packages("rpart.plot")
library(rpart.plot)
prp(tree)

######
##install.packages("randomForest")
library(randomForest)

######
rf.model=randomForest(Kyphosis~.,data=kyphosis,nTrees=500,importance = T)
print(rf.model)
rf.model$confusion
rf.model$importance
#install.packages("ISLR")
#library(ISLR)
#df=College

#####
library(caTools)
sample=sample.split(df$Kyphosis,SplitRatio = 0.70)
train=subset(df,sample==T)
test=subset(df,sample==F)
tree=rpart(Kyphosis~.,method="class",data=train)
tree.predict=predict(tree,test)
head(tree.predict)
tree.predict=as.data.frame(tree.predict)

#########
ranTree=randomForest(Kyphosis~.,data=train,nTrees=500)
ranTree.predict=predict(tree,test)
head(ranTree.predict)
ranTree.predict=as.data.frame((ranTree.predict))
ranTree.predict

#######

joiner <- function(x){
  if (x>=0.5){
    return('Present')
  }else{
    return("Absent")
  }
}
ranTree.predict$Kyphosis_State <- sapply(ranTree.predict$present,joiner)
ranTree.predict

table(ranTree.predict$Kyphosis_State,test$Kyphosis)
