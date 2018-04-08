######

set.seed(1)
x=matrix(rnorm(20*3),ncol=3)
y=c(rep(-1,10),rep(1,10))
x
y
x[y==1,]=x[y==1,]+2
x
plot(x[,2],x[,1],col=(3-y))

####

dat=data.frame(x=x,y=as.factor(y))
View(dat)
##install.packages("e1071")
library(e1071)
svmfit=svm(y~.,data=dat,kernel='linear',cost=10,scale=FALSE)
plot(svmfit,dat,x.1~x.2)
summary(svmfit)

####

svm.fit=svm(y~.,data=dat,kernel="linear",cost=10,scale=FALSE)
plot(svm.fit,dat,x.2~x.1)
summary(svm.fit)

####

set.seed(1)
tune.out=tune(svm,y~.,data=dat,kernel="linear",ranges=list(cost=c(0.001,0.01,0.1,1,5,10,100),gamma=c(0.001,0.01,0.1,1,5,10,100)))
summary(tune.out)
bestmodel= tune.out$best.model
summary(bestmodel)

#####

xtest=matrix(rnorm(60),ncol = 3)
xtest
ytest=sample(c(-1,1),20,rep=TRUE)
ytest
xtest[ytest==1,]=xtest[ytest==1,]+1
testdat=data.frame(x=xtest,y=as.factor(ytest))
testdat

ypred=predict(bestmodel,testdat)
ypred
table(predict=ypred,truth=testdat$y)

##### Radial Kernel

set.seed(1)
x=matrix(rnorm(600),ncol=3)
x[1:100,]=x[1:100,]+2
x[101:150,]=x[101:150,]-2
y=c(rep(1,150),rep(2,50))
dat=data.frame(x=x,y=as.factor(y))
plot(x,col=y)
train=sample(200,100) ## labels 0f train set
svmfit=svm(y~.,data=dat[train,],kernel="radial",gamma=1,cost=1)
plot(svmfit,dat[train,],x.2~x.3)
summary(svmfit)


####
svmfit=svm(y~.,data=dat[train,],kernel="radial",gamma=1,cost=1000)
plot(svmfit,dat[train,],x.1~x.3)

####

tune.out=tune(svm,y~.,data=dat[train,],kernel="radial",
              ranges=list(cost=c(0.1,1,10,100,1000),
                          gamma=c(0.5,1,2,3,4)))
summary(tune.out)
plot(tune.out$best.model,dat[train,],x.2~x.3)
test=-train
table(true=dat[test,"y"],pred=predict(tune.out$best.model,dat[test,]))


#########

set.seed (1)
x
x=rbind(x, matrix(rnorm(50*3), ncol=3))
x
y
y=c(y, rep(0,50))
y
x[y==0,]=x[y==0,]+2
dat=data.frame(x=x, y=as.factor(y))
dat
#par(mfrow=c(1,1))
plot(x,col=(y+1))
svmfit=svm(y~., data=dat, kernel="radial",
           cost=100, gamma=0.01)
plot(svmfit, dat,x.1 ~ x.2)

