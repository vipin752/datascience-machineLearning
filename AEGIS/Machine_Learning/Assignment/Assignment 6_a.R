setwd(getwd())
mushroomroom<-read.csv("mushroomroomrooms.csv",stringsAsFactors = T)
View(mushroomroom)
summary(mushroomroom)
str(mushroomroom)
sum(is.na(mushroomroom))
dim(mushroomroom)

# b)Use Appropriate method to find out the Significant variables.

# install.packages("devtools")
library(devtools)
# install_github("riv","tomasgreif")
library(woe)

iv1<-iv.mult(mushroomroom,"class",TRUE)
iv.plot.summary(iv1)

mushroomroom_log<-glm(class~odor+spore.print.color+stalk.shape+bruises+
                gill.spacing+gill.size,
              data = mushroomroom,family = "binomial")
summary(mushroomroom_log)

library(randomForest)
mushroomroom_rf <-randomForest(class~.,data=mushroomroom,ntree=1500,importance = TRUE)
summary(mushroomroom_rf)

importance(mushroomroom_rf,type=1)
importance(mushroomroom_rf,type=2)
varImpPlot(mushroomroom_rf,sort=T,n.var=10)


# c)Divide the dataset into Development and Validation Samples.


mushroomroom1<-mushroomroom[,!(names(mushroomroom) %in% c("gill.attachment","veil.type","veil.color","stalk.root"))]
View(mushroomroom1)
dim(mushroomroom1)
mushroomroom_hot <-mushroomroom1
dim(mushroomroom_hot)

#pint All independent category variables converted from ordinal to one hot encoding 
install.packages("dummies")
library(dummies)

mushroomroom_hot<-dummy.data.frame(mushroomroom_hot[,-1])

mushroomroom_hot<-cbind(class=mushroomroom1$class,mushroomroom_hot)
View(mushroomroom_hot)
dim(mushroomroom_hot)

# Taking 25% development set and 75% validation set

library(caTools)
set.seed(56667)
split = sample.split(mushroomroom_hot$class, SplitRatio = 0.25)
mushroom_dev = subset(mushroom_hot, split == TRUE)
mushroom_val = subset(mushroom_hot, split == FALSE)


# d)Build SVM Model using linear Kernel and check Accuracy using Validation samples.

 install.packages("e1071")
library(e1071)
mushroom_svm1=svm(class~.,data=mushroom_dev,kernel='linear',scale=FALSE)
summary(mushroom_svm1)

mushroom_pred1 <- predict(mushroom_svm1,mushroom_val[,-1])
table(predict=mushroom_pred1,actual=mushroom_val$class)


# e)Build SVM Model using Radial Basis Kernel and check Accuracy 
# using Validation samples, tune the model for best Result.

mushroom_svm2=svm(class~.,data=mushroom_dev,kernel="radial")
summary(mushroom_svm2)

mushroom_pred2 <- predict(mushroom_svm2,mushroom_val[,-1])
table(predict=mushroom_pred2,actual=mushroom_val$class)


set.seed(123)
tune.out_2=tune(svm,class~.,data=mushroom_dev,kernel="radial",
               ranges=list(cost=c(0.1,1,10,100,1000),
                           gamma=c(0.1,1,2,3,4)))
bestmodel2=tune.out_2$best.model
summary(bestmodel2)

mushroom_pred2a<-predict(bestmodel2,mushroom_val)
table(predict=mushroom_pred2a,actual=mushroom_val$class)


nb_classifier1 <- naiveBayes(mushroom_dev[,-1],mushroom_dev[,1])
nb_predict1 <-predict(nb_classifier1,mushroom_val)

table(predict=nb_predict1,actual=mushroom_val$class)
