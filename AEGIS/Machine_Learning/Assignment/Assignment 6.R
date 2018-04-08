#1
bank <- read.csv(file.choose(),sep = ";")
head(bank)
View(bank)
str(bank)
summary(bank)


# install.packages("e1071")
library(e1071)

#2.

loging_bank <- glm(y~.,data = bank,family = binomial)
summary(loging_bank)


install.packages("Boruta")
library(Boruta)
set.seed(123)
boruta_bank <- Boruta(y~.,data = bank,doTrace=2)
print(boruta_bank)
plot(boruta_bank)

boruta_bank$finalDecision

final_boruta_bank <-TentativeRoughFix(boruta_bank)
print(final_boruta_bank)
plot(final_boruta_bank)

getSelectedAttributes(final_boruta_bank, withTentative = F)

#library(randomForest)
rf_bank <-randomForest(y~.,data=bank,ntree=1500,importance = TRUE)
summary(rf_bank)

importance(rf_bank,type=1)
print(rf_bank$importance)
varImpPlot(rf_bank)

# install.packages("caTools")
library(caTools)
bank_2<-bank[,-c(2,5,6)]
View(bank_2)
str(bank_2)

# scale the numeric variables
sc<-TRUE
if (sc)
{
  for (colnames in names(bank_2))
  {
    if(class(bank_2[,colnames])=="integer" | class(bank_2[,colnames])=="numeric")
    {
      bank_2[,colnames] <- scale(bank_2[,colnames])
    }
  }
}

View(bank_2)


set.seed(12345)
split1 = sample.split(bank_2$y, SplitRatio = 0.75)
bank_dev_1 = subset(bank_2, split1 == TRUE)
bank_val_1 = subset(bank_2, split1 == FALSE)


#4.

svm_1a=svm(y~.,data=bank_dev1,kernel='linear')
summary(svm_1a)
plot(svm_1a,data = bank_dev1,formula = age~duration)
predict <- predict(svm_1a,bank_val1)
t1 = table(predict=predict,actual=bank_val1$y)
t1

sum(t1[1]+t1[4])/sum(t1)

#5.

svm_2a=svm(y~.,data=bank_dev1,kernel="radial",gamma=0.1,cost=1)
summary(svm_2a)
plot(svm_2a,data = bank_dev1,formula = age~duration)
predict_2a <- predict(svm_2a,bank_val1)
t2 = table(predict=pred2a,actual=bank_val1$y)
t2

sum(t2[1]+t2[4])/sum(t2) 
set.seed(123)
tune.out_1a=tune(svm,y~.,data=bank_dev1,kernel="radial",
                ranges=list(cost=c(0.1,1,5,100,1000),
                            gamma=c(0.01,0.05,1,2,3,4)))
bestmodel1a=tune.out_1a$best.model
summary(bestmodel1a)
plot(bestmodel1a,data=bank_dev1,formula=age~duration)
pred_2aa<-predict(bestmodel1a,bank_val1)
t3 =table(predict=pred_2aa,actual=bank_val1$y)
t3