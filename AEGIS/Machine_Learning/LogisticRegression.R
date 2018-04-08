#Source video link: https://www.youtube.com/watch?v=mteljf020EE
#1: Load and try to understand the data 
#install.packages("ISLR")
library(ISLR)
attach(Smarket)
View(Smarket)
?Smarket

summary(Smarket)

cor(Smarket[,1:8])
pairs(Smarket[,1:8])

#2 Split the data into training and test
smp_size <- floor(0.75 * nrow(Smarket))

## set the seed to make your partition reproductible
set.seed(123)
train_ind <- sample(seq_len(nrow(Smarket)), size = smp_size)

train <- Smarket[train_ind, ]
test <- Smarket[-train_ind, ]
train_ind
direction_test <- Direction[-train_ind]
direction_test
#3: Fit logistic regression
stock_model = glm(Direction~ Lag1+Lag2+Lag3+Lag4+Lag5+Volume,
                  data = train, family = binomial)
summary(stock_model)

model_pred_probs = predict(stock_model, test, type="response")
model_pred_dir = rep("Down", nrow(test))
str(model_pred_dir)
model_pred_dir[model_pred_probs > 0.5] = "Up"

table(model_pred_dir, direction_test)

?glm
