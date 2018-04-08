library(ggplot2)

#lets generate a sine wave
#we know the ground truth (the underlying process that generates data)
#in real life scenarios this wouldn't be possible
x=seq(0,10,0.1)
y=sin(x)
qplot(x,y,geom="path", xlab="time", ylab="Sine wave")

#create sample training data -- lets create a small sample
x_samp = sample(x, size=14)
y_samp = sin(x_samp)
plot(x_samp, y_samp, col="red")

#let's apply linear regression (although we know that relationship between x and y is non-linear)
model1 <- lm(y_samp ~ x_samp)
summary(model1)
abline(model1, col='blue')

#Idea1: Let's increase the sample size. I had taken very few samples to detect underlying trends
x_samp = sample(x, size=60)
y_samp = sin(x_samp)
plot(x_samp, y_samp, col="red")
model1a <- lm(y_samp ~ x_samp)
summary(model1a)
abline(model1, col='blue')
abline(model1a, col='green')
#standard error reduced but phew!
#Idea 2: Increase sample size further! NOT good
#Idea 3: Collect more data. Nay, in this case it will not help.
#Idea 4: Build a higher order model. 
#Let's try y = \theta_0 + \theta_1*x + \theta_2*sqr(x)
x_samp = sample(x, size=14)
y_samp = sin(x_samp)
X_samp <- as.matrix(cbind(x_samp, x_samp*x_samp))
model2 <- lm(y_samp ~ X_samp)
summary(model2)
y_pred <- predict(model2, data=X_samp[,1:2])
plot(x, y)
points(x_samp,y_pred, col="blue")

#Build higher order model
X_samp <- as.matrix(cbind(x_samp, x_samp^2, x_samp^3))
model3 <- lm(y_samp ~ X_samp)
summary(model3)
y_pred <- predict(model3, data=X_samp[,1:3])
points(x_samp,y_pred, col="green")

X_samp <- as.matrix(cbind(x_samp, x_samp^2, x_samp^3, x_samp^4))
model4 <- lm(y_samp ~ X_samp)
summary(model4)
y_pred <- predict(model4, data=X_samp[,1:4])
points(x_samp,y_pred, col="brown")

X_samp <- as.matrix(cbind(x_samp, x_samp^2, x_samp^3, x_samp^4, x_samp^5, x_samp^6))
model6 <- lm(y_samp ~ X_samp)
summary(model6)
y_pred <- predict(model6, data=X_samp[,1:4])
points(x_samp,y_pred, col="orange")


##USE GLMNET, y_pred length differs
#ERR: y_pred only returns 60 points
#X_pred <- as.matrix(cbind(x, x^2, x^3, x^4, x^5, x^6, x^7, x^8, x^9))
#y_pred <- predict(model9, data=X_pred[,1:9])
#points(X_pred[1:60,1],y_pred, col="orange")

#install.packages('glmnet')
library(glmnet)
x_samp = sample(x, size=10)
y_samp = sin(x_samp)
X_samp <- as.matrix(cbind(x_samp, x_samp^2, x_samp^3, x_samp^4, x_samp^5, x_samp^6, x_samp^7, x_samp^8, x_samp^9))
fit = glmnet(X_samp, y_samp, nlambda=1, lambda=0)
print(fit)
coef(fit, s = 0)
y_pred <- predict(fit, newx=X_samp[,1:9], s = 0)
#par(mfrow=c(1,2))
plot(x, y, col="red")
points(x_samp, y_samp, col="green")
points(x_samp,y_pred, col="blue")

#lets see how it performs on unseen data
x=seq(0,20,0.1)
y=sin(x)
X_pred <- as.matrix(cbind(x, x^2, x^3, x^4, x^5, x^6, x^7, x^8, x^9))
y_pred <- predict(fit, newx=X_pred[,1:9], s = 0)
plot(x, y)
points(X_pred[,1],y_pred, col="brown")
coef()
