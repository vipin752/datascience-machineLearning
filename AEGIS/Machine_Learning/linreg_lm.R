## House price dataset
## Batch gradient descent for house price dataset
setwd("E:/DataScience/AEGIS/Machine Learning")
houseprice<-read.table("housepricedataset.csv",header=F,sep=",")
View(houseprice)
colnames(houseprice)<-c("Sqft","Rooms","Price")
View(houseprice)
## Define x variable matrices
x0<-c(rep(1,length(houseprice$Sqft)))
x0
x1<-houseprice$Sqft
x1
x2<-houseprice$Rooms
x2
## create the x- matrix of explanatory variables independent variables
x <- as.matrix(cbind(x0,x1,x2))
x
## create the y-matrix of dependent variables for pridiction
y <- as.matrix(houseprice$Price)
y
m<-nrow(y)
m
## mean-normalize the x variables
##https://stats.stackexchange.com/questions/29781/when-conducting-multiple-regression-when-should-you-center-your-predictor-varia
x.norm <- x
x.norm[,2] <- (x[,2] - mean(x[,2]))/sd(x[,2])
x.norm[,3] <- (x[,3] - mean(x[,3]))/sd(x[,3])

plot(x.norm[,2], y)
x.norm[,2]
x.norm[,3]
#on y-axislet's model house price (dependent or target or predicted variable)
#on x-axis using based on house area (independent or explanatory variable)
#y = theta0+theta1*x
#theta0 is the intercept
#theta1 is the coefficient corresponding to x.norm[,2]
model1 <- lm(y ~ x.norm[,2])
summary(model1)
abline(model1, col='red')

predict(model1, data=x.norm[, 2])

#non-mean normalized linear regression
plot(x[,2], y)
model2 <- lm(y ~ x[,2])
summary(model2)
abline(model2, col='blue')
#Observations: Error is large, parameter values are not balanced (theta0 is 10^2 orders larer than theta 1)

#model both features, house area and number of rooms
plot(x.norm[,2], y)
model3 <- lm(y ~ x.norm[,2]+x.norm[,3])
summary(model3)
abline(model3, col='green')


# define the gradient function dJ/dTheta: (h(x)-y))*x where h(x) = t(Theta) * x
# in matrix form this is as follows:
grad <- function(x, y, theta) {
  #gradient <- (1/m)* (t(x) %*% ((x%*%t(theta)) - y))
  gradient <- (t(x) %*% ((x%*%t(theta)) - y))
  return(t(gradient))
}

# define gradient descent update algorithm
grad.descent <- function(x, maxit){
  theta <- matrix(c(0,0,0), nrow=1) # Initialize the parameters
  
  alpha = .01 # set learning rate
  for (i in 1:maxit) {
    theta <- theta - alpha  * grad(x, y, theta)
  }
  return(theta)
}

# results with feature scaling
print(grad.descent(x.norm,1000))
print(grad.descent(x.norm,2000))
print(grad.descent(x.norm,3000))
print(grad.descent(x.norm,4000))
print(grad.descent(x.norm,5000))

#340398 109848 -5866.454
#340412.7 110620.6 -6639.018

#TODO: Plot theta vs Jtheta
#TODO: Stochastic Gradient Descent (ref: )