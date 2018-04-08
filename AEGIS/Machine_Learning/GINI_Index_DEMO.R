########### 

install.packages("ineq")

library(ineq)

###########
data(AirPassengers)

View(AirPassengers)

ineq(AirPassengers,type="Gini")

########## Plotting Lorentz Curve

plot(Lc(AirPassengers))

plot(Lc(AirPassengers),col="darkred",lwd=2)
