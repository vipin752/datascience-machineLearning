library(randomForest)

myurl <- "https://raw.githubusercontent.com/spitakiss/Data607_Pres1/master/FullCoverage.csv"
full.cov = read.csv(myurl,header=TRUE)


full.cov$men <- factor(full.cov$men)
full.cov$urban <- factor(full.cov$urban)
full.cov$private <- factor(full.cov$private)
full.cov$y <- factor(full.cov$y)

head(full.cov)
summary(full.cov)

# Variable Definitions
# 
# AGE: integer age of policyholder
# SENIORITY: number of years at company
# MEN: 1 = male, 0 = female
# URBAN: 1= urban environment, 0 = rural environment
# PRIVATE: 1 = private use, 0 = commercial use
# MARTIAL: "c" = married couple, "s" = single, "o" = other
# Y: dependent, or response variable, 1 = full coverage, 0 = not full coverage


attach(full.cov)

# Fit to Logistic regression 
FullcovModel = glm(y~men+urban+private+factor(marital)+age+seniority,family=binomial(link=logit))

# Model Output
summary(FullcovModel)

###### Random Forrest Model
rf <- randomForest(y~men+urban+private+factor(marital)+age+seniority,data=full.cov)

# Model output
print(rf) 

importance(rf)


######### ROC curves

# Calculate sensitivity and false positive measures for logit model

fity_ypos <- FullcovModel$fitted[y == 1]
fity_yneg <- FullcovModel$fitted[y == 0]

sort_fity <- sort(FullcovModel$fitted.values)
View(sort_fity)
sens <- 0
spec_c <- 0

for (i in length(sort_fity):1){
  sens <- c(sens, mean(fity_ypos >= sort_fity[i]))
  spec_c <- c(spec_c, mean(fity_yneg >= sort_fity[i]))
  
} 
View(sens)
View(spec_c)


# Calculate sensitivity and false positive measure for random forest model

fity_ypos2 <- as.numeric(rf$pred[y == 1]) - 1
fity_yneg2 <- as.numeric(rf$pred[y == 0]) - 1

fity_ypos2
fity_yneg2

###########

sort_fity2 <- as.numeric(sort(rf$pred)) - 1

sens2 <- 0
spec_c2 <- 0

for (i in length(sort_fity2):1){
  sens2 <- (c(sens2, mean(fity_ypos2 >= sort_fity2[i])))
  spec_c2 <- (c(spec_c2, mean(fity_yneg2 >= sort_fity2[i])))
} 

plot(sens,spec_c2)
# plot ROC curves

plot(spec_c, sens, xlim = c(0, 1), ylim = c(0, 1), type = "l", 
     xlab = "false positive rate", ylab = "true positive rate", col = 'blue')
abline(0, 1, col= "black")

lines(spec_c2, sens2, col='green')
legend("topleft", legend = c("logit","random forest") , pch = 15, bty = 'n', col = c("blue","green"))


############## Area Under Curve with Logistic


npoints <- length(sens)

# Discrete approximation area under the curve, using Trapezoidal Rule 
area <- sum(0.5 * (sens[-1] + sens[-npoints]) * (spec_c[-1] - 
                                                   spec_c[-npoints]))



########### Random Forrest

npoints2 <- length(sens2)

# Discrete approximation area under the curve, using Trapezoidal Rule
area2 <- sum(0.5 * (sens2[-1] + sens2[-npoints2]) * (spec_c2[-1] - 
                                                       spec_c2[-npoints2]))