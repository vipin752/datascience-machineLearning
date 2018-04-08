## Let us first import our dataset for modeling


LR_DF <- read.table("LR_DF.csv",sep = ",", header = T)
View(LR_DF)

summary(LR_DF)
str(LR_DF)

## See the percentile distribution
quantile(LR_DF$Balance, 
         c(0.01, 0.05, 0.1, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1))


quantile(LR_DF$Age, 
         c(0.01, 0.05, 0.1, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1))

## What if I want the percentile distribution for all the fields
apply(LR_DF[,sapply(LR_DF, is.numeric)], 
      2, quantile, 
      probs=c(0.01, 0.05, 0.1, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1),
      na.rm=T)



boxplot(LR_DF$Balance , 
             main= "Balance Box Plot" ,
             xlab = "Overall Base"
            )



## Typically we floor and cap the variables at P1 and P99. 
## Let us cap the Balance variable at P99.
LR_DF$BAL_CAP <- 
  ifelse(LR_DF$Balance > 723000, 723000, LR_DF$Balance)

summary(LR_DF$BAL_CAP)
sd(LR_DF$BAL_CAP)

quantile(LR_DF$BAL_CAP, 
         c(0.01, 0.05, 0.1, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1))


## Missing Value Imputation for Holding Period
#### Creating a function to decile the records function
decile <- function(x){
deciles <- vector(length=10)
for (i in seq(0.1,1,.1)){
deciles[i*10] <- quantile(x, i, na.rm=T)
}
return (
ifelse(x<deciles[1], 1,
ifelse(x<deciles[2], 2,
ifelse(x<deciles[3], 3,
ifelse(x<deciles[4], 4,
ifelse(x<deciles[5], 5,
ifelse(x<deciles[6], 6,
ifelse(x<deciles[7], 7,
ifelse(x<deciles[8], 8,
ifelse(x<deciles[9], 9, 10
))))))))))
}



tmp <- LR_DF
tmp$deciles <- decile(tmp$Holding_Period)

library(data.table)
tmp_DT = data.table(tmp)
RRate <- tmp_DT[, list(
  min_hp = min(Holding_Period), 
  max_hp = max(Holding_Period), 
  avg_hp = mean(Holding_Period),
  cnt = length(Target), 
  cnt_resp = sum(Target), 
  cnt_non_resp = sum(Target == 0)) , 
  by=deciles][order(deciles)]
RRate$rrate <- RRate$cnt_resp * 100 / RRate$cnt;
View(RRate)

rm(tmp)

LR_DF$HP_Imputed <- ifelse(is.na(LR_DF$Holding_Period), 
                           18, LR_DF$Holding_Period)


########## Occupation Imputation



ctab <- xtabs(~Target + Occupation, data = LR_DF)
ctab
class(LR_DF$Occupation)
LR_DF$Occupation <- as.character(LR_DF$Occupation)
LR_DF$OCC_Imputed <- ifelse(LR_DF$Occupation=="", 
                            "MISSING", LR_DF$Occupation)
table(LR_DF$OCC_Imputed)



## Let us find the variables Information Value
#install.packages("devtools")
library(devtools)
#install_github("riv","tomasgreif")
library(woe)

iv.plot.summary(iv.mult(LR_DF[,!names(LR_DF) %in% c("Cust_ID")],
                        "Target",TRUE))

iv <- iv.mult(LR_DF[,!names(LR_DF) %in% c("Cust_ID")],
        "Target",TRUE)

iv

############ model building

mydata <- LR_DF

mydata$random <- runif(nrow(mydata), 0, 1)
mydata.dev <- mydata[which(mydata$random <= 0.5),]
mydata.val <- mydata[which(mydata$random > 0.5 
                           & mydata$random <= 0.8 ),]
mydata.hold <- mydata[which(mydata$random > 0.8),]
nrow(mydata)
nrow(mydata.dev)
nrow(mydata.val)
nrow(mydata.hold)

sum(mydata$Target) / nrow(mydata)
sum(mydata.dev$Target)/ nrow(mydata.dev)
sum(mydata.val$Target)/ nrow(mydata.val)
sum(mydata.hold$Target)/ nrow(mydata.hold)


#install.packages("aod")
##install.packages("ggplot2")
library(aod)
library(ggplot2)
## Running Regression Process
mylogit <- glm(
  Target ~  Age + Gender + OCC_Imputed 
  + SCR + Balance + No_OF_CR_TXNS + HP_Imputed , 
  data = mydata.dev, family = "binomial"
)
summary(mylogit)


## After dropping Gender Variable
mylogit <- glm(
  Target ~  Age + OCC_Imputed 
  + SCR + Balance + No_OF_CR_TXNS + HP_Imputed , 
  data = mydata.dev, family = "binomial"
)
summary(mylogit)




## We need to treat Occupation Variable

pp <- as.data.frame.matrix(table(mydata.dev$OCC_Imputed, mydata.dev$Target))
pp
pp$total <- (pp$`0` + pp$`1`)
pp
pp$rrate <- round(pp$`1` * 100 / (pp$`0` + pp$`1`), 3)
pp

mydata.dev$DV_OCC = ifelse ( mydata.dev$OCC_Imputed %in% c("SAL", "SENP"), "SAL-SENP",
                              ifelse (
                                mydata.dev$OCC_Imputed %in% c("MISSING", "PROF"), "MISSING-PROF",
                                mydata.dev$OCC_Imputed
                                )
                            )
table(mydata.dev$DV_OCC)

## After creating new Derived Occupation Categories
mylogit <- glm(
  Target ~  DV_OCC 
  + SCR + Balance + No_OF_CR_TXNS + HP_Imputed , 
  data = mydata.dev, family = "binomial"
)
summary(mylogit)


#install.packages("car")
library(car)
vif(mylogit)



####### MODEL VALIDATION #########


mydata.val$DV_OCC = ifelse ( 
  mydata.val$OCC_Imputed %in% c("SAL", "SENP"), "SAL-SENP",
  ifelse (
   mydata.val$OCC_Imputed %in% c("MISSING", "PROF"), "MISSING-PROF",
   mydata.val$OCC_Imputed
  )
)


mylogit_val <- glm(
  Target ~  Age + DV_OCC 
  + SCR + Balance + No_OF_CR_TXNS + HP_Imputed , 
  data = mydata.val, family = "binomial"
)
summary(mylogit_val)


fitted.probabilities <- predict(mylogit_val,newdata=mydata.val,type='response')
table(mydata.val$Target, fitted.probabilities > 0.05)

library(InformationValue)
optCutOff <- optimalCutoff(mydata.val$Target, fitted.probabilities)[1] 
optCutOff

mydata.hold$DV_OCC = ifelse ( 
  mydata.hold$OCC_Imputed %in% c("SAL", "SENP"), "SAL-SENP",
  ifelse (
    mydata.hold$OCC_Imputed %in% c("MISSING", "PROF"), "MISSING-PROF",
    mydata.hold$OCC_Imputed
  )
)

fitted.probabilities1 <- predict(mylogit_val,newdata=mydata.hold,type='response')
table(mydata.hold$Target, fitted.probabilities1 > 0.5)
