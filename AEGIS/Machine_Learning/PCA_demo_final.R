path <- "C:\\Users\\rohit\\Desktop\\ML Foundation Course\\PCA"
setwd(path)

#load train and test file
train <- read.csv("Train_UWu5bXk.csv")
test <- read.csv("Test_u94Q5KV.csv")
#add a column
 test$Item_Outlet_Sales <- 1

#combine the data set
 combi <- rbind(train, test)

#impute missing values with median
 combi$Item_Weight[is.na(combi$Item_Weight)] <- median(combi$Item_Weight, na.rm = TRUE)

#impute 0 with median
 combi$Item_Visibility <- ifelse(combi$Item_Visibility == 0, median(combi$Item_Visibility),                                   combi$Item_Visibility)

#find mode and impute
  table(combi$Outlet_Size, combi$Outlet_Type)
levels(combi$Outlet_Size)[1] <- "Other"
table(combi$Outlet_Size, combi$Outlet_Type)

#remove the dependent and identifier variables

my_data <- subset(combi, select = -c(Item_Outlet_Sales, Item_Identifier,Outlet_Identifier))

#check available variables
colnames(my_data)

#Since PCA works on numeric variables, let's see if we have any variable other than numeric.

#check variable class
str(my_data)

#Sadly, 6 out of 9 variables are categorical in nature. We have some additional work to do now.
#We'll convert these categorical variables into numeric using one hot encoding.

#load library
#install.packages("dummies")
library(dummies)

#create a dummy data frame
new_my_data <- dummy.data.frame(my_data, names = c("Item_Fat_Content","Item_Type",
                                                     "Outlet_Establishment_Year","Outlet_Size",
                                                     "Outlet_Location_Type","Outlet_Type"))
#check the data set
str(new_my_data)

#divide the new data
pca.train <- new_my_data[1:nrow(train),]
pca.test <- new_my_data[-(1:nrow(train)),]

#The base R function prcomp() is used to perform PCA. By default, 
#it centers the variable to have mean equals to zero. With parameter scale. = T, 
#we normalize the variables to have standard deviation equals to 1.

#principal component analysis
prin_comp <- prcomp(pca.train, scale. = T)
names(prin_comp)


#outputs the mean of variables
prin_comp$center

#outputs the standard deviation of variables
prin_comp$scale

prin_comp$rotation
prin_comp$rotation[1:5,1:4]

biplot(prin_comp, scale = 0)

#compute standard deviation of each principal component
std_dev <- prin_comp$sdev

#compute variance
pr_var <- std_dev^2

#check variance of first 10 components
 pr_var[1:10]
 
 #proportion of variance explained
 prop_varex <- pr_var/sum(pr_var)
 prop_varex[1:20]
 
 
 #scree plot
  plot(prop_varex, xlab = "Principal Component",
        ylab = "Proportion of Variance Explained",
        type = "b")
 
  #cumulative scree plot
   plot(cumsum(prop_varex), xlab = "Principal Component",
         ylab = "Cumulative Proportion of Variance Explained",
         type = "b")
   
#add a training set with principal components
 train.data <- data.frame(Item_Outlet_Sales = train$Item_Outlet_Sales, prin_comp$x)
   
#we are interested in first 30 PCAs
 train.data <- train.data[,1:31]
 
 
 #Now use linear regression to predict the sales.
 
 predicted_test=predict(prin_comp,data=pca.test)
 
 predicted_test=as.data.frame(predicted_test)
 test_new=predicted_test[,1:30]
 
 final_model=lm(Item_Outlet_Sales ~ .,data = train.data)

  final_result=predict(final_model,test_new)
  View(final_result)
  