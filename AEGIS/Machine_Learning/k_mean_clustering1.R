#######

library(ISLR)

library(ggplot2)

iris
#########

pl=ggplot(iris,aes(Petal.Length,Petal.Width,color=Species))
print(pl+geom_point(size=4))

#########

set.seed(101)
iriscluster=kmeans(iris[,1:4],3,nstart = 20,iter.max = 200)
print(iriscluster)
table(iriscluster$cluster,iris$Species)


#########

library(cluster)
clusplot(iris,iriscluster$cluster,color = T,shade = T,lines = 0,labels = 0)


########
#Elbow Method for finding the optimal number of clusters
set.seed(123)
# Compute and plot wss for k = 2 to k = 15.
k.max <- 15
wss <- sapply(1:k.max, 
              function(k){kmeans(iris[,1:4], k, nstart=50,iter.max = 50 )$tot.withinss})
wss
plot(1:k.max, wss,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")
