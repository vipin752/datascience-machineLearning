#https://cran.r-project.org/web/packages/recommenderlab/vignettes/recommenderlab.pdf
#See section 5: Examples

#install.packages("recommenderlab")
library(recommenderlab)

#create a small artificial data set as a matrix
m <- matrix(sample(c(as.numeric(0:5), NA), 50, 
                   replace=TRUE, prob=c(rep(.4/6,6),.6)), ncol=10, 
            dimnames=list(user=paste("u", 1:5, sep=''), 
                          item=paste("i", 1:10, sep='')))
            
m

#converted into a realRatingMatrix object which stores
#the data in sparse format (only non-NA values are stored 
#explicitly; NA values are represented by a dot)
r <- as(m, "realRatingMatrix")
r
ratingMatrix(r)

#realRatingMatrix can be coerced back into a matrix 
#which is identical to the original matrix
identical(as(r, "matrix"),m)

as(r, "list")


as(r, "data.frame")

#An important operation for rating matrices is to normalize the entries to, e.g., centering to
#remove rating bias by subtracting the row mean from all ratings in the row.
r_m <- normalize(r)
r_m
head(as(r_m,"data.frame"))

#binarization of data
r_b <- binarize(r, minRating=4)
r_b
head(as(r_b,"data.frame"))

#dataset - rating [-10,10]
data(Jester5k)
Jester5k
head(as(Jester5k, "data.frame"))

#sample 1000 users only
set.seed(1234)
r <- sample(Jester5k, 1000)
r

#ratings for first user
rowCounts(r[1,])
as(r[1,], "list")
rowMeans(r[1,])
hist(getRatings(r), breaks=100)

recommenderRegistry$get_entries(dataType = "realRatingMatrix")

#create recommender with first 1000 users
rec_pop <- Recommender(Jester5k[1:1000], method = "UBCF")
rec_pop
rec_pop_mdl <- getModel(rec_pop)
names(rec_pop_mdl)
rec_pop_mdl$topN
rec_pop_pred <- predict(rec_pop, Jester5k[1001:1002], n=5)
rec_pop_pred
as(rec_pop_pred, "list")

#predict ratings
rec_pop_pred <- predict(rec_pop, Jester5k[1001:1002], type="ratings")
as(rec_pop_pred, "matrix")[,1:10]

rec_pop_pred <- predict(rec_pop, Jester5k[1001:1002], type="ratingMatrix")
as(rec_pop_pred, "matrix")[,1:10]
