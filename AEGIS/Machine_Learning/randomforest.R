#Random Forest
#https://www.tutorialspoint.com/r/r_random_forest.htm


#install.packages("randomForest")
library(randomForest)

install.packages("party")
library(party)
print(head(readingSkills))

# Create the forest.
output.forest <- randomForest(nativeSpeaker ~ age + shoeSize + score, 
                              data = readingSkills)


# View the forest results.
print(output.forest)

# Importance of each predictor.
print(importance(output.forest,type = 2))
