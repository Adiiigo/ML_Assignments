#importing the dataset into the R 
dataset <- read.csv("/Users/AditiGoyal/Desktop/Mine/ML/datasets/regression.csv")
attach(dataset)

#Setting the seed so that we get same results each time while we run the algorithm
#Using same random number while sampling the dataset   
set.seed(123)

#Display the structure of the dataset 
str(dataset)

# install.packages("caTools")
require(caTools)
#Splitting Training dataset
sample = sample.split(dataset , SplitRatio = .70)
train = subset(dataset , sample == TRUE)
test = subset(dataset , sample == FALSE)

head(dim(train))
head(dim(test))

#Applying Regression model with LSE
train_features = train[4]
train_labels = train[2]
head(train_features)
head(train_labels)
#relation <- lm(train_features,train_labels)
relation <- lm(runs~hits,data = train)
print(summary(relation))

test_features = test[4]
test_labels = test[2]
head(test_features)
head(test_labels)
#Predicting the values
predict(relation , test_features)

#Plotting Training MSE and Test MSE
require("ggplot2")
ggplot()+geom_point(data=train,aes(x=runs , y = hits))+ggtitle("Training Dataset")
ggplot()+geom_point(data=test,aes(x=runs , y = hits))+ggtitle("Test Datatset")

#Subset Selection
#install.packages("leaps")
require("leaps")
subset = regsubsets(runs~. , data = train)
summary(subset)
names(summary(subset)) 
plot(subset,scale="bic")

#Cross Validation
#install.packages("DAAG")
library("DAAG")
cvResults <- suppressWarnings(CVlm(train, relation, m=5, dots=FALSE, seed=29, legend.pos="topleft",  printit=T))
summary(cvResults)
attr(cvResults , "ms")
