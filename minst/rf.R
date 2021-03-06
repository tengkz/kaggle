library(randomForest)
library(readr)

set.seed(0)

numTrain<-10000
numTrees<-25
train<-read_csv("train.csv")
test<-read_csv("test.csv")
rows<-sample(1:nrow(train),numTrain)
labels<-as.factor(train[rows,1])
train<-train[rows,-1]
rf<-randomForest(train,labels,xtest=test,ntree=numTrees)
predictions<-data.frame(ImageId=1:nrow(test),Label=levels(labels)[rf$test$predicted])
