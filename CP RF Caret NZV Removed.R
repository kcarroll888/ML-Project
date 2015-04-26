## --------------- TRAINING WITH RANDOM FORESTS MODEL 2 ------------------
## -------------- REMOVING ALL NEAR ZERO VALUE PREDICTORS ----------------

## Random forests as handled by Caret definitely use cross validation so do 
## not need to create smaller folds to test accuracy.  Train the model on 
## all available dataafter splitting into training and test sets

##  --------------------- SETUP ENVIRONMENT ------------------------------
## Load required libraries
library(caret)
library(kernlab)
library(rattle)
library(rpart.plot)

## Load & split the data
train <- read.csv("pml-training.csv", header=TRUE)
test <- read.csv("pml-testing.csv", header=TRUE)
## Set seed so splits are reproducible
set.seed(4857)
inTrain <- createDataPartition(train$classe, p=0.7, list=FALSE)
training <- train[inTrain,]
testing <- train[-inTrain,]

## Remove the loaded train csv file as duplicated
rm(train)

## Remove the index, user names, time stamps and window names & number 
## as do not want these as predictors.
training <- training[ ,8:160]
testing <- testing[ ,8:160]

## Tidy the data further as RF method only works on numeric data
tr <- as.data.frame(lapply(training[ ,-153], as.numeric))
ts <- as.data.frame(lapply(testing[ ,-153], as.numeric))

## Add back the classe outcome that we are trying to predict
tr$classe <- training$classe
ts$classe <- testing$classe

## Remove near zero value predictors
nzv <- nearZeroVar(training)
tr <- training[, -nzv]
ts <- testing[, -nzv]

## Set seed so forest grown is reproducible
set.seed(35478)

## Caret package random forest
trCont <- trainControl(method="cv", number=5)
modFit <- train(classe ~., method="rf", data=tr, trControl=trCont)

## Generate Predictions
trPred <- predict(modFit, newdata=tr)
tsPred <- predict(modFit, newdata=ts)

## Generate accuracy readings for training and test set
## First find only the observations for which the model has solved for
## since cannot predict when values of the predictor are missing
tr <- tr[complete.cases(subset(tr, select = -classe)), ]
ts <- ts[complete.cases(subset(ts, select = -classe)), ]

## Generate confusion matrix
trCm <- confusionMatrix(tr$classe, trPred)
tsCm <- confusionMatrix(ts$classe, tsPred)