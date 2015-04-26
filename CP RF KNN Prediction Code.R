## ---------- TRAINING WITH RANDOM FORESTS, KNN IMPUTING ------------------

## ----------------------- SETUP ENVIRONMENT ------------------------------

## Load required libraries
library(caret)
library(kernlab)
library(rattle)
library(rpart.plot)
library(randomForest) ## for Random Forest
library(RANN)  

## Load & split the data
train <- read.csv("pml-training.csv", header=TRUE)

## Set seed so splits are reproducible
set.seed(4857)
inTrain <- createDataPartition(train$classe, p=0.7, list=FALSE)
training <- train[inTrain,]
testing <- train[-inTrain,]

## Remove the loaded train csv file as duplicated
rm(train)

## Remove near zero value predictors
nzv <- nearZeroVar(training)
tr <- training[, -nzv]
ts <- testing[, -nzv]

## Remove the index, user names, time stamps and window names & number 
## as do not want these as predictors.
tr <- tr[ ,7:106]
ts <- ts[ ,7:106]

## Set seed so forest grown is reproducible
set.seed(35478)

## Caret train function to generate RF with KNN Imputing
trCont <- trainControl(method="cv", number=5)
modFit <- train(classe ~., method="rf", preProcess="knnImpute", data=tr, trControl=trCont)

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

## Display accuracy table
trTab <- table(trPred, tr$classe)
tsTab <- table(tsPred, ts$classe)

## Inspect the in & out of sample results
prop.table(trTab, 1)
prop.table(tsTab, 1)

## ----------- APPLYING TO TEST TO OBTAIN FINAL PREDICTION ---------------------

## Read in the final test data
test <- read.csv("pml-testing.csv", header=TRUE)

## Only use the columns that were used to train the model
ts <- test[, -nzv]

## Numeric values only
ts <- as.data.frame(lapply(ts, as.numeric))

## Remove the index, user names, time stamps and window names & number 
## as do not want these as predictors.
ts <- ts[ ,7:106]

## Get the prediction
finalPred <- predict(modFit, newdata=ts)

