## ---------------------- TRAINING WITH TREES ----------------------------
## This code creates three folds in the training data & then applies a tree
## method to each.  Records the accuracy of each tree in a data frame and
## displays the accuracy at the end

## --------------------- SETUP ENVIRONMENT ------------------------------

## Load required libraries
library(caret)
library(kernlab)
library(rattle)
library(rpart.plot)
library(randomForest) ## for trees
library(RANN)  

## ----------------- MODEL TRAINING AND TESTING -------------------------

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
## as do not need these as predictors.
training <- training[ ,8:160]
testing <- testing[ ,8:160]

## Set seed so results can be reproduced
set.seed(35478)
folds <- createFolds(y=training$classe, k=3, list=TRUE, returnTrain=TRUE)

## Matrix to hold the result of in and out of sample accuracy
treeAcc <- matrix(c(0,0,0,0,0,0),nrow=2,ncol=3)
row.names(treeAcc) <- c("In Sample","Out of Sample")
colnames(treeAcc) <- c("Fold 1", "Fold 2", "Fold 3")

for(i in 1:3) {
    tr <- training[folds[[i]], ]  ## Go through each fold in turn
    ts <- testing[-folds[[i]], ]  ## get the testing and traing set
    
    ## Turn data into numeric, apart from the outcome variable
    tr <- as.data.frame(lapply(tr[ ,-153], as.numeric))
    ts <- as.data.frame(lapply(ts[ ,-153], as.numeric))
    
    ## Add the outcome variable back in
    tr$classe <- training[folds[[i]], ]$classe
    ts$classe <- testing[-folds[[i]], ]$classe
    
    ## Train the model
    modFit <- rpart(classe ~ ., data=tr, method="class")
    ## If wanted to use Caret package then can use train fuction below
    ## modFit <- train(classe ~. , method="rpart", data=tr)
    
    ## Make predictions on training set
    trPredict <- predict(modFit, tr, type="class")
    
    ## Make predictions on test set
    predictions <- predict(modFit, ts, type="class")
    
    ## Generate confusion matrix's for training & test results
    trCm <- confusionMatrix(tr$classe, trPredict)
    cm <- confusionMatrix(ts$classe, predictions)
    
    ## Put the confusion matrix results into a list
    treeAcc[1,i] <- trCm$overall[1]
    treeAcc[2,i] <- cm$overall[1]
}

treeAcc

## After training on can view the tree using
fancyRpartPlot(modFit)  ## For non Caret models

## Generate Predictions
tsPred <- predict(modFit, ts, type="class")

## Generate confusion matrix
confusionMatrix(ts$classe, tsPred)

## ------- GENERATE PREDICTIONS FOR ACTUAL TEST SET -------------------

## Work with the test set
ts <- test

## Turn factor data into numeric
ts <- as.data.frame(lapply(ts, as.numeric))

## Make prediction using model built on training set
tsPred <- predict(modFit, ts, type="class")

## Turn predictions into character vector
answers <- as.vector(tsPred)

## Use code given to create function to convert each answer to
## a single file
pml_write_files = function(x){
         n = length(x)
         for(i in 1:n){
                 filename = paste0("problem_id_",i,".txt")
                 write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
             }
     }

## Write the answer files to current directory
pml_write_files(answers)