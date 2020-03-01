#Tuning CTS
# This script evaluates several different models on the Hillstrom data set which is contained in this repository.
# The models evaluated are Contextual Treatment Selection (CTS), Causal Forest, Separate Model Approach with 
# Random Forest and a custom Tree/Random Forest with two different gain functions ("Simple" and "Frac). More 
# information about the custom Tree and Random Forest can be found under ModelImplementations/DecisionTreeImplementation.R
# The user can specify the parameter n_predictions. If it is set to one, each model is trained once on the 
# original data set and then evaluated. If n_predictions is greater than 1, there will be n_predictions iterations.
# For each iteration a bootstrap sample is taken as the new data and then the models are build. After the models
# have been built and the predictions have been made, the predictions are evaluated and the results ploted.

library(ggplot2)
library(caret)
library(plyr)
library(dplyr)
library(reshape2)

source('ModelImplementations/DecisionTreeImplementation.R')
source('ModelImplementations/RzepakowskiTree.R')
source('Evaluation Methods.R')
source('ModelImplementations/CausalTree.R')
source('ModelImplementations/Causal Forest.R')
source('ModelImplementations/Separate Model Approach.R')
source('ModelImplementations/ContextualTreatmentSelection.R')
source('ModelImplementations/VisualizationHelper.R')
source("ModelImplementations/PredictionFunctions.R")


set.seed(1234)
n_predictions <- 5
#Data import and preprocessing
email <- read.csv('Data/Email.csv')

email$men_treatment <- ifelse(email$segment=='Mens E-Mail',1,0)
email$women_treatment <- ifelse(email$segment=='Womens E-Mail',1,0)
email$control <- ifelse(email$segment=='No E-Mail',1,0)
email$mens <- as.factor(email$mens)
email$womens <- as.factor(email$womens)
email$newbie <- as.factor(email$newbie)

email$visit <- email$conversion <- email$segment <- NULL

response <- 'spend'
control <- "control"
treatment_list <- c('men_treatment','women_treatment')

original_email <- email

for(f in 1:n_predictions){
  
  # If n_predictions is > 1 as bootstrap sample is created
  if(n_predictions > 1){
    email <- original_email[sample(nrow(original_email),nrow(original_email),replace = TRUE),]
  }
  idx <- createDataPartition(y = email[ , response], p=0.2, list = FALSE)
  train <- email[-idx, ]
  
  test <- email[idx, ]
  
  test_list <- set_up_tests(train[,c("recency","history_segment","history","mens","womens","zip_code",
                                     "newbie","channel")],TRUE, max_cases = 10)
  # Partition training data for pruning
  p_idx <- createDataPartition(y = train[ , response], p=0.3, list = FALSE)
  
  val <- train[p_idx,]
  train_val <- train[-p_idx,]
  
  
  
  start_time <- Sys.time()
  for(c in c("max")){
    for(ntrees in c(100,300,500)){
      for(min_split in c(100,250,500)){
        forest <- parallel_build_random_forest(train,treatment_list,response,control,n_trees = ntrees,n_features = 3,
                                                    criterion = c, remain_cores = 10, min_split = min_split)
        pred <- predict_forest_df(forest,test,remain_cores = 10)
        write.csv(pred, paste("Predictions/Tuning/random_forest",c,as.character(ntrees),as.character(min_split),
                              as.character(f),".csv",sep = "_"), row.names = FALSE)
      }
    }
  }
  
  end_time <- Sys.time()
  print(difftime(end_time,start_time,units = "mins"))
}

start_time <- Sys.time()
folder <- "Predictions/Tuning/"
outcomes <- c()
#n_predictions <- 5
p_treated <- c()
model <- "random_forest"
for(f in 1:n_predictions){
  
  for(c in c("max")){
    for(ntrees in c(100,300,500)){
      for(min_split in c(100,250,500)){
        model2 <- paste(c,as.character(ntrees),as.character(min_split),sep = "_")
        if(length(outcomes) == 0){
          pred <- read.csv(paste("Predictions/Tuning/random_forest",c,as.character(ntrees),as.character(min_split),
                                 as.character(f),".csv",sep = "_"))
          outcomes <- c(new_expected_quantile_response(response,control,treatment_list,pred),model2)
          p_treated <- cbind(perc_treated(pred,treatment_list),treatment_list,
                             rep(model2,length(treatment_list)))
        } else{
          pred <- read.csv(paste("Predictions/Tuning/random_forest",c,as.character(ntrees),as.character(min_split),
                                  as.character(f),".csv",sep = "_"))
          outcomes <- rbind(outcomes,c(new_expected_quantile_response(response,control,treatment_list,pred),model2))
          p_treated <- rbind(p_treated, cbind(perc_treated(pred,treatment_list),treatment_list,
                                              rep(model2, length(treatment_list))))
        }
      }
    }
  }
}
outcome_df <- data.frame(outcomes)
perc_treated_df <- data.frame(p_treated)
colnames(outcome_df) <- c(0,10,20,30,40,50,60,70,80,90,100,"Model")
colnames(perc_treated_df) <- c("PercTreated","Treatment","Model")
rownames(outcome_df) <- 1:nrow(outcome_df)
rownames(perc_treated_df) <- 1:nrow(perc_treated_df)
for(c in 1:11){
  outcome_df[,c] <- as.numeric(as.character(outcome_df[,c]))
}
outcome_df[,12] <- as.character(outcome_df[,12])
perc_treated_df[,1] <- as.numeric(as.character(perc_treated_df[,1]))
perc_treated_df[,2] <- as.character(perc_treated_df[,2])
print(difftime(Sys.time(),start_time,units = "mins"))



#Visualize the results
if(n_predictions > 1){
  for(model in unique(outcome_df$Model)){
    temp_data <- outcome_df[outcome_df$Model == model,]
    n_treated <- perc_treated_df[perc_treated_df$Model == model,]
    visualize(temp_data = temp_data, multiple_predictions = TRUE)
  }
  visualize(temp_data = outcome_df, multiple_predictions = TRUE,errorbars = FALSE)
} else{
  for(model in unique(outcome_df$Model)){
    temp_data <- outcome_df[outcome_df$Model == model,]
    n_treated <- perc_treated_df[perc_treated_df$Model == model,]
    visualize(temp_data = temp_data, multiple_predictions = FALSE, n_treated = n_treated)
  }
  visualize(temp_data = outcome_df, multiple_predictions = FALSE, n_treated = perc_treated_df)
}
