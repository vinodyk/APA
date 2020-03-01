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
#source('ModelImplementations/Causal Forest.R')
source('ModelImplementations/Separate Model Approach.R')
source('ModelImplementations/ContextualTreatmentSelection.R')
source('ModelImplementations/VisualizationHelper.R')
source("ModelImplementations/PredictionFunctions.R")


set.seed(1234)
n_predictions <- 10
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

# The training and prediction part
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
  for(ntree in c(200,500,1000)){
    for(split_rule in c("CT","TOT")){
      for(split_honest in c(T,F)){
        causal_forest_pred <- causalForestPredicitons(train, test, treatment_list, response, control, ntree = ntree, 
                                                      s_rule = split_rule,s_true = split_honest)
        write.csv(causal_forest_pred, paste("Predictions/TuningCausal/causal_forest", as.character(ntree), split_rule, 
                                            as.character(split_honest), as.character(f),".csv",sep = "_"),
                  row.names = FALSE)
        
      }
    }
  }
  # Causal Forest
 
    end_time <- Sys.time()
  print(difftime(end_time,start_time,units = "mins"))
}


# Here the predictions are evaluated. Additionally we look at the treatment distribution, to see which treatments
# are assigned how often by the models.
start_time <- Sys.time()
folder <- "Predictions/TuningCausal/causal_forest"
outcomes <- c()
decile_treated <- c()
n_predictions <- 10
model <- "causal_forest"
for(ntree in c(1500)){
  for(split_rule in c("CT","TOT")){
    for(split_honest in c(T,F)){
      for(f in 1:n_predictions){
        pred <- read.csv(paste(folder, as.character(ntree), split_rule, 
                               as.character(split_honest), as.character(f),".csv",sep = "_"))
        if(length(outcomes) == 0){
          outcomes <- c(new_expected_quantile_response(response,control,treatment_list,pred),
                        paste(as.character(ntree), split_rule, 
                              as.character(split_honest),".csv",sep = "_"))
          decile_treated <- cbind(decile_perc_treated(pred,treatment_list),
                                  rep(paste(as.character(ntree), split_rule, 
                                            as.character(split_honest),".csv",sep = "_"),11*length(treatment_list)))
        } else{
          outcomes <- rbind(outcomes,c(new_expected_quantile_response(response,control,treatment_list,pred),
                                       paste(as.character(ntree), split_rule, 
                                             as.character(split_honest),".csv",sep = "_")))
          decile_treated <- rbind(decile_treated,
                                  cbind(decile_perc_treated(pred,treatment_list),
                                        rep(paste(as.character(ntree), split_rule, 
                                                  as.character(split_honest),".csv",sep = "_"),11*length(treatment_list),11)))
        }
      }
    }
  }
}
outcome_df <- data.frame(outcomes)
decile_treated_df <- data.frame(decile_treated)
colnames(outcome_df) <- c(0,10,20,30,40,50,60,70,80,90,100,"Model")
colnames(decile_treated_df) <- c("PercTreated","Treatment","Decile","Model")
rownames(outcome_df) <- 1:nrow(outcome_df)
rownames(decile_treated_df) <- 1:nrow(decile_treated_df)
for(c in 1:11){
  outcome_df[,c] <- as.numeric(as.character(outcome_df[,c]))
}
outcome_df[,12] <- as.character(outcome_df[,12])
decile_treated_df[,1] <- as.numeric(as.character(decile_treated_df[,1]))
decile_treated_df[,3] <- as.numeric(as.character(decile_treated_df[,3]))
print(difftime(Sys.time(),start_time,units = "mins"))


#Visualize the results.
if(n_predictions > 1){
  for(model in unique(outcome_df$Model)){
    temp_data <- outcome_df[outcome_df$Model == model,]
    n_treated <- decile_treated_df[decile_treated_df$Model == model,]
    visualize(temp_data = temp_data, multiple_predictions = TRUE, n_treated = n_treated)
  }
  visualize(temp_data = outcome_df, multiple_predictions = TRUE, n_treated = decile_treated_df,errorbars = FALSE)
} else{
  for(model in unique(outcome_df$Model)){
    temp_data <- outcome_df[outcome_df$Model == model,]
    n_treated <- decile_treated_df[decile_treated_df$Model == model,]
    visualize(temp_data = temp_data, multiple_predictions = FALSE, n_treated = n_treated)
  }
  visualize(temp_data = outcome_df, multiple_predictions = FALSE, n_treated = decile_treated_df)
}
