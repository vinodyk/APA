library(causalTree)

causalForestPredicitons <- function(train,test,treatment_list,response,control){
  pred <- data.frame(rep(0,nrow(test)))
  for(t in treatment_list){
    train_data <- train[train[,setdiff(treatment_list,t)] == 0,]
    train_data_new <- train_data[,setdiff(colnames(train_data),c(control,treatment_list))]

    test_data <- test[,setdiff(colnames(train_data),response)]
    
    forest <- causalForest(as.formula(paste(response,paste(setdiff(colnames(train_data_new),response),collapse = "+"),
                                            sep = "~")), data = train_data_new, treatment = train_data[,t], 
                           split.Rule = "CT", cv.option = "CT", split.Honest = T, cv.Honest = T, split.Bucket = F, 
                           minsize = 20, propensity = 0.5, mtry = 2, num.trees = 200, ncov_sample = 3, 
                           ncolx = ncol(train_data_new)-1)
    
    assign(paste('predictions',t,sep = '_'), predict(forest, newdata = test_data))
    pred <- cbind(pred,eval(as.name(paste('predictions',t,sep = '_'))))
  }
  colnames(pred) <- c(control,treatment_list)
  for (t in treatment_list) {
    pred[,paste("uplift",t,sep = "_")] <- pred[t] - pred[control]
  }
  pred[ , "Treatment"] <- predictions_to_treatment(pred, treatment_list, control)
  pred[ , "Assignment"] <- predictions_to_treatment(test, treatment_list, control)
  pred[, "Outcome"] <- test[,response]
  return(pred)
}
