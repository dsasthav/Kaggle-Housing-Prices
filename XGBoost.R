# XGBoost for Kaggle Housing Prices Dataset
setwd("C:/Users/sasth/Documents/Github/Kaggle-Housing-Prices")

# Load Libraries ----------------------------------------------------------


library(caret)
library(plyr)
library(dplyr)
library(ggplot2)
library(xgboost)
library(randomForest)


# Load Data ---------------------------------------------------------------


train <- read.csv("train.csv")
test <- read.csv("test.csv")

labels <- train$SalePrice

all_data <- rbind(select(train, -SalePrice), test)
all_data <- select(all_data, -Id)


# Data Exploration --------------------------------------------------------

# Plot chart for each variable against SalePrice
# for (col in colnames(train)){


# Data Cleaning -------------------------------------

# MSSubClass is a categorical feature
all_data$MSSubClass <- as.factor(all_data$MSSubClass)

# Data Cleaning - Fill in NA's --------------------------------------------

# Check number of NA's in each column
na_cols <- sapply(all_data, FUN = function(x) sum(is.na(x)))
na_cols <- na_cols[na_cols > 0]
na_cols <- na_cols[! names(na_cols) %in% c("MiscFeature", "MiscVal")] # Handle these variables differently

# For all columns with NA's:
# Factors - add a level for NA's
# Numerics - impute NA's with 0's

for (col in names(na_cols)){
  
  if (is.factor(all_data[[col]])){
   
    numLevels <- length(levels(all_data[[col]]))
    all_data[[col]] <- addNA(all_data[[col]])
    all_data[[col]] <- mapvalues(x = all_data[[col]], 
                                from = levels(all_data[[col]]), 
                                to = c(levels(all_data[[col]])[1:numLevels],paste("No_",col)))
    
  } else {
    
    all_data[[col]][is.na(all_data[[col]])] <- 0
    
  }
}

# Handle Miscellaneous Features
all_data$ShedValue <- ifelse(all_data$MiscFeature == "Shed", all_data$MiscVal,0)
all_data$SecondGarageValue <- ifelse(all_data$MiscFeature == "Gar2", all_data$MiscVal,0)
all_data$TennisCourtValue <- ifelse(all_data$MiscFeature == "TenC", all_data$MiscVal,0)
all_data$OtherValue <- ifelse(all_data$MiscFeature == "Othr", all_data$MiscVal,0)
added_cols <- c("ShedValue", "SecondGarageValue", "TennisCourtValue", "OtherValue")
all_data[added_cols][is.na(all_data[added_cols])] <- 0
all_data <- select(all_data, -MiscFeature,-MiscVal)


# Data Cleaning - Unskewing -----------------------------------------------

# Unskew target variable
labels_unskewed <- log1p(labels)

# Data Cleaning - One Hot Encoding ----------------------------------------

# One Hot Encoding
for (col in names(all_data)){
  if(is.factor(all_data[[col]])){
    formula <- paste("~", col)
    dummy <- dummyVars(formula, all_data)
    dummy_df <- predict(dummy, all_data)
    all_data <- cbind(all_data, dummy_df)
    all_data <- select(all_data, -one_of(c(col)))
  }
  
}


# Baseline - Random Forest ------------------------------------------------

# Variable Importance
df_train <- all_data[1:nrow(train),]
rf <- randomForest(x = df_train, y = labels, ntree = 50, importance = TRUE)
imp <- data.frame(importance(rf))
imp$Variable <- rownames(imp)
names(imp) <- c("IncMSE", "IncNodePurity", "Variable")
imp <- imp %>% arrange(desc(IncMSE))

# Check baseline performance
rfControl <- trainControl(method = "cv", number = 5)
rf <- train(x = df_train, y = labels, method = "rf",
            trControl = rfControl)
print(rf)

# RMSE of 29117


# XGBoost 1 ---------------------------------------------------------------


important_vars <- filter(imp, IncMSE > 0)
all_data_filtered <- select(all_data, one_of(important_vars$Variable))

train_set <- all_data_filtered[1:nrow(train),]
test_set <- all_data_filtered[1461:nrow(all_data_filtered),]

# First use default parameters
xgb_default_control <- trainControl(method = "cv", number = 5)
xgb_default <- train(x = train_set, y = labels, 
                     method = "xgbTree", trControl = xgb_default_control)

print(xgb_default)


# XGBoost Library ---------------------------------------------------------

# First tune min_child_weightand max_depth
xgb_grid_1 <- expand.grid(nrounds = 100,
                          max_depth = c(2,4,6,8,10, 12),
                          min_child_weight = c(0.5,1,5),
                          eta = 0.1,
                          gamma = 0.1,
                          colsample_bytree = 1,
                          subsample = 1)

xgb_control_1 <- trainControl(method = "cv", number = 5)

xgb1 <- train(x = train_set, y = labels,
              method = "xgbTree", trControl = xgb_control_1,
              tuneGrid = xgb_grid_1)

# Max Depth: 4, Min Child Weight: 0.5, RMSE: 26071
# Get a little more specific with max depth and min_child_weight

xgb_grid_2 <- expand.grid(nrounds = 100,
                          max_depth = c(2,3,4,5,6),
                          min_child_weight = c(0.3,0.4,0.5,0.6,0.7),
                          eta = 0.1,
                          gamma = 0.1,
                          colsample_bytree = 1,
                          subsample = 1)
xgb_control_2 <- trainControl(method = "cv", number = 5)

xgb2 <- train(x = train_set, y = labels,
              method = "xgbTree", trControl = xgb_control_2,
              tuneGrid = xgb_grid_2)
print(xgb2)

# Max_depth: 3, Min_child_weigt: 0.3
# Best RMSE: 30152

# Tune eta, gamma, colsample_bytree
xgb_grid_3 <- expand.grid(nrounds = 1000,
                          max_depth = c(3,4,5),
                          min_child_weight = c(0.2,0.3,0.4),
                          eta = c(0.01,0.025,0.05,0.1),
                          subsample = 1,
                          gamma = c(0, 0.1),
                          colsample_bytree = c(0.3,0.5))
xgb_control_3 <- trainControl(method = "cv", number = 5)

xgb3 <- train(x = train_set, y = labels,
              method = "xgbTree", trControl = xgb_control_3,
              tuneGrid = xgb_grid_3)
print(xgb3)

# nrounds = 1000, The final values used for the model were nrounds = 1000, max_depth = 3,
#eta = 0.05, gamma= 0.1, colsample_bytree = 0.3, min_child_weight = 0.4 and subsample = 1. 
# Best RMSE: 24316

# Create submission for xgb3

preds <- predict(xgb3, test_set)
submission_xgb3 <- data.frame(cbind(test$Id, preds))
colnames(submission_xgb3) <- c("Id", "SalePrice")
write.csv(submission_xgb3, "Submission_xgb_12.29.16.csv", row.names = FALSE)
