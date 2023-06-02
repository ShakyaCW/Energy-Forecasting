# Load necessary libraries
library(readxl)
library(neuralnet)
library(ggplot2)

# Load the data
consumption_data <- read_excel("uow_consumption.xlsx")

# Rename columns
colnames(consumption_data) <- c("date", "hour_18", "hour_19", "hour_20")

# Create lag variables and remove rows with missing values
consumption_data$lag_1 <- lag(consumption_data$hour_20, 1)
consumption_data$lag_2 <- lag(consumption_data$hour_20, 2)
consumption_data$lag_3 <- lag(consumption_data$hour_20, 3)
consumption_data$lag_4 <- lag(consumption_data$hour_20, 4)
consumption_data$lag_7 <- lag(consumption_data$hour_20, 7)
consumption_data <- na.omit(consumption_data)

# Define a normalization function
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

# Normalize the lag variables
consumption_data$lag_1 <- normalize(consumption_data$lag_1)
consumption_data$lag_2 <- normalize(consumption_data$lag_2)
consumption_data$lag_3 <- normalize(consumption_data$lag_3)
consumption_data$lag_4 <- normalize(consumption_data$lag_4)
consumption_data$lag_7 <- normalize(consumption_data$lag_7)

# Split the data into training and testing sets
train_data <- consumption_data[1:380, ]
test_data <- consumption_data[381:nrow(consumption_data), ]


# Convert data frames to matrices and apply normalization function to each column
train_matrix <- as.matrix(train_data[, -1])
train_normalized <- as.data.frame(apply(train_matrix, 2, normalize))
colnames(train_normalized) <- colnames(train_data)[-1]

test_matrix <- as.matrix(test_data[, -1])
test_normalized <- as.data.frame(apply(test_matrix, 2, normalize))
colnames(test_normalized) <- colnames(test_data)[-1]

# Set column names of test_normalized to match those of train_normalized
colnames(test_normalized) <- colnames(train_normalized)

# Define a list of input vectors for the neural network
input_vectors <- list(
  c("lag_1"),
  c("lag_1", "lag_2"),
  c("lag_1", "lag_2", "lag_3"),
  c("lag_1", "lag_2", "lag_3", "lag_4"),
  c("lag_1", "lag_3"),
  c("lag_2", "lag_3"),
  c("lag_4", "lag_7"),
  c("lag_2", "lag_3", "lag_4"),
  c("lag_2", "lag_3", "lag_4", "lag_7"),
  c("lag_1", "lag_2", "lag_3", "lag_4", "lag_7")
)


# Define function to build MLP model
build_mlp_model <- function(train_data, test_data, input_vars, hidden_structure) {
  
  # Create formula for model using the input variables
  formula <- paste("hour_20 ~", paste(input_vars, collapse = " + "))
  
  # Train MLP model using neuralnet package
  nn <- neuralnet(as.formula(formula), train_data, hidden = hidden_structure)
  
  # Create matrix of test data using the input variables
  test_matrix <- as.matrix(test_data[, input_vars, drop = FALSE])
  
  # Rename the columns of the test matrix to match those of the training data
  colnames(test_matrix) <- colnames(train_data[, input_vars, drop = FALSE])
  
  # Use the trained model to generate predictions for the test data
  predictions <- predict(nn, test_matrix)
  
  # Return the model and predictions in a list
  return(list(model = nn, predictions = predictions))
}

# Create a list to store all models
models <- list()
for (i in 1:length(input_vectors)) {
  models[[i]] <- build_mlp_model(train_normalized, test_normalized, input_vectors[[i]], c(5))
}

# Define function to calculate evaluation metrics
calculate_metrics <- function(actual, predicted) {
  rmse <- sqrt(mean((actual - predicted)^2))
  mae <- mean(abs(actual - predicted))
  actual_smooth <- actual + 0.001  # add a smoothening factor
  mape <- mean(abs((actual - predicted) / actual_smooth)) * 100
  smape <- mean(abs(actual - predicted) / (abs(actual) + abs(predicted)) * 2) * 100
  return(list(RMSE = rmse, MAE = mae, MAPE = mape, sMAPE = smape))
}

# Create a list to store evaluation metrics of all models
evaluation_metrics <- list()
for (i in 1:length(models)) {
  evaluation_metrics[[i]] <- calculate_metrics(test_normalized$hour_20, models[[i]]$predictions)
}

# Create a data frame to compare the evaluation metrics of all models
comparison_table <- data.frame(
  Model_Description = c("AR(1)", "AR(1,2)", "AR(1,2,3)", "AR(1,2,3,4)", "AR(1,3)", "AR(2,3)", "AR(4,7)", "AR(2,3,4)", "AR(2,3,4,7)", "AR(1,2,3,4,7)"),
  RMSE = sapply(evaluation_metrics, function(x) x$RMSE),
  MAE = sapply(evaluation_metrics, function(x) x$MAE),
  MAPE = sapply(evaluation_metrics, function(x) x$MAPE),
  sMAPE = sapply(evaluation_metrics, function(x) x$sMAPE)
)

# Print the comparison table
print(comparison_table)

#Build more models with different hidden layer structures and input vectors to create 12-15 models in total
#Compare the efficiency between one-hidden layer and two-hidden layer networks
model1_layer1 <- build_mlp_model(train_normalized, test_normalized, c("lag_1", "lag_2", "lag_3", "lag_4", "lag_7"), c(5))
model2_layer1 <- build_mlp_model(train_normalized, test_normalized, c("lag_1", "lag_2", "lag_3", "lag_4", "lag_7"), c(10))

model1_layer2 <- build_mlp_model(train_normalized, test_normalized, c("lag_1", "lag_2", "lag_3", "lag_4", "lag_7"), c(3, 2))
model2_layer2 <- build_mlp_model(train_normalized, test_normalized, c("lag_1", "lag_2", "lag_3", "lag_4", "lag_7"), c(5, 3))

# Calculate the total number of weight parameters for each network
num_weights_1_hidden1 <- sum(sapply(model1_layer1$model$weights, length))
num_weights_2_hidden1 <- sum(sapply(model1_layer2$model$weights, length))
num_weights_1_hidden2 <- sum(sapply(model2_layer1$model$weights, length))
num_weights_2_hidden2 <- sum(sapply(model2_layer2$model$weights, length))

# Print the total number of weight parameters for each network
cat("Total number of weight parameters for the first one-hidden layer network:", num_weights_1_hidden1, "\n")
cat("Total number of weight parameters for the first two-hidden layer network:", num_weights_2_hidden1, "\n")
cat("Total number of weight parameters for the second one-hidden layer network:", num_weights_1_hidden2, "\n")
cat("Total number of weight parameters for the second two-hidden layer network:", num_weights_2_hidden2, "\n")



# Part 2 - NARX Approach

# Build and evaluate NARX models with input vectors including the 18th and 19th hour attributes

# Define input vectors
input_vectors <- list(
  c("lag_1", "hour_18", "hour_19"),
  c("lag_1", "lag_2", "hour_18", "hour_19"),
  c("lag_1", "lag_2", "lag_3", "hour_18", "hour_19"),
  c("lag_1", "lag_2", "lag_3", "lag_4", "hour_18", "hour_19"),
  c("lag_1", "lag_2", "lag_3", "lag_4", "lag_7", "hour_18", "hour_19")
)

# Define different internal structures
structures <- list(
  c(5),
  c(10),
  c(3, 2),
  c(5, 3),
  c(10, 5)
)

# Define the vector to store model names
model_names <- c()

# Build MLP models with different structures and input variables
mlp_models <- list()
for (i in 1:length(structures)) {
  for (j in 1:length(input_vectors)) {
    model_name <- paste0("model_", i, "_", j)
    mlp_models[[model_name]] <- build_mlp_model(train_normalized, test_normalized, input_vectors[[j]], structures[[i]])
    model_names <- c(model_names, model_name)
  }
}

# Evaluate NARX models
narx_evaluation_metrics <- list()
for (i in 1:length(mlp_models)) {
  narx_evaluation_metrics[[i]] <- calculate_metrics(test_normalized$hour_20, mlp_models[[i]]$predictions)
}

# Create a comparison table for NARX models
narx_comparison_table <- data.frame(
  Model = model_names,
  RMSE = sapply(narx_evaluation_metrics, function(x) x$RMSE),
  MAE = sapply(narx_evaluation_metrics, function(x) x$MAE),
  MAPE = sapply(narx_evaluation_metrics, function(x) x$MAPE),
  sMAPE = sapply(narx_evaluation_metrics, function(x) x$sMAPE)
)

print(narx_comparison_table)

# Define weights for each metric
weights <- c(0.25, 0.25, 0.25, 0.25)

# Calculate weighted metric scores for each model
weighted_scores <- as.matrix(narx_comparison_table[,2:5]) %*% weights
narx_comparison_table$Weighted_Score <- weighted_scores

# Combine model names and weighted scores into a table
weighted_scores_table <- data.frame(
  Model = narx_comparison_table$Model,
  Weighted_Score = weighted_scores
)

# Sort the table by weighted score in ascending order
weighted_scores_table_sorted <- weighted_scores_table[order(weighted_scores_table$Weighted_Score), ]

# Select the top 2 models
best_models <- head(weighted_scores_table_sorted, 2)

# Print the details of the best models
cat("Best MLP models:\n")
for (i in 1:nrow(best_models)) {
  cat(best_models[i, "Model"], "\n")
  cat("Weighted Score:", best_models[i, "Weighted_Score"], "\n")
  cat("\n")
}


# Denormalize the predictions and plot the predicted output vs. desired output

# Define a function to denormalize values
denormalize <- function(x, min_value, max_value) {
  return(x * (max_value - min_value) + min_value)
}

# Getting predictions using best model 
best_model <- mlp_models[[1]]
best_model_predictions <- best_model$predictions

# Find the minimum and maximum values of the target variable
min_value <- min(train_data$hour_20)
max_value <- max(train_data$hour_20)

# Denormalize the predictions
denormalized_predictions <- denormalize(best_model_predictions, min_value, max_value)

# Plot the predicted output vs. desired output using a line chart
plot(test_data$hour_20, type = "l", col = "blue", xlab = "Time", ylab = "Hour 20 Consumption", main = "Line Chart of Desired vs. Predicted Output")
lines(denormalized_predictions, col = "red")
legend("topleft", legend = c("Desired Output", "Predicted Output"), col = c("blue", "red"), lty = 1, cex = 0.8)