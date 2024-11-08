# Carregar pacotes
#install.packages("caret")
#install.packages("xgboost")
#install.packages("Metrics")

#install.packages("fastDummies")
library(fastDummies)
library(readxl)
library(dplyr)
library(caret)
library(xgboost)
library(Metrics)
library(Matrix)
options(warn=-1)


dataset_dados <- read_excel("./RTA_Dataset.xlsx")


dataset_dados <- dataset_dados %>%
      select(age_1, age_2, Sex_of_driver_1, Driving_experience_1, Road_surface_conditions, Weather_conditions, 
             Accident_severity, Day_of_week, Educational_level)


head(dataset_dados)


dataset_dados <- dataset_dados %>%
      mutate(across(c(Road_surface_conditions, Weather_conditions, Educational_level, Day_of_week), as.factor)) %>%
      dummy_cols(select_columns = c("Road_surface_conditions", "Weather_conditions", "Educational_level", "Day_of_week"))


x <- dataset_dados %>% select(-Accident_severity)
y <- as.factor(dataset_dados$Accident_severity)


set.seed(42)
train_index <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- x[train_index, ]
X_test <- x[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index]


X_train <- data.frame(lapply(X_train, function(x) {
      if(is.character(x) || is.factor(x)) as.numeric(as.factor(x)) else x
}))

dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = as.numeric(y_train) - 1)

X_test <- data.frame(lapply(X_test, function(x) {
      if(is.character(x) || is.factor(x)) as.numeric(as.factor(x)) else x
}))

dtest <- xgb.DMatrix(data = as.matrix(X_test), label = as.numeric(y_test) - 1)


# Definindo parâmetros
params <- list(
      objective = "multi:softprob",  # Usa softmax para múltiplas classes
      eval_metric = "mlogloss",      # Métrica para múltiplas classes
      num_class = length(unique(y_train))  # Número de classes únicas no target
)

# Treinamento
model <- xgb.train(
      params = params,
      data = dtrain,
      nrounds = 10000,
      watchlist = list(eval = dtest),
      early_stopping_rounds = 10,
      print_every_n = 100
)


xgb.importance(model = model) %>%
      xgb.plot.importance(top_n = 20)


train_pred <- predict(model, dtrain)
train_pred_class <- ifelse(train_pred > 0.5, 1, 0)
train_score <- mean(as.numeric(y_train) - 1 == train_pred_class)
cat("Score de treino:", train_score, "\n")


y_pred <- predict(model, dtest)
predictions <- ifelse(y_pred > 0.5, 1, 0)
cat("Número de previsões:", length(predictions), "\n")


taxa_erro <- function(valores_reais, valores_previstos) {
     
      rmse <- rmse(valores_reais, valores_previstos)
      cat("Erro médio quadrático:", rmse, "\n")
      
      mae <- mae(valores_reais, valores_previstos)
      cat("Erro médio absoluto:", mae, "\n")
}


taxa_erro(as.numeric(y_test) - 1, predictions)


head(x, 1)
x <- data.frame(lapply(x, function(coluna) {
      if (is.character(coluna) || is.factor(coluna)) as.numeric(as.factor(coluna)) else coluna
}))

predictions <- predict(model, xgb.DMatrix(data = as.matrix(head(x, 3))))
predictions
head(y, 1)
