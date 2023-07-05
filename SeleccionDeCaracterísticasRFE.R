# Cargar el dataset iris
data(iris)

# Convertir la variable de respuesta a binaria
iris$Species <- ifelse(iris$Species == "versicolor", 1, 0)

# Dividir el dataset en conjunto de entrenamiento y conjunto de prueba
set.seed(123)
indices <- sample(1:nrow(iris), nrow(iris) * 0.7)
train_data <- iris[indices, ]
test_data <- iris[-indices, ]

# Ajustar el modelo
library(caret)
control <- rfeControl(functions = rfFuncs, method = "cv", number = 5)
modelo_rfe <- rfe(train_data[, -5], train_data$Species, sizes = c(1:4), rfeControl = control)

# Obtener las características seleccionadas
caracteristicas_seleccionadas <- predictors(modelo_rfe)

# Imprimir las características seleccionadas
cat("Características seleccionadas:", caracteristicas_seleccionadas, "\n")
