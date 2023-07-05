# Cargar el dataset iris
data(iris)

# Seleccionar las variables predictoras (ancho del sépalo) y la variable de respuesta (largo del pétalo)
x <- iris$Sepal.Width
y <- iris$Petal.Length

# Ajustar el modelo de regresión lineal
modelo <- lm(y ~ x)

# Obtener los coeficientes del modelo
coeficientes <- coef(modelo)

# Imprimir los coeficientes
cat("Intercepto:", coeficientes[1], "\n")
cat("Coeficiente de x:", coeficientes[2], "\n")

# Realizar predicciones con el modelo
predicciones <- predict(modelo)

# Crear un gráfico de dispersión con la línea de regresión
plot(x, y, main = "Regresión Lineal - Iris Dataset",
     xlab = "Ancho del Sépalo", ylab = "Largo del Pétalo")
abline(modelo, col = "red")

# Agregar las predicciones al gráfico
points(x, predicciones, col = "blue", pch = 16)
legend("topleft", legend = c("Observaciones", "Predicciones"), 
       col = c("black", "blue"), pch = c(1, 16))

# Cargar el dataset iris
data(iris)

# Convertir la variable de respuesta a binaria
iris$Species <- ifelse(iris$Species == "versicolor", 1, 0)

# Dividir el dataset en conjunto de entrenamiento y conjunto de prueba
set.seed(123)
indices <- sample(1:nrow(iris), nrow(iris)*0.7)
train_data <- iris[indices, ]
test_data <- iris[-indices, ]

# Ajustar el modelo de regresión logística
modelo <- glm(Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, 
              data = train_data, family = binomial)

# Obtener las predicciones de clase en el conjunto de prueba
predicciones <- predict(modelo, newdata = test_data, type = "response")
clases_predichas <- ifelse(predicciones >= 0.5, 1, 0)

# Calcular la precisión del modelo
precision <- sum(clases_predichas == test_data$Species) / length(test_data$Species)
cat("Precisión del modelo:", precision, "\n")

# Crear una matriz de confusión
library(caret)
confusion_matrix <- confusionMatrix(factor(clases_predichas), factor(test_data$Species))
print(confusion_matrix)

# Graficar las clases reales y las predichas
library(ggplot2)
ggplot(test_data, aes(x = Sepal.Length, y = Sepal.Width, color = factor(Species))) +
  geom_point() +
  geom_point(data = data.frame(test_data$Sepal.Length, test_data$Sepal.Width, clases_predichas),
             aes(x = test_data$Sepal.Length, y = test_data$Sepal.Width),
             color = "blue", size = 4, alpha = 0.5) +
  labs(title = "Regresión Logística - Iris Dataset",
       x = "Ancho del Sépalo", y = "Largo del Sépalo",
       color = "Especie Real") +
  scale_color_manual(values = c("red", "green"), labels = c("Versicolor", "No Versicolor")) +
  theme_minimal()

# Cargar el dataset iris
data(iris)

# Seleccionar las variables predictoras (ancho del sépalo) y la variable de respuesta (largo del pétalo)
x <- iris$Sepal.Width
y <- iris$Petal.Length

# Ajustar el modelo de regresión polinomial de grado 2
modelo <- lm(y ~ poly(x, 2, raw = TRUE))

# Obtener los coeficientes del modelo
coeficientes <- coef(modelo)

# Imprimir los coeficientes
cat("Intercepto:", coeficientes[1], "\n")
cat("Coeficiente de x:", coeficientes[2], "\n")
cat("Coeficiente de x^2:", coeficientes[3], "\n")

# Realizar predicciones con el modelo
predicciones <- predict(modelo)

# Crear un gráfico de dispersión con la curva de regresión polinomial
plot(x, y, main = "Regresión Polinomial - Iris Dataset",
     xlab = "Ancho del Sépalo", ylab = "Largo del Pétalo")
points(x, predicciones, col = "red", pch = 16)

# Agregar leyenda
legend("topleft", legend = "Regresión Polinomial", col = "red", pch = 16)

# Cargar el dataset iris
data(iris)

# Dividir el dataset en conjunto de entrenamiento y conjunto de prueba
set.seed(123)
indices <- sample(1:nrow(iris), nrow(iris)*0.7)
train_data <- iris[indices, ]
test_data <- iris[-indices, ]

# Ajustar el modelo de árbol de decisión
library(rpart)
modelo <- rpart(Species ~ ., data = train_data)

# Realizar predicciones en el conjunto de prueba
predicciones <- predict(modelo, newdata = test_data, type = "class")

# Calcular la precisión del modelo
precision <- sum(predicciones == test_data$Species) / length(test_data$Species)
cat("Precisión del modelo:", precision, "\n")

# Graficar el árbol de decisión
library(rpart.plot)
rpart.plot(modelo, box.palette = "RdBu", shadow.col = "gray", nn = TRUE)

# Cargar el dataset iris
data(iris)

# Convertir la variable de respuesta a binaria
iris$Species <- ifelse(iris$Species == "versicolor", 1, 0)

# Dividir el dataset en conjunto de entrenamiento y conjunto de prueba
set.seed(123)
indices <- sample(1:nrow(iris), nrow(iris)*0.7)
train_data <- iris[indices, ]
test_data <- iris[-indices, ]

# Ajustar el modelo SVM
library(e1071)
modelo <- svm(Species ~ ., data = train_data, kernel = "linear")

# Realizar predicciones en el conjunto de prueba
predicciones <- predict(modelo, newdata = test_data)

# Calcular la precisión del modelo
precision <- sum(predicciones == test_data$Species) / length(test_data$Species)
cat("Precisión del modelo:", precision, "\n")

# Graficar el SVM con margen
library(ggplot2)
ggplot(train_data, aes(x = Sepal.Length, y = Sepal.Width, color = factor(Species))) +
  geom_point() +
  geom_point(data = data.frame(test_data$Sepal.Length, test_data$Sepal.Width, predicciones),
             aes(x = test_data$Sepal.Length, y = test_data$Sepal.Width),
             color = "blue", size = 4, alpha = 0.5) +
  geom_smooth(method = "svm", se = FALSE, size = 1, linetype = "dashed",
              aes(fill = factor(Species)), alpha = 0.2) +
  labs(title = "Máquinas de Vectores de Soporte - Iris Dataset",
       x = "Ancho del Sépalo", y = "Largo del Sépalo",
       color = "Especie Real") +
  scale_color_manual(values = c("red", "green"), labels = c("Versicolor", "No Versicolor")) +
  theme_minimal()

# Cargar el dataset iris
data(iris)

# Convertir la variable de respuesta a factor
iris$Species <- as.factor(iris$Species)

# Dividir el dataset en conjunto de entrenamiento y conjunto de prueba
set.seed(123)
indices <- sample(1:nrow(iris), nrow(iris)*0.7)
train_data <- iris[indices, ]
test_data <- iris[-indices, ]

# Ajustar el modelo KNN
library(class)
k <- 5  # Número de vecinos a considerar
modelo <- knn(train_data[, 1:4], test_data[, 1:4], train_data$Species, k)

# Calcular la precisión del modelo
precision <- sum(modelo == test_data$Species) / length(test_data$Species)
cat("Precisión del modelo:", precision, "\n")

# Graficar el KNN
library(ggplot2)
ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Species)) +
  geom_point() +
  geom_point(data = data.frame(test_data$Sepal.Length, test_data$Sepal.Width, modelo),
             aes(x = test_data$Sepal.Length, y = test_data$Sepal.Width),
             color = "blue", size = 4, alpha = 0.5) +
  labs(title = "K-Vecinos más Cercanos - Iris Dataset",
       x = "Largo del Sépalo", y = "Ancho del Sépalo",
       color = "Especie") +
  theme_minimal()



