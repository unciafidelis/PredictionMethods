# Módulo III: Métodos de predicción.

3.1 Modelos de regresión.
Los modelos de regresión son una técnica estadística utilizada para analizar la relación entre una variable dependiente y una o más variables independientes. Estos modelos buscan establecer una función matemática que describa la relación entre las variables y permita predecir el valor de la variable dependiente en función de los valores de las variables independientes.

## 3.1.1 Lineal.

La regresión lineal es utilizada cuando la variable dependiente es continua y se busca establecer una relación lineal entre esta variable y una o más variables independientes. El modelo de regresión lineal asume que la relación entre las variables se puede representar mediante una línea recta. El objetivo es encontrar los coeficientes que minimizan la diferencia entre los valores observados y los valores predichos por la línea de regresión.

## 3.1.2 Logística.
La regresión logística se utiliza cuando la variable dependiente es binaria o categórica y se busca predecir la probabilidad de que ocurra un evento o una categoría en particular. A diferencia de la regresión lineal, que se basa en una relación lineal, la regresión logística utiliza una función logística para modelar la relación entre las variables independientes y la probabilidad de ocurrencia del evento.

El objetivo es encontrar los coeficientes que maximizan la verosimilitud de los datos observados y, por lo tanto, estimen la probabilidad de ocurrencia del evento.

## 3.1.3 Polinomial.

La regresión polinomial se utiliza cuando la relación entre las variables no es lineal y puede ser mejor aproximada mediante un polinomio de grado superior. En este caso, se ajusta un modelo polinomial a los datos para capturar la relación no lineal.

El grado del polinomio (n) determina la flexibilidad del modelo y puede ajustarse según la complejidad de la relación no lineal.

<img src="https://github.com/unciafidelis/PredictionMethods/blob/main/Assets/Captura%20de%20pantalla%202023-07-05%20a%20la(s)%208.45.50.png" width="70%" height="70%" alt="Polinomial" /> 

http://sthda.com/english/articles/40-regression-analysis/162-nonlinear-regression-essentials-in-r-polynomial-and-spline-regression-models/ 

## 3.2 Modelos de clasificación. 
Los modelos de clasificación son técnicas y algoritmos utilizados para asignar instancias o ejemplos a diferentes categorías o clases. Estos modelos son ampliamente utilizados en el campo del aprendizaje automático (machine learning) y se aplican en diversos problemas, como la detección de spam en correos electrónicos, reconocimiento de imágenes, diagnóstico médico, detección de fraudes y muchas otras aplicaciones.

## 3.2.1 Árboles de decisión.
Los árboles de decisión son modelos de clasificación y regresión que utilizan una estructura en forma de árbol para representar decisiones y sus posibles resultados. Estos modelos son ampliamente utilizados en el campo del aprendizaje automático (machine learning) y son especialmente útiles cuando se requiere interpretabilidad y explicabilidad en el proceso de toma de decisiones.

Los árboles de decisión procuran dividir el conjunto de datos en subconjuntos más pequeños y homogéneos en términos de su variable objetivo (clase o valor a predecir). Cada nodo interno del árbol representa una prueba o condición en una variable, y cada rama representa el resultado de esa prueba. Las hojas del árbol representan las clases o valores predichos.

El proceso de construcción de un árbol de decisión se basa en la selección de características y la determinación de los puntos de división óptimos que maximicen la pureza o la reducción de la incertidumbre en cada nodo. Algunos algoritmos populares para construir árboles de decisión son el algoritmo ID3, el algoritmo C4.5 y el algoritmo CART.

Una vez que se ha construido el árbol de decisión, se puede utilizar para clasificar nuevas instancias o predecir valores continuos siguiendo el camino apropiado a través del árbol. En el caso de clasificación, se asigna una clase a una instancia siguiendo el camino desde la raíz hasta una de las hojas. En el caso de regresión, se predice un valor continuo en función del valor promedio de las instancias en la hoja correspondiente.

Los árboles de decisión ofrecen varias ventajas. Son fáciles de entender y visualizar, permiten capturar relaciones no lineales, manejan tanto características categóricas como numéricas y pueden ser utilizados para clasificación y regresión. Además, los árboles de decisión son menos susceptibles al ruido y los datos atípicos en comparación con otros modelos más complejos.

Sin embargo, los árboles de decisión también tienen algunas limitaciones. Pueden ser propensos al sobreajuste si no se controla su crecimiento, lo que significa que pueden adaptarse demasiado a los datos de entrenamiento y no generalizar bien a nuevos datos. Para mitigar este problema, se utilizan técnicas como la poda del árbol o el uso de ensamblados de árboles, como los Bosques Aleatorios (Random Forests) o Gradient Boosting.

## 3.2.2 Máquinas de vectores de soporte (SVM). 

El principio fundamental de las SVM es encontrar un hiperplano óptimo en un espacio de alta dimensión que separe de manera óptima las instancias de diferentes clases. El hiperplano se define como el límite de decisión que maximiza el margen entre las instancias de diferentes clases. El margen es la distancia perpendicular desde el hiperplano a las instancias más cercanas de cada clase y se busca maximizarlo para lograr una mejor generalización.

Las SVM pueden manejar datos linealmente separables utilizando un hiperplano lineal como límite de decisión. Sin embargo, también pueden manejar datos no linealmente separables mediante el uso de funciones de transformación no lineales, como el kernel trick. El kernel trick permite mapear los datos originales a un espacio de características de mayor dimensión donde sean linealmente separables, lo que permite la clasificación efectiva de datos complejos.

Algunos de los tipos de kernel comunes utilizados en las SVM son:

1. Kernel lineal: Se utiliza para problemas de clasificación linealmente separables. Es el kernel más simple y se basa en el producto escalar de los datos originales.

2. Kernel polinomial: Permite manejar relaciones no lineales mediante la aplicación de funciones polinomiales a los datos originales. El grado del polinomio es un parámetro ajustable.

3. Kernel RBF (Radial Basis Function): Es uno de los kernels más utilizados y permite manejar problemas no linealmente separables. Mapea los datos a un espacio de características infinito utilizando funciones de base radial y es adecuado para problemas con fronteras de decisión más complejas.

4. Kernel sigmoide: Se utiliza para problemas de clasificación binaria y se basa en la función sigmoide. También se puede utilizar para la clasificación multiclase mediante la técnica de One-vs-All.

Las SVM son conocidas por su capacidad de generalización y su resistencia al sobreajuste, lo que las hace adecuadas para problemas con conjuntos de datos pequeños o ruidosos. Sin embargo, también pueden ser computacionalmente costosas en términos de tiempo y recursos cuando se aplican a grandes conjuntos de datos.


## 3.2.3 K-vecinos más cercanos (KNN).


El algoritmo KNN se basa en la idea de que las instancias similares tienden a estar cerca unas de otras en el espacio de características. Para clasificar una nueva instancia, el algoritmo KNN busca los K ejemplos más cercanos en el conjunto de entrenamiento y asigna la clase más frecuente entre esos vecinos a la instancia de prueba.

El funcionamiento básico del algoritmo KNN se puede resumir en los siguientes pasos:

1. Calcular la distancia: Se calcula la distancia entre la instancia de prueba y todas las instancias en el conjunto de entrenamiento. La distancia puede ser medida utilizando diversas métricas, como la distancia euclidiana o la distancia Manhattan.

2. Seleccionar los vecinos: Se seleccionan los K ejemplos más cercanos a la instancia de prueba basándose en la distancia calculada. Estos ejemplos se denominan vecinos más cercanos.

3. Clasificación (para problemas de clasificación): Para clasificar la instancia de prueba, se toma la clase más frecuente entre los vecinos más cercanos. Esta clase se asigna a la instancia de prueba.

4. Regresión (para problemas de regresión): Para predecir el valor continuo de la instancia de prueba, se puede tomar el promedio de los valores objetivo de los vecinos más cercanos.

La elección del valor de K es un parámetro importante en el algoritmo KNN. Un valor demasiado pequeño de K puede llevar a una clasificación sensible al ruido y a valores atípicos, mientras que un valor demasiado grande puede llevar a una clasificación menos precisa y a una mayor complejidad computacional. La elección óptima de K depende del conjunto de datos y del problema específico.

El algoritmo KNN tiene algunas ventajas notables, como su simplicidad y facilidad de interpretación, su capacidad para manejar problemas con múltiples clases y su capacidad para capturar relaciones no lineales en los datos. Sin embargo, también tiene limitaciones, como su sensibilidad a la elección de K y a la distribución de los datos, y su ineficiencia en grandes conjuntos de datos.

EJEMPLOS Sencillos, hay varios tutoriales: 
https://tutoriales.edu.lat/pub/r/r-logistic-regression/r-regresion-logistica 
https://www.cienciadedatos.net/machine-learning-r.html
https://bookdown.org/content/2031/arboles-de-decision-parte-ii.html
http://fce.unal.edu.co/uifce-blog/2021/09/28/machine-learning-en-r/
https://fervilber.github.io/Aprendizaje-supervisado-en-R/arboles.html
https://gonzalezgouveia.com/modelo-de-clasificacion-con-arboles-de-decision-en-r-con-ejemplo-en-rstudio/ 

## 3.3 Métodos de validación y evaluación de modelos.
En el campo del machine learning, los métodos de validación y evaluación de modelos son fundamentales para comprender el desempeño y la eficacia de un modelo predictivo. Estos métodos nos permiten medir cómo se generaliza el modelo a datos no vistos y cómo se comportará en situaciones del mundo real. A continuación, describiré algunos de los métodos de validación y evaluación más comunes:

Validación cruzada (Cross-Validation): Este método implica dividir el conjunto de datos en múltiples subconjuntos o "pliegues" (folds) para entrenar y evaluar el modelo de manera repetida. El enfoque más común es la validación cruzada k-fold, donde los datos se dividen en k pliegues, y se realizan K iteraciones, utilizando cada pliegue como conjunto de prueba y el resto como conjunto de entrenamiento. Al final, se promedian las métricas de evaluación obtenidas en cada iteración.

<img src="https://github.com/unciafidelis/PredictionMethods/blob/main/Assets/Captura%20de%20pantalla%202023-07-05%20a%20la(s)%208.46.42.png" width="70%" height="70%" alt="CrossValidation" /> 

Holdout: En este enfoque, se divide el conjunto de datos en dos conjuntos distintos: un conjunto de entrenamiento y un conjunto de prueba. El modelo se entrena en el conjunto de entrenamiento y luego se evalúa en el conjunto de prueba. El tamaño de los conjuntos puede variar, pero es común usar un porcentaje alto para entrenamiento (por ejemplo, 80%) y el resto para pruebas.

<img src="https://github.com/unciafidelis/PredictionMethods/blob/main/Assets/Captura%20de%20pantalla%202023-07-05%20a%20la(s)%208.47.16.png" width="70%" height="70%" alt="CrossValidation" /> 

Validación en retención (Holdout Validation): En este método, se separa una porción del conjunto de datos para su validación mientras se entrena el modelo en el resto de los datos. Luego, se evalúa el modelo utilizando los datos de validación retenidos para obtener una estimación del rendimiento en datos no vistos. Es importante asegurarse de que los datos retenidos sean representativos de la distribución general de los datos.
NOTA: Existen diferentes maneras de separar un conjunto de datos.

<img src="https://github.com/unciafidelis/PredictionMethods/blob/main/Assets/Captura%20de%20pantalla%202023-07-05%20a%20la(s)%208.47.52.png" width="70%" height="70%" alt="CrossValidation" /> 

Validación cruzada estratificada (Stratified Cross-Validation): Este método es similar a la validación cruzada k-fold, pero se asegura de que la proporción de muestras de cada clase en los pliegues sea similar a la proporción en el conjunto de datos original. Es especialmente útil cuando se trabaja con conjuntos de datos desequilibrados, donde las clases tienen diferentes cantidades de muestras.

<img src="https://github.com/unciafidelis/PredictionMethods/blob/main/Assets/Captura%20de%20pantalla%202023-07-05%20a%20la(s)%208.48.24.png" width="70%" height="70%" alt="CrossValidation" /> 

Métricas de evaluación: Las métricas de evaluación se utilizan para medir el rendimiento del modelo y pueden variar dependiendo del tipo de problema. Algunas métricas comunes incluyen la precisión (accuracy), la precisión (precisión), la exhaustividad (recall), la puntuación F1 (F1-score), el área bajo la curva ROC (AUC-ROC) y el error cuadrático medio (mean squared error), entre otros.

<img src="https://github.com/unciafidelis/PredictionMethods/blob/main/Assets/Captura%20de%20pantalla%202023-07-05%20a%20la(s)%208.48.48.png" width="70%" height="70%" alt="CrossValidation" /> 

<img src="https://github.com/unciafidelis/PredictionMethods/blob/main/Assets/Captura%20de%20pantalla%202023-07-05%20a%20la(s)%208.49.14.png" width="20%" height="20%" alt="CrossValidation" /> 

Es importante tener en cuenta que la elección del método de validación y las métricas de evaluación dependen del problema específico y del tipo de datos que se esté utilizando. La elección adecuada de estos métodos puede ayudar a evaluar y comparar diferentes modelos, seleccionar el mejor modelo y comprender su rendimiento en situaciones del mundo real.

## 3.4 Selección de características y reducción de dimensionalidad.
La selección de características y la reducción de dimensionalidad son técnicas utilizadas en machine learning para mejorar el rendimiento de los modelos y facilitar el procesamiento de datos con alta dimensionalidad. Estas técnicas tienen como objetivo identificar las características más relevantes o reducir el número de dimensiones del conjunto de datos sin perder demasiada información. A continuación, se explica cada una de ellas:

Selección de características: Esta técnica implica identificar un subconjunto relevante de características del conjunto de datos original. Al seleccionar características relevantes, se busca reducir el ruido y eliminar características redundantes o poco informativas. Algunos métodos comunes para la selección de características son:

Filtro de características: Se utilizan medidas estadísticas o heurísticas para evaluar la importancia de cada característica individualmente. Se seleccionan las características que tienen una mayor correlación o relación con la variable objetivo.

Wrapper de características: Se utilizan algoritmos de aprendizaje automático para evaluar diferentes subconjuntos de características. Se buscan subconjuntos que maximicen el rendimiento del modelo en términos de una métrica específica.

Incorporación de características: Se utilizan técnicas de aprendizaje automático para incorporar características de manera incremental y evaluar su impacto en el rendimiento del modelo.

Reducción de dimensionalidad: Esta técnica se utiliza para disminuir el número de dimensiones en conjuntos de datos de alta dimensionalidad. Al reducir la dimensionalidad, se busca eliminar características irrelevantes o redundantes, y comprimir la información en un espacio de menor dimensión. Algunos métodos populares de reducción de dimensionalidad incluyen:

Análisis de componentes principales (PCA): Utiliza una transformación lineal para proyectar los datos en un nuevo espacio de menor dimensión que maximice la varianza de los datos proyectados. Las nuevas dimensiones son combinaciones lineales de las características originales.

Análisis discriminante lineal (LDA): Similar al PCA, pero se enfoca en maximizar la separabilidad entre las clases en lugar de la varianza total. Es especialmente útil en problemas de clasificación.

Selección de características basada en modelos: Se utilizan algoritmos de aprendizaje automático para seleccionar un subconjunto de características que son más relevantes para el modelo. Esto puede lograrse mediante técnicas como la eliminación recursiva de características (Recursive Feature Elimination, RFE) o la selección basada en árboles de decisión.

Técnicas de descomposición de matrices: Utilizan técnicas como descomposición en valores singulares (Singular Value Decomposition, SVD) o factorización de matrices no negativas (Non-Negative Matrix Factorization, NMF) para encontrar representaciones de menor dimensión de los datos.

La selección de características y la reducción de dimensionalidad son herramientas poderosas para mejorar el rendimiento de los modelos y abordar problemas de alta dimensionalidad. Sin embargo, es importante tener en cuenta que la elección de la técnica adecuada depende del problema específico, el conjunto de datos y los objetivos del análisis.


