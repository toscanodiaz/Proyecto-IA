# Proyecto-IA

# Predicción de desempeño de estudiantes

Este proyecto implementa distintos enfoques de Machine Learning para predecir el desempeño de estudiantes con base en sus hábitos de estudio y otras variables.  
Se utilizan tanto algoritmos manuales (gradient descent) como modelos de ensamble (random forest y bagging).

---

## Archivos

### 1. gradientdescent.py
Implementa regresión lineal implementada manualmente con gradient descent

- **Pasos principales:**
  - Preprocesamiento: escalado de variables y codificación de actividades extracurriculares
  - Implementación desde cero de la función de costo (MSE) y la actualización de pesos y sesgo
  - Entrenamiento con un número fijo de épocas y tasa de aprendizaje
  - Cálculo de métricas
  - Visualizaciones:
    - Valores reales vs predicciones
    - Residuos y su distribución
    - Evolución del costo durante el entrenamiento
    - Importancia de características (pesos aprendidos)

---

### 2. randomforest.py
Implementa un modelo de random forest utilizando scikit-learn

- **Pasos principales:**
  - Preprocesamiento con escalado y codificación de variables
  - Entrenamiento de random forest con 200 árboles
  - Evaluación con métricas de R² y MSE
  - Visualizaciones:
    - Valores reales vs predicciones
    - Residuos y su distribución
    - Importancia de características
    
---

### 3. bagging.py
Implementa un modelo de bagging regressor con árboles de decisión como estimadores base

- **Pasos principales:**
  - División en train/test/validation
  - Escalado con StandardScaler
  - Entrenamiento de un bagging regressor con 200 árboles.
  - Evaluación en los conjuntos train, test y validation
  - Visualizaciones:
    - Valores reales vs predicciones en test
    - Residuos y su distribución en test
    - Importancia promedio de características en todos los árboles


# Reporte --> TBA... 


---
