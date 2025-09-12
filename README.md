# Proyecto-IA

# Predicción de Performance de Estudiantes

Este proyecto implementa distintos enfoques de **Machine Learning** para predecir el **Performance Index** de estudiantes en base a sus hábitos de estudio y otras variables.  

Se utilizan tanto algoritmos manuales (descenso de gradiente) como modelos de ensamble (Random Forest y Bagging).

---

## Archivos principales

### 1. `gradientdescent.py`
Implementa **Regresión Lineal** entrenada manualmente con **descenso de gradiente**.

- **Pasos principales:**
  - Preprocesamiento: escalado de variables y codificación de actividades extracurriculares.
  - Implementación desde cero de la función de costo (MSE) y la actualización de pesos y sesgo.
  - Entrenamiento con un número fijo de épocas y tasa de aprendizaje.
  - Cálculo de métricas (R²).
  - Visualizaciones:
    - Valores reales vs predicciones.
    - Residuos y su distribución.
    - Evolución del costo durante el entrenamiento.
    - Importancia de características (pesos aprendidos).

- **Objetivo:** entender cómo funciona el descenso de gradiente para regresión lineal sin usar librerías de alto nivel.

---

### 2. `randomforest.py`
Implementa un modelo de **Random Forest Regressor** utilizando `scikit-learn`.

- **Pasos principales:**
  - Preprocesamiento con escalado y codificación de variables.
  - Entrenamiento de un Random Forest con 200 árboles.
  - Evaluación con métricas de R² y MSE.
  - Visualizaciones:
    - Valores reales vs predicciones.
    - Residuos y su distribución.
    - Importancia de características (basada en árboles).

- **Objetivo:** usar un modelo de ensamble basado en árboles para mejorar la predicción y reducir el sobreajuste.

---

### 3. `bagging.py`
Implementa un modelo de **Bagging Regressor** con árboles de decisión como estimadores base.

- **Pasos principales:**
  - División en **train / validation / test**.
  - Escalado con `StandardScaler`.
  - Entrenamiento de un Bagging Regressor con 200 árboles.
  - Evaluación en los tres conjuntos (train, validation, test).
  - Visualizaciones:
    - Valores reales vs predicciones en test.
    - Residuos y su distribución en test.
    - Importancia promedio de características en todos los árboles.

- **Objetivo:** mostrar cómo Bagging mejora la estabilidad y generalización en comparación con un único árbol.

---

## Dependencias
El proyecto utiliza las siguientes librerías de Python:

```bash
numpy
pandas
scikit-learn
matplotlib
