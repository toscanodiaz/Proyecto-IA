# Proyecto-IA

Este proyecto aborda dos enfoques distintos para resolver problemas de regresión: uno implementado desde cero sin frameworks y otro utilizando _scikit-learn_.

---

## 1. Modelo sin frameworks – `gradientdescent.py`

- **Algoritmo:** Implementación propia de gradient descent aplicado a una regresión lineal.  
- **Dataset:** `Student_Performance.csv`  
  - Incluye variables académicas y hábitos de los estudiantes (horas de estudio, puntajes previos, actividades extracurriculares, etc.).  
  - Objetivo: predecir el **Performance Index**.  
- **Características principales:**  
  - Preprocesamiento básico (codificación de variables, escalamiento).  
  - Entrenamiento manual de pesos y bias con gradient descent.  
  - Cálculo de métricas: función de costo (MSE) y R².  
  - Visualización de la evolución del error durante el entrenamiento.

---

## 2. Modelo con frameworks – `randomforest.py`

- **Algoritmo:** Random Forest Regressor implementado con _scikit-learn_.  
- **Dataset:** `insurance.csv`  
  - Contiene información demográfica y de salud (edad, BMI, fumador, número de hijos, región, etc.).  
  - Objetivo: predecir el costo médico (**charges**).  
- **Pasos principales:**  
  - Preprocesamiento con _OneHotEncoder_ y _StandardScaler_.  
  - Entrenamiento de Random Forest con distintos hiperparámetros (`n_estimators`, `max_depth`, etc.).  
  - Evaluación en train/validation/test con métricas: **R², MAE, MSE, RMSE**.  
  - Visualizaciones:  
    - Valores reales vs. predicciones (parity plots).  
    - Análisis de residuales y su distribución.  
    - Calibración del modelo.  
    - Importancia de características.  
    - Evolución del error según el número de árboles.

---
