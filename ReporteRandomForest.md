# Random Forest: Insurance Dataset
## Descripción del problema 
El dataset [insurance.csv](https://www.kaggle.com/datasets/mirichoi0218/insurance) contiene información de asegurados, con variables demográficas y de estilo de vida como edad, sexo, índice de masa corporal (BMI), número de hijos, hábito de fumar y región de residencia. 
La variable objetivo charges representa el costo del seguro médico para cada persona.
El objetivo de este proyecto es predecir los costos médicos (charges) a partir de las variables disponibles, utilizando un modelo de Random Forest Regressor.

### Random Forest
Random Forest es un algoritmo de aprendizaje basado en árboles de decisión que consiste en entrenar muchos árboles de manera independiente, cada uno usando una muestra aleatoria de los datos y de las variables (técnica llamada bagging). 
Las predicciones finales se obtienen combinando los resultados de todos los árboles (en regresión se promedian y en clasificación se toma la mayoría), lo cual permite reducir la varianza de los modelos individuales, mejorar la estabilidad y aumentar la capacidad de generalización.
Un árbol de decisión puede ser inestable porque cambia mucho si cambian los datos, pero un bosque de árboles, Random Forest, es mucho más robusto y preciso.

### Uso de Random Forest en este proyecto
- **Datos mixtos**:
el dataset contiene variables categóricas (sex, smoker, region) y numéricas (age, bmi, children).
En Random Forest cada árbol puede tomar decisiones basadas en cualquier tipo de variable.
- **No linealidad**:
la relación entre variables (fumar, edad, BMI) y charges no es lineal, pero Random Forest puede capturar relaciones complejas sin necesidad de transformar los datos.
- **Reducción de sobreajuste**:
Random Forest reduce la varianza al combinar múltiples árboles lo cual logra mejor generalización.
- **Desempeño**:
para este proyecto, Random Forest alcanzó un R² de 0.87 en test, explicando la mayor parte de la varianza de los costos médicos, con errores relativamente bajos frente al rango de valores.

### Principios éticos aplicados 

- **Privacidad y confidencialidad de datos**
    - El dataset se usó únicamente con fines académicos, sin exponer información personal identificable.
    - Los datos en cualquier aplicación real deben manejarse de acuerdo con normas como GDPR (Europa) o HIPAA (EE.UU.), las cuales protegen datos de salud. 

- **No discriminación y sesgos**
    - El dataset incluye variables como sexo, región y hábito de fumar lo cual podría generar predicciones sesgadas si no se interpretan correctamente.
    - En el análisis se enfatizó la importancia de interpretar la influencia de estas variables con cuidado, evitando justificar discriminación en precios de seguros.

- **Transparencia y explicabilidad**
    - Se calcularon y se analizaron importancias de variables, mostrando que `smoker`, `bmi` y `age` son los principales factores en el costo médico.
    - Esto facilita explicar al usuario cómo y por qué el modelo llega a sus predicciones.

- **Uso responsable**
    - Los modelos se entrenaron y evaluaron en conjuntos de train, validación y test, evitando conclusiones falsas por sobreajuste.
    - El trabajo muestra escenarios de underfitting, overfitting y ajuste adecuado, destacando la importancia de no usar un modelo ciego en producción.

---

## Preprocesamiento de datos
### Exploración inicial
- Total de registros: 1338
- Variables categóricas: `sex`, `smoker`, `region`
- Variables numéricas: `age`, `bmi`, `children`, `charges`
- Valores nulos: ninguno
- La correlación muestra que `smoker` y `bmi` tienen una fuerte relación con `charges`.

### Codificación de variables categóricas
Columnas como **sex**, **smoker**, **region** se transformaron en variables binarias (0/1), permitiendo que el modelo interprete categorías sin imponer un orden artificial.

### Escalamiento
Variables como `age`, `bmi` y `children` presentan rangos diferentes. 
Se aplicó _StandardScaler_ para normalizarlas y mantenerlas en la misma escala, con media 0 y desviación estándar 1, mejorando la estabilidad del modelo y evitando que variables con valores grandes dominen el entrenamiento.

### Imputación 
El dataset no presentaba valores nulos, pero se verificó con `df.isnull().sum()`.
En un caso real sería necesario imputar valores faltantes (media, mediana u otras técnicas).

### EDA (detección de anomalías)
La variable `charges` tiene valores extremos (outliers > 50,000).
Se decidió mantenerlos porque reflejan casos reales: fumadores con alto BMI y edad avanzada. 
Estos outliers son importantes para el negocio porque afectan costos de aseguradoras.

### División de datos 
- **Train**: 60%
- **Validation**: 20%
- **Test**: 20%

---

# Visualización de los datos
Antes de entrenar el modelo de Random Forest, se realizó un análisis exploratorio de los datos con el objetivo de comprender mejor la naturaleza de las variables y su relación con el costo de los seguros médicos (charges).

### Variables categóricas
<img width="1919" height="1056" alt="categoricas" src="https://github.com/user-attachments/assets/e0b6553c-221e-4297-8237-6c0b44ef39da" />

En las gráficas de barras se observa la distribución de las categorías **smoker**, **children**, **sex** y **region**.

Este análisis permite identificar desequilibrios en los datos, como el predominio de personas no fumadoras sobre fumadoras o la distribución relativamente uniforme de las regiones.

Estas variables requieren un proceso de codificación (_OneHotEncoding_) para ser utilizadas por el modelo.

### Variables numéricas
<img width="1919" height="1032" alt="numericas" src="https://github.com/user-attachments/assets/825646c8-83cf-4dc8-a937-cb6331096b93" />

En los histogramas se muestran las distribuciones de **age**, **BMI** y **charges**.

Se observa por ejemplo que la edad está concentrada entre los 20 y 50 años, el BMI sigue una distribución cercana a la normal con ligera cola hacia valores altos, y los cargos presentan asimetría hacia la derecha (skewness), lo cual refleja la existencia de valores extremos en personas con altos costos médicos.

Este análisis justifica la aplicación de escalamiento y la detección de outliers.

### Matriz de dispersión (pairplot)
<img width="1919" height="1053" alt="dispersion" src="https://github.com/user-attachments/assets/905cd8b4-23e5-4327-8515-a48945155355" />

La visualización conjunta de las variables permite ver tendencias relevantes, como la fuerte relación entre **smoker** y **charges**, donde las personas fumadoras tienden a presentar cargos más altos.

También se aprecian relaciones no lineales entre **age**, **BMI** y los cargos, lo que motiva el uso de un modelo no lineal y flexible como Random Forest.

---

# Entrenamiento del modelo
El modelo de Random Forest se entrenó con los siguientes parámetros:
- `max_depth = 8`
- `n_estimators = 150`
- `random_state = 42`
Se eligieron estos valores después de comparar distintas configuraciones, desde modelos simples con `max_depth = 2` hasta muy complejos con `max_depth = 1000`.

---

# Resultados del modelo
Una vez realizado el análisis exploratorio y el preprocesamiento de los datos, se procedió a la etapa de aprendizaje automático mediante el modelo de Random Forest Regressor.
El entrenamiento se planteó en diferentes iteraciones con variaciones en los hiperparámetros, con el objetivo de observar cómo influyen en el desempeño del modelo y diagnosticar fenómenos como sesgo (bias), varianza y nivel de ajuste (underfit, fit, overfit).

Cada iteración se evaluó en tres conjuntos: entrenamiento (train), validación (validation) y prueba (test), lo que permitió analizar la estabilidad del modelo y evitar conclusiones basadas únicamente en un solo subconjunto de datos.
También se calcularon métricas como R², MAE, MSE y RMSE y gráficas de error por iteración y análisis de residuales.

## Primera iteración
### Hiperparámetros
- `max_depth = 2`
- `n_estimators = 50`
- `random_state = 42`

### Resultados
- **R² Train**: 0.8326
- **R² Validation**: 0.8113
- **R² Test**: 0.8422
- **MAE Test**: 3139.7
- **MSE Test**: 24,504,651.0
- **RMSE Test**: 4950.2

### Evaluación del modelo
**Diagnóstico**
- **Bias**: medio–alto → el modelo no logra capturar la complejidad de la relación entre variables y charges pues se limita con una profundidad de árbol muy baja.
- **Varianza**: baja → los valores de R² entre train, validación y test son muy parecidos, indicando que el modelo es estable pero poco expresivo.
- **Nivel de ajuste**: underfitting → el modelo está demasiado simple, no extrae suficiente información de los datos y por eso los errores (RMSE: 4950) siguen siendo altos.

### Gráficas 
**Parity Plot**
<img width="1919" height="1040" alt="parityplot" src="https://github.com/user-attachments/assets/b13420fc-17ee-4545-be69-de79914eb287" />
- Los puntos deberían estar alineados a la diagonal roja (predicho = real).
- Se observa una alta dispersión, con grupos planos de predicciones (efecto de la poca profundidad de los árboles).
- Indica que el modelo no diferencia bien entre valores intermedios y termina prediciendo rangos muy generales.

**Residuales vs. predicho**
<img width="1919" height="1042" alt="residualesvspredicho" src="https://github.com/user-attachments/assets/1ef194e4-e961-4b67-9033-6395c97c9a4b" />
- Los residuales deberían estar distribuidos aleatoriamente alrededor de 0.
- En este caso se ven patrones verticales (valores repetidos de predicción), indicando un modelo poco flexible.
- No hay forma clara de abanico (heterocedasticidad) pero sí errores grandes en valores altos de charges.

**Histogramas de residuales**
<img width="1919" height="1041" alt="histogramasresiduales" src="https://github.com/user-attachments/assets/2d59144f-1431-43de-89b7-c351f95714c1" />
- Deben estar centrados en 0 y simétricos.
- Aparecen colas largas hacia la derecha → el modelo subestima costos altos (ej. fumadores con BMI elevado).

**Calibración**
<img width="1919" height="1050" alt="calibracion" src="https://github.com/user-attachments/assets/7f4bbfe7-4986-45ae-9930-00b1b041f384" />
- La línea azul ajustada tiene pendiente <1 (0.78 en validación y 0.81 en test).
- Esto significa que el modelo tiende a subestimar valores altos y sobreestimar valores bajos.

**Evolución del error**
<img width="1715" height="982" alt="error" src="https://github.com/user-attachments/assets/60ae9367-dcdb-4bc6-bdc1-36527a168009" />
- Se observa que el error disminuye con más árboles pero se estabiliza rápido.
- Agregar árboles no mejora el rendimiento pues la profundidad máxima de cada árbol es muy baja. 

### Mejoras
- Aumentar max_depth para permitir que los árboles capturen relaciones más complejas.
- Incrementar n_estimators para mejorar estabilidad, aunque en este caso el limitante es la profundidad.

### Resumen
- En la primera iteración se produjo un modelo subajustado, lo cual indica que es estable pero poco potente. 
- Explica un 84% de la varianza en test con errores significativos en predicciones altas.
- Las gráficas refuerzan esta conclusión: dispersión elevada en parity plots, residuales con patrones claros y pendiente de calibración <1.
- Es necesario aumentar la complejidad del modelo (más profundidad, más árboles) para capturar mejor la relación entre variables y charges.

## Segunda iteración
### Hiperparámetros
- `max_depth = None`
- `n_estimators = 1000`
- `random_state = 42`

### Resultados
- **R² Train**: 0.9780
- **R² Validation**: 0.8221
- **R² Test**: 0.8670
- **MAE Test**: 2408.87
- **MSE Test**: 20,647,542.25
- **RMSE Test**: 4543.96

### Evaluación del modelo
**Diagnóstico**
- **Bias**: bajo → el modelo ahora captura bien las relaciones no lineales.
- **Varianza**: media → el aumento de profundidad y número de árboles genera una mayor sensibilidad a los datos de entrenamiento, pero aún se controla gracias al ensamble.
- **Nivel de ajuste**: fit con tendencia a overfit → el R² en Train es muy superior al de Validation, aunque no extremo.

* Ahora el modelo logra un mejor balance pues explica más varianza y reduce errores, pero mueestra indicios de sobreajuste leve.

### Gráficas 
**Parity Plot**
<img width="1919" height="1052" alt="parityplot" src="https://github.com/user-attachments/assets/80c71c3f-ff61-4b08-8ddd-1dedb40d101e" />

- Train: los puntos se alinean casi perfectamente con la diagonal indicando un muy buen ajuste en entrenamiento.
- Validation/Test: los puntos están cercanos a la diagonal con dispersión moderada en valores altos, lo cual representa una mejora clara respecto a la primera iteración.

**Residuales vs. predicho**
<img width="1919" height="1033" alt="residualesvspredicho" src="https://github.com/user-attachments/assets/951502d9-87c1-4092-8f63-c8d25386375f" />

- Distribuidos alrededor de cero con menor sesgo sistemático que antes.
- Aún hay algunos outliers (valores altos de charges).

**Histogramas de residuales**
<img width="1919" height="1029" alt="histogramasresiduales" src="https://github.com/user-attachments/assets/73cc7d97-aa25-42fc-8321-3632fbd8a4aa" />

- Centrados en cero y más compactos que en la primera iteración.
- Aún se ven colas pesadas (outliers).

**Calibración**
<img width="1919" height="1038" alt="calibracion" src="https://github.com/user-attachments/assets/8f66f324-c107-4750-9364-1b30f6f24cea" />

- Pendientes cercanas a 0.85–0.89 → el modelo tiende a subestimar valores altos de charges aunque mucho menos que en la primera iteración.

**Evolución del error**
<img width="1919" height="1044" alt="error" src="https://github.com/user-attachments/assets/7250422c-f90d-42b9-a775-a88182825e0b" />

- El RMSE en Train, Validation y Test converge y se estabiliza después de aproximadamente 200 árboles.
- Esto confirma que al aumentar los árboles de 50 a 1000 se redujo el error y mejoró la estabilidad del modelo.

### Mejoras
- Limitar `max_depth` para reducir el riesgo de overfitting. 

### Resumen
- El modelo alcanzó un R² de 98% en Train, mostrando una capacidad muy alta para explicar la varianza en el conjunto de entrenamiento.
- En Validation (82%) y Test (86%) los valores de R² se mantienen sólidos, aunque con una caída respecto a Train, lo que evidencia cierto sobreajuste pero dentro de un rango controlado.
- Los errores absolutos y cuadrados (MAE y RMSE) disminuyeron significativamente en comparación con la primera iteración (MAE = 3139 → 2408 y RMSE = 4950 → 4544).
- Esto indica que el modelo no solo explica mejor la varianza, sino que también genera predicciones más cercanas a los valores reales.

* La segunda iteración supera a la primera en capacidad predictiva y reducción de errores aunque introduce un pequeño riesgo de sobreajuste.

## Tercera iteración
### Hiperparámetros
- `max_depth = 8`
- `n_estimators = 150`
- `random_state = 42`

### Resultados
- **R² Train**: 0.9589
- **R² Validation**: 0.8276
- **R² Test**: 0.8725
- **MAE Test**: 2365.19
- **MSE Test**: 19,791,884.38
- **RMSE Test**: 4448.80

### Evaluación del modelo
**Diagnóstico**
- **Bias**: medio → el sesgo se redujo respecto a la primera iteración pero sigue presente en comparación con el ajuste casi perfecto de entrenamiento.
- **Varianza**: media → se redujo en comparación con la segunda iteración ya que el modelo dejó de sobreajustar con `max_depth = None`.
- **Nivel de ajuste**: fit → el modelo encuentra un punto intermedio pues no está tan limitado como en la primera iteración, ni tan sobreajustado como en la segunda.

### Gráficas 
**Parity Plot**
<img width="1919" height="1040" alt="parityplot" src="https://github.com/user-attachments/assets/c703848f-639d-4ab3-b7a9-065a7234699a" />

- La nube de puntos se alinea mejor a la diagonal ideal respecto a la primera y segunda iteración.
- En validación y test aún se observan algunas desviaciones en los valores altos, pero en general el ajuste es más equilibrado.

**Residuales vs. predicho**
<img width="1919" height="1044" alt="residualesvspredicho" src="https://github.com/user-attachments/assets/b11135e9-675d-4434-bbc0-dfec60cf41b1" />

- La dispersión es más homogénea que en la primera iteración.
- Se aprecia menos estructura que represente un sesgo sistemático.
- Los residuos tienden a crecer en magnitud para valores altos pero en menor medida que en iteraciones previas.

**Histogramas de residuales**
<img width="1919" height="1037" alt="histogramasresiduales" src="https://github.com/user-attachments/assets/53fe21b0-6a8c-4a27-ab3d-f9ed69efe880" />

- Distribuciones más centradas en cero y relativamente simétricas.
- Menos outliers extremos que en la primera iteración.
- Eso confirma que el modelo ya no tiene un sesgo fuerte.

**Calibración**
<img width="1919" height="1043" alt="calibracion" src="https://github.com/user-attachments/assets/d8f3acbd-4001-41ef-b03a-0c6b8679bc12" />

- La pendiente mejora (validation: 0.85 y test: 0.89) acercándose más a 1.
- El modelo ya no subestima tanto los valores altos como en la primera y segunda iteración.

**Evolución del error**
<img width="1919" height="1041" alt="error" src="https://github.com/user-attachments/assets/9cab2a01-c715-42e4-9350-2fc4cc3104cc" />

- La curva se estabiliza más rápido (con alrededor de 50–100 árboles).
- Validación y test muestran un descenso consistente y luego convergencia estable sin señales de sobreajuste.
- Es la mejor dinámica de las tres iteraciones.

### Mejoras
- `min_samples_split` o `min_samples_leaf` para suavizar árboles.
- Optimizar parámetros con _GridSearch_ o _RandomizedSearch_.
- Incluir interacciones (ejemplo: smoker * bmi) que podrían capturar relaciones no lineales clave.
- Cross validation (_k-fold_) → confirmar que la mejora se sostiene en diferentes particiones.

### Resumen
- El modelo alcanza un R² Test de 0.8725, mejor que la primera (0.8421) y la segunda iteración (0.8670).
- El RMSE Test baja a 4448 mostrando la mejor capacidad predictiva hasta ahora.
- El MAE Test también mejora respecto a las iteraciones previas (3139 → 2408 → 2365).
- Esto indica un balance entre complejidad y generalización pues aunque el R² de entrenamiento es menor que el de la segunda iteración, el desempeño en validación y prueba mejora.

La tercera iteración ofrece el mejor balance hasta ahora, reduciendo tanto el error como la varianza. 
Representa una clara mejora frente al underfitting de la primera y al overfitting de la segunda. 
El modelo se acerca a un ajuste óptimo con predicciones más estables y generalizables.

---

# Conclusiones generales del proyecto 

Después de realizar las distintas configuraciones de hiperparámetros en el momdelo, se tienen estos hallazgos:

### Resultados finales 
- La primera iteración (profundidad muy baja) mostró underfitting → el modelo fue estable pero no pudo capturar la complejidad de los datos.
- La segunda iteración (profundidad ilimitada y muchos árboles) presentó un caso de overfitting parcial → excelente en entrenamiento pero con un gap marcado respecto a validación y prueba.
- La tercera iteración (max_depth=8, n_estimators=150) alcanzó el mejor balance entre bias y varianza:
    - R² Train: 0.959
    - R² Validation: 0.828
    - R² Test: 0.873
    - RMSE Test: ≈ 4449
    - MAE Test: ≈ 2365

* Esto significa que el modelo explica alrededor del 87% de la variabilidad de los costos médicos con un error medio absoluto de aproximadamente 2365 dólares, lo cual es razonable considerando que los costos alcanzan valores superiores a 63,000 dólares.

### Variables más influyentes
<img width="1919" height="1058" alt="featureimportance" src="https://github.com/user-attachments/assets/6031a8fd-e73f-41fd-92e4-f2c2c6e23575" />

El análisis de importancia de características (feature importance) del Random Forest muestra que las variables con mayor peso en la predicción de charges son:
- `smoker` (yes/no): es la más determinante → fumar eleva fuertemente el costo del seguro.
- `bmi` (índice de masa corporal): valores altos se asocian a mayores costos médicos.
- `age` (edad): a mayor edad mayor probabilidad de costos médicos altos.
- `children`: el número de hijos también influye pero en menor medida.
- `sex` y `region`: tienen impacto marginal aunque añaden diversidad al modelo.

La configuración final alcanzó un modelo robusto y generalizable, capaz de predecir los costos médicos con buena precisión y de identificar las variables más relevantes que afectan al target.
Este proyecto no solo permitió utilizar Random Forest para problemas de regresión con datos mixtos (categóricos y numéricos), sino también logró explicar que los hábitos de salud, el estado físico y la edad son los factores más determinantes en los costos de un seguro médico.

---

### Referencias 

- Choi, M. (2018). Medical Cost Personal Dataset (insurance.csv). Kaggle.
https://www.kaggle.com/datasets/mirichoi0218/insurance
- Shivam Singh (2023). Insurance Cost Prediction | EDA + ML (R² = 0.87). Kaggle Notebook.
https://www.kaggle.com/code/shivams811/insurance-cost-prediction-eda-ml-r-0-87
- Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.
https://link.springer.com/article/10.1023/A:1010933404324
- Scikit-learn Developers. Random Forest Regressor Documentation.
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
- European Union (2018). General Data Protection Regulation (GDPR).
https://gdpr.eu/
- U.S. Department of Health & Human Services. HIPAA for Professionals.
https://www.hhs.gov/hipaa/for-professionals/index.html
