import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

file_path = 'Student_Performance.csv'
df = pd.read_csv(file_path)

# encoding de columna categórica
df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})

# drop valores nulos
df.dropna(inplace=True)

# seleccionar features y target
features = ['Hours Studied', 'Previous Scores', 'Sleep Hours', 'Sample Question Papers Practiced']
target = 'Performance Index'

X = df[features]
y = df[target]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Escalado (usamos scaler ajustado en train y aplicado en val/test)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

bagging = BaggingRegressor(
    estimator=DecisionTreeRegressor(),
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

bagging.fit(X_train_scaled, y_train)

y_train_pred = bagging.predict(X_train_scaled)
y_val_pred = bagging.predict(X_val_scaled)
y_test_pred = bagging.predict(X_test_scaled)

def evaluar(y_real, y_pred, nombre):
    r2 = r2_score(y_real, y_pred)
    mse = mean_squared_error(y_real, y_pred)
    print(f"{nombre} -> R²: {r2:.4f}, MSE: {mse:.4f}")

evaluar(y_train, y_train_pred, "train")
evaluar(y_val, y_val_pred, "validation")
evaluar(y_test, y_test_pred, "test")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# valores reales vs predicciones
axes[0, 0].scatter(y_test, y_test_pred, alpha=0.3, edgecolor='k')
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
axes[0, 0].set_xlabel("valores reales")
axes[0, 0].set_ylabel("predicciones")
axes[0, 0].set_title("valores reales vs predicciones (test)")

# residuos vs valores reales
residuos = y_test - y_test_pred
axes[0, 1].scatter(y_test, residuos, alpha=0.3, color="purple", edgecolor='k')
axes[0, 1].axhline(0, color='red', linestyle='--')
axes[0, 1].set_xlabel("valores reales")
axes[0, 1].set_ylabel("error (y - y_pred)")
axes[0, 1].set_title("residuos del modelo (test)")

# histograma de residuos
axes[1, 0].hist(residuos, bins=30, color="skyblue", edgecolor="black")
axes[1, 0].set_xlabel("error")
axes[1, 0].set_ylabel("frecuencia")
axes[1, 0].set_title("distribución de errores de predicción (test)")

# importancia de características
importancias = np.mean([tree.feature_importances_ for tree in bagging.estimators_], axis=0)
axes[1, 1].barh(features, importancias, color="teal", edgecolor="black")
axes[1, 1].set_xlabel("importancia")
axes[1, 1].set_title("importancia de características")

plt.tight_layout()
plt.show()
