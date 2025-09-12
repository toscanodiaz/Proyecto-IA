import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
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

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    random_state=42
)
rf.fit(X_scaled, y)

y_pred = rf.predict(X_scaled)

r_squared = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)

print(f"R² del modelo: {r_squared:.4f}")
print(f"MSE del modelo: {mse:.4f}")
print(f"promedio real de Performance Index: {df['Performance Index'].mean():.2f}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# valores reales vs predicciones
axes[0, 0].scatter(y, y_pred, alpha=0.3, edgecolor='k')
axes[0, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
axes[0, 0].set_xlabel("valores reales")
axes[0, 0].set_ylabel("predicciones")
axes[0, 0].set_title("valores reales vs predicciones")

# residuos vs valores reales
residuos = y - y_pred
axes[0, 1].scatter(y, residuos, alpha=0.3, color="purple", edgecolor='k')
axes[0, 1].axhline(0, color='red', linestyle='--')
axes[0, 1].set_xlabel("valores reales")
axes[0, 1].set_ylabel("error (y - y_pred)")
axes[0, 1].set_title("residuos del modelo")

# histograma de residuos
axes[1, 0].hist(residuos, bins=30, color="skyblue", edgecolor="black")
axes[1, 0].set_xlabel("error")
axes[1, 0].set_ylabel("frecuencia")
axes[1, 0].set_title("distribución de errores de predicción")

# importancia de características
importancias = rf.feature_importances_
axes[1, 1].barh(features, importancias, color="teal", edgecolor="black")
axes[1, 1].set_xlabel("importancia")
axes[1, 1].set_title("importancia de características")

plt.tight_layout()
plt.show()
