import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

file_path = 'Student_Performance.csv'
df = pd.read_csv(file_path)
print(df.head())

print("-- valores nulos por columna ---")
print(df.isnull().sum())

print("\n---- información ---")
df.info()

print("\n--- columnas a lista ---")
np.array(df.columns.to_list())
print(df.columns)

# encoding para columna categórica
df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes' : 1 , 'No' : 0})
df['Extracurricular Activities'].value_counts()

# drop valores nulos
df.dropna(inplace=True)

# seleccionar features y target
features = ['Hours Studied', 'Previous Scores', 'Sleep Hours', 'Sample Question Papers Practiced']
target = 'Performance Index'

X_df = df[features]
y_df = df[target]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_df)

y = y_df.values
y_scaled = (y - np.mean(y)) / np.std(y)

X_tr, X_val, y_tr, y_val = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

def gradient_descent(X, y, epochs=1000, lr=0.1, l2=0.0):
    n_features = X.shape[1]
    weights = np.zeros(n_features)
    bias = 0
    n = X.shape[0]
    cost_history = []
    for i in range(epochs):
        y_pred = np.dot(X, weights) + bias
        cost = (1/n) * np.sum((y - y_pred)**2)
        wd = -(2/n) * (X.T @ (y - y_pred)) + 2*l2*weights
        bd = -(2/n) * np.sum(y - y_pred)
        weights -= lr * wd
        bias    -= lr * bd
        cost_history.append(cost)
        
        print(f"epochs {i+1}: weights={weights}, bias={bias:.4f}, cost={cost:.4f}")
        
    return weights, bias, cost_history


final_weights, final_bias, cost_history = gradient_descent(
    X_tr, y_tr,
    epochs=800, lr=0.1, l2=0.0
)

y_pred_scaled = np.dot(X_scaled, final_weights) + final_bias
y_pred = y_pred_scaled * np.std(y) + np.mean(y)

sse = np.sum((y - y_pred) ** 2)
sst = np.sum((y - np.mean(y)) ** 2)
r_squared = 1 - (sse / sst)

print(f"R² del modelo: {r_squared:.4f}")
print(df['Performance Index'].mean())

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

# evolución del costo
axes[1, 1].plot(range(len(cost_history)), cost_history, color="blue", linewidth=2)
axes[1, 1].set_xlabel("épocas")
axes[1, 1].set_ylabel("costo (MSE)")
axes[1, 1].set_title("evolución del costo durante el entrenamiento")
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

# importancia de características
feature_importance = pd.Series(final_weights, index=features)
feature_importance.sort_values().plot(kind="barh", color="teal", edgecolor="black", figsize=(8,6))
plt.xlabel("peso del modelo (coeficiente)")
plt.title("importancia de características en la predicción")
plt.show()


