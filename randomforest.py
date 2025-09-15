import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import sqrt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import warnings 
warnings.filterwarnings('ignore')

file_path = 'insurance.csv'
df = pd.read_csv(file_path)
print(df.head())
print(df.shape)
print("-- INFO ---")
print(df.info())
print("-- DESCRIBE ---")
print(df.describe())
print("-- NULL ---")
print(df.isnull().sum())
print("-- UNIQUE ---")
print(df.nunique())

"""
# distribuciones numéricas y categóricas

custom_colors = ["#b5179e", "#fb6f92"] 
plt.figure(figsize=(12, 10))  

plt.subplot(2, 2, 1)
sns.countplot(x='smoker', data=df, palette=custom_colors)
plt.title("Distribución Smoker")

plt.subplot(2, 2, 2)
sns.countplot(x='children', data=df, palette=custom_colors)
plt.title("Distribución Children")

plt.subplot(2, 2, 3)
sns.countplot(x='sex', data=df, palette=custom_colors)
plt.title("Distribución Sex")

plt.subplot(2, 2, 4)
sns.countplot(x='region', data=df, palette=custom_colors)
plt.title("Distribución Region")

plt.tight_layout()
plt.suptitle("Distribución de features", fontsize=16, y=1.03)  
plt.show()

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
sns.histplot(df['age'], kde=True, color=custom_colors[0])
plt.title("Distribución Age")

plt.subplot(1, 3, 2)
sns.histplot(df['bmi'], kde=True, color=custom_colors[1])
plt.title("Distribución BMI")

plt.subplot(1, 3, 3)
sns.histplot(df['charges'], kde=True, color=custom_colors[0])
plt.title("Distribución Charges")

plt.tight_layout()
plt.show()

sns.pairplot(df, hue='smoker', palette=custom_colors)
plt.show()
"""

print(df.corr(numeric_only=True)['charges'])

X = df.drop(columns=['charges'])
y = df[['charges']]

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=42)

ohe = OneHotEncoder(sparse_output=False)
ohe.fit(X_train[['sex','smoker','region']])
encoded_columns = ohe.get_feature_names_out(['sex','smoker','region'])

def transform(df):
    encoded = pd.DataFrame(ohe.transform(df[['sex','smoker','region']]),
                           columns=encoded_columns, index=df.index)
    return pd.concat([df.drop(columns=['smoker','sex','region']), encoded], axis=1)

X_train_f = transform(X_train)
X_val_f   = transform(X_val)
X_test_f  = transform(X_test)

st = StandardScaler()
X_train_s = st.fit_transform(X_train_f)
X_val_s   = st.transform(X_val_f)
X_test_s  = st.transform(X_test_f)

rf = RandomForestRegressor(max_depth=8, n_estimators=150, random_state=42)
rf.fit(X_train_s, y_train)

y_train_pred = rf.predict(X_train_s)
y_val_pred   = rf.predict(X_val_s)
y_test_pred  = rf.predict(X_test_s)

# parity plot 
plt.figure(figsize=(15,5))

for i, (y_true, y_pred, title) in enumerate([
    (y_train, y_train_pred, "Train"),
    (y_val,   y_val_pred,   "Validation"),
    (y_test,  y_test_pred,  "Test")
]):
    plt.subplot(1,3,i+1)
    plt.scatter(y_true, y_pred, alpha=0.5, edgecolor="k")
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()],
             "r--", lw=2)
    plt.xlabel("Precio real")
    plt.ylabel("Precio predicho")
    plt.title(f"Parity plot - {title}")

plt.tight_layout()
plt.show()

# residuales vs predicho 
plt.figure(figsize=(12,5))
for i, (y_true, y_pred, title) in enumerate([
    (y_val, y_val_pred, "Validation"),
    (y_test, y_test_pred, "Test")
]):
    residuales = y_true.values.ravel() - y_pred
    plt.subplot(1,2,i+1)
    plt.scatter(y_pred, residuales, alpha=0.5, edgecolor="k", color="purple")
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicho")
    plt.ylabel("Residual (real - pred)")
    plt.title(f"Residuales vs Predicho - {title}")

plt.tight_layout()
plt.show()


#histogramas de residuales
plt.figure(figsize=(12,5))
for i, (y_true, y_pred, title) in enumerate([
    (y_val, y_val_pred, "Validation"),
    (y_test, y_test_pred, "Test")
]):
    residuales = y_true.values.ravel() - y_pred
    plt.subplot(1,2,i+1)
    sns.histplot(residuales, bins=30, kde=True, color="skyblue", edgecolor="black")
    plt.axvline(0, color="red", linestyle="--")
    plt.xlabel("Residual")
    plt.ylabel("Frecuencia")
    plt.title(f"Histograma de residuales - {title}")

plt.tight_layout()
plt.show()


# calibración
from sklearn.linear_model import LinearRegression

plt.figure(figsize=(12,5))
for i, (y_true, y_pred, title) in enumerate([
    (y_val, y_val_pred, "Validation"),
    (y_test, y_test_pred, "Test")
]):
    plt.subplot(1,2,i+1)
    plt.scatter(y_true, y_pred, alpha=0.5, edgecolor="k")
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()],
             "r--", lw=2)
    reg = LinearRegression().fit(y_true.values.reshape(-1,1), y_pred)
    y_line = reg.predict(y_true.values.reshape(-1,1))
    plt.plot(y_true, y_line, color="blue", lw=2)
    plt.xlabel("Real")
    plt.ylabel("Predicho")
    plt.title(f"Calibración - {title}\nPendiente={reg.coef_[0]:.2f}, Intercepto={reg.intercept_:.2f}")

plt.tight_layout()
plt.show()

print("\nRandom Forest")
print("R² Train      :", r2_score(y_train, y_train_pred))
print("R² Validation :", r2_score(y_val,   y_val_pred))
print("R² Test       :", r2_score(y_test,  y_test_pred))
print("MAE Test      :", mean_absolute_error(y_test, y_test_pred))
print("MSE Test      :", mean_squared_error(y_test, y_test_pred))
print("RMSE Test     :", np.sqrt(mean_squared_error(y_test, y_test_pred)))

train_errors, val_errors, test_errors = [], [], []

# predicciones acumuladas
pred_train = np.zeros(len(y_train))
pred_val   = np.zeros(len(y_val))
pred_test  = np.zeros(len(y_test))

for i, tree in enumerate(rf.estimators_, start=1):
    pred_train += tree.predict(X_train_s)
    pred_val   += tree.predict(X_val_s)
    pred_test  += tree.predict(X_test_s)

    avg_train = pred_train / i
    avg_val   = pred_val / i
    avg_test  = pred_test / i

    train_rmse = sqrt(mean_squared_error(np.ravel(y_train), avg_train))
    val_rmse   = sqrt(mean_squared_error(np.ravel(y_val),   avg_val))
    test_rmse  = sqrt(mean_squared_error(np.ravel(y_test),  avg_test))

    train_errors.append(train_rmse)
    val_errors.append(val_rmse)
    test_errors.append(test_rmse)

plt.figure(figsize=(10,6))
plt.plot(range(1, len(rf.estimators_)+1), train_errors, label="Train RMSE", color="blue")
plt.plot(range(1, len(rf.estimators_)+1), val_errors, label="Validation RMSE", color="red")
plt.plot(range(1, len(rf.estimators_)+1), test_errors, label="Test RMSE", color="green")
plt.xlabel("Número de árboles (n_estimators)")
plt.ylabel("Error (RMSE)")
plt.title("Evolución del error en Random Forest")
plt.legend()
plt.grid(True)
plt.show()

# feature importance
importances = rf.feature_importances_
features = X_train_f.columns

feat_imp = pd.DataFrame({'Variable': features, 'Importancia': importances})
feat_imp = feat_imp.sort_values(by="Importancia", ascending=True)

plt.figure(figsize=(8,6))
plt.barh(feat_imp['Variable'], feat_imp['Importancia'], color="pink", edgecolor="black")
plt.xlabel("Importancia en el modelo")
plt.title("Importancia de variables - Random Forest")
plt.show()
