import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Parte 1: Cargar y preparar los datos
downloads_folder = os.path.expanduser('~/Downloads')
data_file = os.path.join(downloads_folder, 'exercise4_svr.xlsx')
data = pd.read_excel(data_file)

print("First 5 rows of data:")
print(data.head())
print("\n")

# Parte 2: Separar X y Y
X = data[['Number of Customers']].values
# Use a 1D array for the target so scalar formatting works (avoid (n,1) arrays)
y = data['Daily Revenue ($)'].values

print("Features (X) - Number of Customers:")
print(X[:5])
print("\n")

print("Target (y) - Daily Revenue:")
print(y[:5])

# Parte 3: Dividir los datos en conjuntos de entrenamiento y prueba (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {len(X_train)} days")
print(f"Testing set size: {len(X_test)} days")
print("\n")

# Parte 4: Escalar las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Original training data (first 4):")
print(X_train[:4])
print("\n")

print("Scaled training data (first 4):")
print(X_train_scaled[:4])
print("\n")

# Parte 5: Entrenar el modelo SVR con kernel RBF
svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr_model.fit(X_train_scaled, y_train)
print("SVR Model trained successfully.")
print(f"Number of support vectors: {len(svr_model.support_vectors_)}")

# Parte 6: Hacer predicciones y evaluar el modelo
y_pred_test = svr_model.predict(X_test_scaled)

print("Test Set Results:")
print("-"*50)
for i in range(len(y_test)):
    print(f"Customers: {X_test[i][0]:>3.0f} | Actual: ${y_test[i]:>6.2f} | Predicted: ${y_pred_test[i]:>6.2f} | Error: ${abs(y_test[i]-y_pred_test[i]):5.2f}")
print("\n")

# Parte 7: Predecir para 150 clientes
customers_tomorrow = 150
customers_input = np.array([[customers_tomorrow]])
customers_scaled = scaler.transform(customers_input)
predicted_revenue = svr_model.predict(customers_scaled)[0]

print("-"*10)
print("Final Prediction:")
print(f"Expected Customers: {customers_tomorrow}")
print(f"Predicted Revenue: ${predicted_revenue:.2f}")
print("-"*10)

# Parte 8: Crear un gráfico de los resultados
plt.figure(figsize=(12,6))
plt.scatter(X_train, y_train, color='blue', label='Training Data', alpha=0.6, s=50)
plt.scatter(X_test, y_test, color='green', label='Test Data (Actual)', alpha=0.6, s=50)
plt.scatter(X_test, y_pred_test, color='red', label='Test Data (Predicted)', marker="x", s=100)

X_range = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
X_range_scaled = scaler.transform(X_range)
y_range_pred = svr_model.predict(X_range_scaled)

plt.plot(X_range, y_range_pred, color="orange", linewidth=2, label='SVR Model Prediction Line')

plt.scatter([customers_tomorrow], [predicted_revenue], color='purple', s=200, marker='*', label=f'Prediction for {customers_tomorrow} customers', edgecolors='black', linewidths=2, zorder=5)

plt.xlabel('Number of Customers', fontsize=12, fontweight='bold')
plt.ylabel('Daily Revenue ($)', fontsize=12, fontweight='bold')
plt.title('Coffe Shop Revenue Prediction using SVR', fontsize=14, fontweight='bold')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
