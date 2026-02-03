import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Parte 1: Cargar y preparar los datos

downloads_folder = os.path.expanduser('~/Downloads')
data_file = os.path.join(downloads_folder, 'data_exercise2_pr.xlsx')
data = pd.read_excel(data_file)

# Paso 2: Separar X y Y

X = data[['Sunlight_Hours', 'Water_Liters']].values
y = data['Plant_Height_cm'].values

# Paso 3: Crear características polinomiales
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Paso 4: Entrenar el modelo de regresión polinómica
model = LinearRegression()
model.fit(X_poly, y)
print("\nModelo de regresión polinómica entrenado correctamente.")

# Parte 5: Mostrar los coeficientes del modelo
b0 = model.intercept_
coefficients = model.coef_

print("\n"+"="*50)
print("Coeficientes del modelo")
print("="*50)
print(f"bº (intercepto): {b0:.2f}")
print(f"\nRepresentacion de los coeficientes:")
print(f"    - Efectos lineales de Sunlight y Water")
print(f"    - Efectos cuadráticos de Sunlight^2 y Water^2")
print("="*50)

# Parte 6: Hacer la prediccion
new_plant = [[8, 2.5]]  # 8 horas de sol y 2.5 litros de agua
new_plant_poly = poly.transform(new_plant)
predicted_height = model.predict(new_plant_poly)[0]

print("\n"+"="*50)
print("Predicción para nueva planta")
print("="*50)
print(f"Sunlight: 8 hours, Water: 2.5 liters per day")
print(f"\n -> Predicted Plant Height: {predicted_height:.2f} cm")
print("="*50)

# Parte 7: Visualización de la grafica
plt.figure(figsize=(10, 6))
plt.scatter(data['Sunlight_Hours'], y, color='green', s=100, alpha=0.6, label='Datos Reales')
plt.scatter(8, predicted_height, color='gold', marker='*', s=500, label=f'Predicción (8h, 2.5L) = {predicted_height:.1f}cm', edgecolors='black', linewidths=2, zorder=5)

plt.xlabel('Luz de sol diaria', fontsize=12)
plt.ylabel('Altura de la planta (cm)', fontsize=12)
plt.title('Regresión Polinómica: Prediccion del crecimiento de las plantas', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

output_file = os.path.join(downloads_folder, 'polynomial_regression_grafic.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.show()