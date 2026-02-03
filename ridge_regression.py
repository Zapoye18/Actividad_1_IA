import os
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# Parte 1: Cargar y preparar los datos
downloads_folder = os.path.expanduser('~/Downloads')
data_file = os.path.join(downloads_folder, 'data_exercise3_ridge_regression.xlsx')
data = pd.read_excel(data_file)

print("Ice Cream Sales Data:")
print(data)
print(f"Total days: {len(data)}")

# Paso 2: Separar X y Y
X = data[['Temperature_C', 'Humidity_Percent', 'Day_Type', 'Hours_Open']].values
y = data['Sales_Revenue'].values

print(f"\nX has {X.shape[1]} features")
print(f"y has {len(y)} sales records")

# Paso 3: Estandarizar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Paso 4: Entrenar el modelo de regresión Ridge 
alpha_value = 1.0
model = Ridge(alpha=alpha_value)
model.fit(X_scaled, y)

print(f"\nRidge Regression model trained with alpha={alpha_value}.")

# Parte 5: Mostrar los coeficientes del modelo
b0 = model.intercept_
coefficients = model.coef_

print("\n"+"="*15)
print("Model Coefficients")
print("="*60)
print(f"bº (intercept): ${b0:.2f}")
print(f"\nFeature Coefficients:")
print(f"Temperature: {coefficients[0]:.2f}")
print(f"Humidity:    {coefficients[1]:.2f}")
print(f"Day Type:    {coefficients[2]:.2f}")
print(f"Hours Open:  {coefficients[3]:.2f}")

# Parte 6: Predeccion
new_day = [[28, 65, 1, 8]]  # 85°F, 70% humidity, Weekend (1), 10 hours open
new_day_scaled = scaler.transform(new_day)
predicted_revenue = model.predict(new_day_scaled)[0]

print("\n" + "--"*20)
print("Prediction for New Day")
print(f"Temperature: 28°C")
print(f"Humidity: 65%")
print(f"Day Type: Weekend")
print(f"Hours Open: 8 hours")
print(f"\nPredicted Renevenue: ${predicted_revenue:.2f}")

# Parte 7: Encontrar el mejor alpha 
print("\n" + "+"*20)
print("Testing different alpha values")

alphas = [0.01, 0.1, 1, 10, 100]
best_alpha = None
best_score = -999999

for alpha in alphas:
    model_test = Ridge(alpha=alpha)
    # Usar validación cruzada para evaluar el modelo
    scores = cross_val_score(model_test, X_scaled, y, cv=5, scoring='r2')
    avg_score = scores.mean()
    print(f"Alpha: {alpha:6} | Average R2 Score = {avg_score:.4f}")
    
    if avg_score > best_score:
        best_score = avg_score
        best_alpha = alpha

print(f"\nBest alpha found: {best_alpha} with R2 Score = {best_score:.4f}")

# Parte 8: Reentrenar el modelo con el mejor alpha
final_model = Ridge(alpha=best_alpha)
final_model.fit(X_scaled, y)

print(f"\nBest alpha found: {best_alpha} with R2 Score = {best_score:.4f}")

#Hacer la predicción final
final_prediction = final_model.predict(new_day_scaled)[0]
print(f"Final prediction for new day: ${final_prediction:.2f}")

# Parte 9: Crear gráfica de los resultados
y_pred = final_model.predict(X_scaled)

plt.figure(figsize=(10, 6))

plt.scatter(range(len(y)), y, color='blue', s=100, label='Actual Revenue', alpha=0.6, edgecolors='black')
plt.scatter(range(len(y_pred)), y_pred, color='red', s=80, label='Predicted Revenue', alpha=0.5, linewidths=2, marker='x')

plt.scatter(len(y), final_prediction, color='gold', marker='*', s=500, label=f'Prediction for New Day = ${final_prediction:.0f}', edgecolors='black', linewidths=2, zorder=5)

plt.xlabel('Days Number', fontsize=12)
plt.ylabel('Sales Revenue ($)', fontsize=12)
plt.title('Ridge Regression: Ice Cream Sales Prediction', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

output_file = os.path.join(downloads_folder, 'ridge_regression_visualization.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.show()