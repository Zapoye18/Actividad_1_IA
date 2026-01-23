import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Patch

#Importamos los datos de xlml

print("\nPrimer Paso")

downloads_file = os.path.expanduser('~\\Downloads\\data_exercise1.csv')

print("Buscando el archivo en: ", {downloads_file})
print("El archivo existe: ", {os.path.isfile(downloads_file)})

if not os.path.exists(downloads_file):
    print("El archivo no se encuentra en la ruta especificada.")
    print("Por favor, asegurate de que el archivo 'data_exercise1.csv' esté en la carpeta Descargas.")
    exit()

#Leemos los datos
print("Leyendo los datos del archivo CSV...")
df_training = pd.read_csv(downloads_file)

print("Datos leidos correctamente. Mostrando las primeras filas:")
print(f"Columnas disponibles: {df_training.columns.tolist()}")
print(f"Numero de filas: {len(df_training)}")
print("\nPrimeras filas del DataFrame:")
print(df_training.head())

# Limpieza de datos
print("\nLimpiando los datos...")
# Eliminar comas del Income y convertir a numérico
df_training['Income'] = df_training['Income'].astype(str).str.replace(',', '').astype(float)
# Convertir Default de YES/NO a 1/0
df_training['Default'] = (df_training['Default'] == 'YES').astype(int)
print("Datos limpios y convertidos correctamente.")

# Segundo paso: Preparar la información de entrenamiento
print("\n" + "="*60)
print("\nSegundo Paso")

x = df_training[['Age', 'Credit_Score', 'Income']]
y = df_training['Default']

print("Caracteristicas y variable objetivo separadas correctamente.")
print(f"Caracteristicas (X): Age, Credit_Score, Income")
print(f"Variable objetivo (y): Default (0=NO, 1=YES)")

print(f"\nDistribucion de clase:")
print(f"    NO Default (0): {(y==0).sum()} clientes")
print(f"    Default (1): {(y==1).sum()} clientes")

# Tercer paso: Entrenamiento regresión logística
print("\n" + "="*40)
print("\nTercer Paso")

Ir_model = LogisticRegression(random_state=42, max_iter=1000)
Ir_model.fit(x, y)

print("Modelo de regresion logistica entrenado correctamente.")

# Cuarto paso: Prueva en los datos de entrenamiento

print("\n" + "="*40)
print("\nCuarto Paso")

y_pred = Ir_model.predict(x)
accuracy = accuracy_score(y, y_pred)

print(f"\nPrecision del modelo en los datos de entrenamiento: {accuracy*100:.1f}%")

# Quinto paso: Ajustar los pesos
print("\n" + "="*60)
print("\nQuinto Paso")

print("\nPesos (coeficientes) del modelo de regresion logistica:")
print("Que tan importante es cada caracteristica?\n")

features = ['Age', 'Credit_Score', 'Income']
weights = Ir_model.coef_[0]

for feature, weight in zip(features, weights):
    if weight > 0:
        effect = "X aumenta la probabilidad de Default"
    else:
        effect = "X disminuye la probabilidad de Default"
    print(f"    {feature:15}: {weight:10.7f}   {effect}")

print(f"\nIntercepto (bias) del modelo: {Ir_model.intercept_[0]:.6f}")

print("\nQue significa esto?")

print("\nEdad 1")
if weights[0] > 0:
    print("Gente mas joven tiende a defaultar mas.")
else:
    print("Gente mas vieja tiende a defaultar mas.")

print("\nCredit Score:")
if weights[1] > 0:
    print("Gente con mejor credit score tiende a defaultar mas.")
else:
    print("Gente con mejor credit score tiende a defaultar menos (mas seguro).")

print("\nIncome:")
if weights[2] > 0:
    print("Gente con mas income tiende a defaultar mas.")
else:
    print("Gente con mas income tiende a defaultar menos (mas seguro).")

# Sexto paso: Predecir nuevos clientes
print("\n" + "="*40)
print("\nSexto Paso")

new_customers = pd.DataFrame({
    'Age': [25, 40, 60, 30, 50],
    'Credit_Score': [700, 650, 600, 720, 680],
    'Income': [50000, 80000, 60000, 75000, 90000]
})

predictions = Ir_model.predict(new_customers)
probabilities = Ir_model.predict_proba(new_customers)

print("\nResultado para los 5 nuevos clientes:\n")

for i in range(len(new_customers)):
    age = new_customers.loc[i, 'Age']
    credit = new_customers.loc[i, 'Credit_Score']
    income = new_customers.loc[i, 'Income']

    predicion = predictions[i]
    prob_no_default = probabilities[i][0] * 100
    prob_default = probabilities[i][1] * 100

    print(f"Cliente {i+1}:")
    print(f"    Age: {age}, Credit_Score: {credit}, Income: ${income:,}")

    if predicion == 1:
        result = "X: DEFAULT"
    else:
        result = "OK: NO DEFAULT"

    print(f"  Prediccion: {result}")
    print(f"    Probabilidad NO DEFAULT: {prob_no_default:.1f}%")
    print(f"    Probabilidad DEFAULT: {prob_default:.1f}%\n")
    print()

# Septimo paso: Visualizar la regresión logística
print("\n" + "="*40)
print("\nSeptimo Paso... Desplegando grafica de la curva sigmoide")

X_scores = Ir_model.decision_function(x)

score_range = np.linspace(X_scores.min() - 1, X_scores.max() + 1, 300)
sigmoid_curve = 1 / (1 + np.exp(-score_range))

plt.figure(figsize=(12,7))
plt.plot(score_range, sigmoid_curve, linewidth=4, label='Curva Sigmoide', color='red', zorder=3)

plt.axhline(y=0.5, color='blue', linestyle='--', linewidth=3, label='Umbral de Decision (0.5)', zorder=2)

plt.fill_between(score_range, 0, sigmoid_curve, where=(sigmoid_curve <= 0.5), color='green', alpha=0.5, label='Prediccion: NO DEFAULT', zorder=1)
plt.fill_between(score_range, 1, sigmoid_curve, where=(sigmoid_curve > 0.5), color='red', alpha=0.5, label='Prediccion: DEFAULT', zorder=1)

no_default_mask = y == 0
default_mask = y == 1

y_no_default_jitter = np.random.normal(0, 0.02, no_default_mask.sum())
y_default_jitter = np.random.normal(1, 0.02, default_mask.sum())

plt.scatter(X_scores[no_default_mask], y_no_default_jitter, color='green', s=200, marker="o", edgecolors="black", linewidth=2.5,  label='Entrenamiento: NO DEFAULT', zorder=5, alpha=0.8)
plt.scatter(X_scores[default_mask], y_default_jitter, color='red', s=200, marker="o", edgecolors="black", linewidth=2.5, label='Entrenamiento: DEFAULT', zorder=5, alpha=0.8)

plt.xlabel('Puntaje (output del modelo)', fontsize=13, fontweight='bold')
plt.ylabel('Probabilidad de Default', fontsize=13, fontweight='bold')
plt.title('Regresion Logistica con la informacion de entrenamiento', fontsize=14, fontweight='bold', pad=20)
plt.ylim([-0.15, 1.15])
plt.grid(True, linestyle='--', alpha=0.3, linewidth=1)
plt.legend(loc='center left', fontsize=11, framealpha=0.95, edgecolor='black', fancybox=True)

plt.tight_layout()
plt.savefig('sigmoid_curve.png', dpi=300, bbox_inches='tight')
plt.show()

