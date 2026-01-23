import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
import os

#Importamos los datos de xlml

downloads_file = os.path.expanduser('~\\Downloads\\data_exercise1.csv')

print("\nPrimer Paso")
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

# Segundo paso: Separamos las características y la variable objetivo
print("\nSegundo Paso")
try:
    X_training = df_training[['Age', 'Credit_Score', 'Income']]
    y_training = df_training['Default']
    print("\nCaracteristicas y variable objetivo separadas correctamente.")

except KeyError as e:
    print(f"Error: Columna no encontrada - {e}")
    print(f"Tus columnas disponibles son: {df_training.columns.tolist()}")
    print("Necesitas las columnas 'Age', 'Credit_Score', 'Income' y 'Default'.")
    exit()

# Tercer paso: Entrenamos el modelo de árbol de decisión
print("\nTercer Paso")
print("Entrenando el modelo de arbol de decision...")
tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_model.fit(X_training, y_training)
print("Modelo entrenado correctamente.")

# Cuarto paso: Testamos el modelo con datos de prueba
print("\nCuarto Paso")
y_predictions = tree_model.predict(X_training)
accuracy = accuracy_score(y_training, y_predictions)

print(f"\nPrecision del modelo en los datos de entrenamiento: {accuracy*100:.1f}%")

# Quinto paso: Predecimos los nuevos clientes
print("\nQuinto Paso")
new_customers = pd.DataFrame({
    'Age': [25, 40, 60, 30, 50],
    'Credit_Score': [700, 650, 600, 720, 680],
    'Income': [50000, 80000, 60000, 75000, 90000]
})

print("\n" + "="*50)
print("Nuevos clientes para predecir:")
print("="*50)

new_predictions = tree_model.predict(new_customers)
probabilities = tree_model.predict_proba(new_customers)

for i, (idx, row) in enumerate(new_customers.iterrows()):
    pred = "DEAFULT" if new_predictions[i] == 1 else "NO DEFAULT"
    confidence = max(probabilities[i]) * 100
    print(f"\nCliente {i+1}")
    print(f"    Age: {int(row['Age'])}, Credit: {int(row['Credit_Score'])}, Income: ${int(row['Income']):,}")
    print(f"    Prediccion: {pred} (Confianza: {confidence:.1f}% confianza)")
