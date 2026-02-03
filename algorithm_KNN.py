import os
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Parte 1: Cargar y preparar los datos

downloads_file = os.path.expanduser('~\\Downloads\\data_knn.csv')
data = pd.read_csv(downloads_file)

print("Nuestra informacion de los estudiantes")
print(data.head())
print(f"Total de estudiantes: {len(data)}")

# Parte 2: Separar X y Y

X = data[['Study_Hours', 'Sleep_Hours', 'Attendance_Percent']]
y = data['Pass_Fail']

# Parte 3: Entrenar el modelo KNN

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
print("\n Knn modelo entrenado correctamente.")

# Parte 4: Probar el modelo con nuevos datos
new_students = [[6,7,80]]
predictions = knn.predict(new_students)

# Parte 5: Crear un grafico
plt.figure(figsize=(10,6))

pass_students = data[data['Pass_Fail'] == 'Pass']
fail_students = data[data['Pass_Fail'] == 'Fail']

plt.scatter(pass_students['Study_Hours'], pass_students['Attendance_Percent'], color='green', label='Pass', s=100, alpha=0.6)
plt.scatter(fail_students['Study_Hours'], fail_students['Attendance_Percent'], color='red', label='Fail', s=100, alpha=0.6)

plt.scatter(6, 80, color='gold', marker='*', s=500, label='New Student', edgecolors='black', linewidths=2)

plt.xlabel('Study Hours per Day', fontsize=12)
plt.ylabel('Attendance Percentange (%)', fontsize=12)
plt.title('KNN Classification: Student Performance', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

output_file = os.path.expanduser('~\\Downloads\\knn_visualization.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.show()