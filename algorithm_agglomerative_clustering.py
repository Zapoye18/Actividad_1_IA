import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Parte 1: Cargar y preparar los datos
downloads_folder = os.path.expanduser('~/Downloads')
data_file = os.path.join(downloads_folder, 'hierchical_cluster_dataset_exercise1.xlsx')
df = pd.read_excel(data_file)

print("First 5 rows of data:")
print(df.head())
print(f"Data shape: {df.shape}")

# Parte 2: Preparar los datos, extraer las columnas relevantes
X = df[['Annual_Income', 'Spending_Score']].values
print(f"\nMissing values: {df.isnull().sum().sum()}")
print(f"\nData array shape: {X.shape}")

# Parte 3: Crear el modelo de clustering jerárquico
model = AgglomerativeClustering(n_clusters=5, linkage='ward')
clusters = model.fit_predict(X)
print(f"\nCluster labels: {clusters}")

# Parte 4: Visualizar los clusters
Z = linkage(X, method='ward')

plt.figure(figsize=(12, 6))
dendrogram(Z)
plt.title('Hierarchical Clustering Dendrogram', fontsize=16, fontweight='bold')
plt.xlabel('Customer Index', fontsize=12)
plt.ylabel('Distance', fontsize=12)
plt.savefig('dendrogram.png', dpi=300, bbox_inches='tight')
plt.show()

# Paso extra: Como fue facil, vamos a crear otra grafica para visualizar los clusters
plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green', 'purple', 'orange']

for i in range(5):
    cluster_points = X[clusters == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i], label=f'Cluster {i+1}', s=100, alpha=0.6, edgecolors='black')

plt.title('Customer Segments - Hierarchical Clustering', fontsize=16)
plt.xlabel('Annual Income ($k)', fontsize=12)
plt.ylabel('Spending Score (0-100)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('clusters_scatter.png', dpi=300, bbox_inches='tight')
plt.show()

# Paso extra 2: Analiza y interpretar los clusters
print("\n" + "="*40)
print("CLUSTER ANALYSIS - Customer Segments")
df['Cluster'] = clusters

for i in range(5):
    cluster_data = df[df['Cluster'] == i]
    print(f"\n CLUSTER {i+1}:")
    print(f" Number of customers: {len(cluster_data)}")
    print(f" Averge Income: ${cluster_data['Annual_Income'].mean():.2f}")
    print(f" Average Spending Score: {cluster_data['Spending_Score'].mean():.2f}")

    avg_income = cluster_data['Annual_Income'].mean()
    avg_spending = cluster_data['Spending_Score'].mean()

    if avg_income < 40 and avg_spending < 40:
        customer_type = "Budget customer - low income, low spending"
    elif avg_income < 40 and avg_spending >= 40:
        customer_type = "Impulsive Buyers - Low income, Hight spending"
    elif avg_income >= 40 and avg_income < 70 and avg_spending >= 40 and avg_spending < 60:
        customer_type = "Average Customer (Medium Income, Medium Spending)"
    elif avg_income >= 70 and avg_spending < 40:
        customer_type = "Conservative Shoppers (Hight Income, Low Spending)"
    elif avg_income >= 70 and avg_spending >= 40:
        customer_type = "Premium Customers (Hight Income, Hight Spending)"
    else:
        customer_type = "Mixed Segment"

# Summary table
print("\n" + "="*50)
print("SUMMARY TABLE")
print("="*50)
summary = df.groupby('Cluster').agg({
    'Customer_ID': 'count',
    'Annual_Income': 'mean',
    'Spending_Score': 'mean'
}).round(2)
    
summary.columns = ['Count', 'Avg Income ($k)', 'Avg Spending Score']
print(summary)
