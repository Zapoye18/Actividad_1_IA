import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Parte 1: Cargar y preparar los datos
downloads_folder = os.path.expanduser('~/Downloads')
data_file = os.path.join(downloads_folder, 'gmm_exercise3.xlsx')
df = pd.read_excel(data_file)

print("First 5 rows of data:")
print(df.head())
print(f"\nData shape: {df.shape}")

X = df[['Annual_Income_k', 'Spending_Score']].values

# plot 1: Visualizar la fila de datos antes de clustering
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c='gray', s=50, alpha=0.7)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Raw Customer Data')
plt.grid(True, alpha=0.3)
plt.show()

# Paso 0: Escoger n_components usando BIC, No necesitamos saber cuantos grupos hay

k_range = range(2, 8)
bic_scores = []

for k in k_range:
    gmm_test = GaussianMixture(n_components=k, random_state=42)
    gmm_test.fit(X)
    bic_scores.append(gmm_test.bic(X))
    print(f"K={k}, BIC= {gmm_test.bic(X):.2f}")

# Paso 2: BIC scores
plt.figure(figsize=(8, 5))
plt.plot(k_range, bic_scores, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Components (k)')
plt.ylabel('BIC Score (lower is better)')
plt.title('Step 0: Finding the best number of components')
plt.grid(True, alpha=0.3)

best_k = list(k_range)[np.argmin(bic_scores)]
plt.axvline(x=best_k, color='red', linestyle='--', label=f'Best k={best_k}')
plt.legend()
plt.show()
print(f"\n>>> Best number of components: {best_k}")

# Aplicar GMM con el mejor número de componentes, establecemos una media random de inicio, varianzas, y pesos, corremos E-steps, M-steps, y repetimos hasta convergencia
gmm = GaussianMixture(n_components=best_k, covariance_type='full', random_state=42)
gmm.fit(X)

# Paso 1 (E-step): Vemos las responsabilidades
probabilities = gmm.predict_proba(X)
labels = gmm.predict(X)

print(f"\n--- STEP 1: Soft Probabilities (firt 10 customers) ---")
prob_df = pd.DataFrame(probabilities, columns=[f'Cluster {i}' for i in range(best_k)])
prob_df.insert(0, 'CustomerID', df['CustomerID'])
print(prob_df.head(10).to_string(index=False))

max_prob = probabilities.max(axis=1)
uncertain = np.where(max_prob < 0.70)[0]
print(f"\nCustomers with mixed memberships (max probability < 70%): {len(uncertain)}")

if len(uncertain) > 0:
    print("\nThese customers don't clearly belong to one group:")
    for idx in uncertain[:5]:
        prob_str = " | ".join([f"c{i}: {probabilities[idx][i]:.1%}" for i in range(best_k)])
        print(f"Customer {df['CustomerID'].iloc[idx]} - {prob_str}")
    
# Paso 2 (M-Step): Actualizamos los parámetros del modelo, despues de la convergencia, podremos inspectar que algoritmo
print("\nStep 2 - Cluster Center (means):")
for i in range(best_k):
    print(f"    Cluster {i}: Income={gmm.means_[i][0]:.1f}k$, Spending Score={gmm.means_[i][1]:.1f}")

print("\nMixing Weights (proportion of data per cluster):")
for i in range(best_k):
    print(f"    Cluster {i}: {gmm.weights_[i]:.1%}")

# Paso 3: Converge la informacion
print("\nStep 3 - Convergence")
print(f"Converged: {gmm.converged_}")
print(f"Iterations needed: {gmm.n_iter_}")

# Paso 4: Visualizar los resultados, plot 3 GMM Clusters
colors = ['blue', 'red', 'green', 'orange', 'purple', 'pink']

plt.figure(figsize=(8, 6))
for i in range(best_k):
    mask = labels == i
    plt.scatter(X[mask, 0], X[mask, 1], color=colors[i], label=f'Cluster {i}', s=60, alpha=0.7)
plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='black', marker='X', s=200, label='Cluster Centers', zorder=5)

plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('GMM Clustering Results (Soft Clustering)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Paso 4: Mostrar insertidumbre
plt.figure(figsize=(8, 6))
uncertanty = 1 - max_prob
scatter = plt.scatter(X[:, 0], X[:, 1], c=uncertanty, cmap='RdYlGn_r', s=60, edgecolors='gray', linewidth=0.5)
plt.colorbar(scatter, label='Uncertainty (higher = more mixed)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Uncertainty Map: Who Belongs to Multiple Segments?')
plt.grid(True, alpha=0.3)
plt.show()