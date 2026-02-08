import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

downloads_folder = os.path.expanduser('~/Downloads')
file_path = os.path.join(downloads_folder, 'kmeans_exercise5.xlsx')
df = pd.read_excel(file_path)
data = df[['Followers_k', 'Avg_Likes_k']].values

print("First 5 rows of data:")
print(df.head())
print("\n")

#Paso 1: Escoger 2 influencers al azar para ser los centros iniciales
k = 2
np.random.seed(42)  # For reproducibility
random_indices = np.random.choice(len(data), k, replace=False)
centroids = data[random_indices]

print(f"Centroid 1: {centroids[0]}")
print(f"Centroid 2: {centroids[1]}")

def distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

#Paso 2: Asignar cada influencer al centroide más cercano
print("Paso 2: Asignar cada influencer al centroide más cercano")

tiers = []
for influencer in data:
    dist1 = distance(influencer, centroids[0])
    dist2 = distance(influencer, centroids[1])

    if dist1 < dist2:
        tiers.append(1)  
    else:
        tiers.append(2)
df['Tier'] = tiers
print(df)

#Paso 3: Recalcular los centroides
print("\nPaso 3: Recalcular los centroides")
tier1_data = data[df['Tier'] == 1]
tier2_data = data[df['Tier'] == 2]

new_centroid1 = tier1_data.mean(axis=0)
new_centroid2 = tier2_data.mean(axis=0)

print(f"Nuevo Centroid 1: {new_centroid1}")
print(f"Nuevo Centroid 2: {new_centroid2}")

#Paso 4 & 5: Repetir hasta convergencia
print("\nPaso 4 & 5: Repetir hasta convergencia")

centroids = [new_centroid1, new_centroid2]
max_iterations = 10

for iteration in range(max_iterations):
    old_tiers = tiers.copy()

    tiers = []
    for influencer in data:
        dist1 = distance(influencer, centroids[0])
        dist2 = distance(influencer, centroids[1])

        if dist1 < dist2:
            tiers.append(1)  
        else:
            tiers.append(2)

    if tiers == old_tiers:
        print(f"Convergence reached after {iteration} iterations.")
        break

    df['Tier'] = tiers
    tier1_data = data[df['Tier'] == 1]
    tier2_data = data[df['Tier'] == 2]
    centroids = [tier1_data.mean(axis=0), tier2_data.mean(axis=0)]

#Resultados finales
df['Tier'] = tiers
print("\nResultados finales:")
print("\nTier 1 - Influencers en crecimiento:")
print(df[df['Tier'] == 1])
print("\nTier 2 - Influencers consolidados:")
print(df[df['Tier'] == 2])

#Visualización
plt.figure(figsize=(12, 8))

tier1_mask = df['Tier'] == 1
plt.scatter(df[tier1_mask]['Followers_k'], df[tier1_mask]['Avg_Likes_k'], color='red', label='Tier 1 - En crecimiento', s=300, alpha=0.7, edgecolors='black', linewidths=2)

tier2_mask = df['Tier'] == 2
plt.scatter(df[tier2_mask]['Followers_k'], df[tier2_mask]['Avg_Likes_k'], color='blue', label='Tier 2 - Consolidados', s=300, alpha=0.7, edgecolors='black', linewidths=2)

plt.scatter(centroids[0][0], centroids[0][1], color='yellow', marker='*', s=800, label='Centroid 1', edgecolors='black', linewidths=3, zorder=5)
plt.scatter(centroids[1][0], centroids[1][1], color='cyan', marker='*', s=800, label='Centroid 2', edgecolors='black', linewidths=3, zorder=5)

for i, row in df.iterrows():
    plt.annotate(row["Creator_ID"], (row['Followers_k'], row['Avg_Likes_k']), fontsize=10, fontweight='bold', ha='center', va='center')

plt.annotate('Centroid 1', (centroids[0][0], centroids[0][1]), textcoords="offset points", xytext=(0,7), ha='center', fontsize=12, fontweight='bold', color='yellow')
plt.annotate('Centroid 2', (centroids[1][0], centroids[1][1]), textcoords="offset points", xytext=(0,7), ha='center', fontsize=12, fontweight='bold', color='cyan')
plt.xlabel('Followers (thousands)', fontsize=14, fontweight='bold')
plt.ylabel('Average Likes (thousands)', fontsize=14, fontweight='bold')
plt.title('Influencer tiers - K-Means Clustering', fontsize=18, fontweight='bold')
plt.legend(fontsize=12, loc='upper left')
plt.grid(True, alpha=0.3)

plt.tight_layout()
output_file = os.path.expanduser('~\\Downloads\\Influencer_tiers_grafic.png')
plt.savefig(output_file, dpi=150, bbox_inches='tight')
plt.show()