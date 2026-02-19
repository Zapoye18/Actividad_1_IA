import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# 1: Load Dataset
downloads_foleder = os.path.expanduser("~/Downloads")
data_file = os.path.join(downloads_foleder, "dbscan_exercise2.xlsx")
df = pd.read_excel(data_file)

print("First 5 rows of data:")
print(df.head())
print(f"\nData shape: {df.shape}")

# 2: Prepare Data, extract relevant columns
X = df[['Latitude', 'Longitude']].values
print(f"\nMissing values: {df.isnull().sum().sum()}")
print(f"Data array shape: {X.shape}")

# 3: Create DBSCAN model
model = DBSCAN(eps=0.002, min_samples=3)

clusters = model.fit_predict(X)
df['Cluster'] = clusters
print(f"\nCluster labels: {clusters}")
print(f"\nUnique clusters found: {len(set(clusters)) - (1 if -1 in clusters else 0)}")
print(f"Noise points: {list(clusters).count(-1)}")

# 4: Visualize clusters
plt.figure(figsize=(12, 8))
unique_clusters = set(clusters)
colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta']

for cluster_id in unique_clusters:
    if cluster_id == -1:
        cluster_points = X[clusters == cluster_id]
        plt.scatter(cluster_points[:, 1], cluster_points[:, 0], color='black', marker= "x", label='Noise (Isolated Crimes)', s=100, linewidths=3)
    else:
        cluster_points = X[clusters == cluster_id]
        plt.scatter(cluster_points[:, 1], cluster_points[:, 0], c=colors[cluster_id % len(colors)], label=f'Hotspot {cluster_id + 1}', s=150, alpha=0.7, edgecolors='black', linewidths=2)

plt.title('Crime Hotspots Analysis - DBSCAN', fontsize=16, fontweight='bold')
plt.xlabel('Longitude', fontsize=12)
plt.ylabel('Latitude', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.savefig('crime_hotspots.png', dpi=300, bbox_inches='tight')
plt.show()

# 5: Analyze and interpret clusters
print("\n" + "="*60)
print("CRIME HOTSPOT ANALYSIS RESULTS")

for cluster_id in sorted(unique_clusters):
    if cluster_id == -1:
        count = list(clusters).count(-1)
        print(f"\nX NOISE (Isolated Crimes): {count} crimes")
        print(f"    These crimes are random crimes, not part of a pattern.")
    else:
        cluster_data = df[df['Cluster'] == cluster_id]
        count = len(cluster_data)
        avg_lat = cluster_data['Latitude'].mean()
        avg_lon = cluster_data['Longitude'].mean()

        print(f"\n!!! HOTSPOT {cluster_id + 1}: {count} crimes")
        print(f"    Center Location: lat {avg_lat:.4f}, lon {avg_lon:.4f}")
        print(f"    Action: Increase police patrols in this area")

# Summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
total_hotspots = len(unique_clusters) - (1 if -1 in clusters else 0)
total_noise = list(clusters).count(-1)
total_incidents = len(df)

print(f"Total crime incidents: {total_incidents}")
print(f"Hotspots Found: {total_hotspots}")
print(f"Isolated Incidents (Noise): {total_noise}")
print(f"Incidents in Hotspots: {total_incidents - total_noise}")