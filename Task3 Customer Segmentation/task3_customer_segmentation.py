# =====================================
# Task 3: Customer Segmentation
# =====================================

# 1. Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 2. Generate Synthetic Dataset
np.random.seed(42)

n_customers = 200

data = {
    "CustomerID": range(1, n_customers + 1),
    "Age": np.random.randint(18, 70, n_customers),
    "Annual Income (k$)": np.random.randint(15, 150, n_customers),
    "Spending Score (1-100)": np.random.randint(1, 100, n_customers)
}

df = pd.DataFrame(data)

print("First 5 Rows:")
print(df.head())

# 3. Select Features for Clustering
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# 4. Scale the Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Elbow Method to Find Optimal Clusters
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss)
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# 6. Apply KMeans (Using 5 Clusters)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# 7. Visualize Customer Segments
plt.figure(figsize=(8, 6))
plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["Cluster"]
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segments")
plt.show()

# 8. Print Cluster Summary (Means)
print("\nCluster Summary (Means):")
cluster_summary = df.groupby("Cluster").mean(numeric_only=True)
print(cluster_summary)

# 9. Print Segment Sizes
print("\nSegment Sizes:")
segment_sizes = df["Cluster"].value_counts().sort_index()
print(segment_sizes)
