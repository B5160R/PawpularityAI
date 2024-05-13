import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import decomposition
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

df = pd.read_csv("../../data/pawpularity/train.csv")
df = df.dropna()

# split data
X = df.drop(columns=["Id", "Pawpularity"])
y = df["Pawpularity"]

# decompose data
pca = decomposition.PCA(n_components=2)
X_pca = pca.fit_transform(X)

print(X_pca)

# visualize the data in 2D
fig = plt.figure(figsize=(8, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=plt.cm.nipy_spectral, edgecolor='k')
plt.colorbar(label='Pawpularity')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Visualization')
plt.show()

# save plot
fig.savefig("plots/pca_visualization.png")

# find the optimal number of clusters
error = []
for i in range(1, 21):
		kmeans = KMeans(n_clusters=i).fit(X_pca)
		error.append(kmeans.inertia_)

# plot the elbow method
fig = plt.figure(figsize=(8, 8))
plt.plot(range(1, 21), error)
plt.xlabel('Number of clusters')
plt.ylabel('Error')
plt.title('Elbow Method')
plt.show()

# cluster the data using KMeans
kmeans = KMeans(n_clusters=5)
kmeans.fit(X_pca)
y_kmeans = kmeans.predict(X_pca)

# visualize the KMeans clusters
fig = plt.figure(figsize=(8, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap=plt.cm.nipy_spectral, edgecolor='k')
centers = kmeans.cluster_centers_
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('KMeans Clustering')
plt.show()

# save plot
fig.savefig("plots/kmeans_clustering.png")

# cluster the data using DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
y_dbscan = dbscan.fit_predict(X_pca)

# visualize the DBSCAN clusters
fig = plt.figure(figsize=(8, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_dbscan, cmap=plt.cm.nipy_spectral, edgecolor='k')
plt.title('DBSCAN Clustering')
plt.show()