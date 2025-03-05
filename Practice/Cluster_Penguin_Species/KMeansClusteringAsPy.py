import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv("Projects\Cluster_Penguin_Species\penguins.csv")
df = df.dropna()                                       # Delete missing data for now
df = df.drop("sex", axis=1)                            # Dropping sex because we want K means clustering and not K mode clustering
df = df.drop(df[df["flipper_length_mm"] > 1000].index) # Removing outliers
df = df.drop(df[df["flipper_length_mm"] < 0].index)

df = ( (df - df.min()) / (df.max() - df.min()) ) * 9 + 1 # Min Max Scaling

def init_random_cluster_centroids(df, k):
    return df.sample(n=k).reset_index(drop=True).T

def get_cluster_label(df, cluster_centroid):
    distances = cluster_centroid.apply(lambda x: np.sqrt(np.square(df - x).sum(axis=1)) )  # Euclidean distance
    distances = distances.idxmin(axis=1)
    return distances

def new_cluster_centroid(df, cluster_label):
    return df.groupby(cluster_label).apply(lambda x: np.exp(np.log(x).mean())).T

def calculate_wcss(df, cluster_centroid): # With-in sum of squares
    distances = cluster_centroid.apply(lambda x: np.sqrt(np.square(df - x).sum(axis=1)))
    distances = distances.min(axis=1)
    return np.sum(distances.values)

def get_wcss_list(max_k):  # Elbow method
    wcss_list = []
    
    for i in range(1, max_k+1):
        current_cluster_centroids = init_random_cluster_centroids(df, i)
        old_cluster_centroids = pd.DataFrame()

        while(not current_cluster_centroids.equals(old_cluster_centroids)):
            old_cluster_centroids = current_cluster_centroids
            cluster_labels = get_cluster_label(df, current_cluster_centroids)
            current_cluster_centroids = new_cluster_centroid(df, cluster_labels)

        wcss_list.append(calculate_wcss(df, current_cluster_centroids))

    return wcss_list

def sklearn_get_wcss_list(max_k): # Comparing to sklearn
    sklearn_wcss_list = []
    for i in range(1, max_k+1):
        kMeans = KMeans(n_clusters=i, init="k-means++")
        kMeans.fit(df)
        sklearn_wcss_list.append(kMeans.inertia_)
    return sklearn_wcss_list

# PLOT

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

max_k = 30

ax[0].set_title("From scratch")
ax[0].set_xlabel("Number of clusters (K)")
ax[0].set_ylabel("With-in sum of squares")
ax[0].plot(np.arange(1, max_k+1), get_wcss_list(max_k), marker="o")

ax[1].set_title("Sklearn")
ax[1].set_xlabel("Number of clusters (K)")
ax[1].set_ylabel("With-in sum of squares")
ax[1].plot(np.arange(1, max_k+1), sklearn_get_wcss_list(max_k), marker="o")

plt.show()