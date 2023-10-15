import numpy as np
from sklearn.cluster import KMeans

from cluster_info import ClusterInfo


def create_clusters(dates, features, n_clusters):
    # Clusters
    kmeans = KMeans(init="k-means++", n_clusters=n_clusters,
                    n_init=10, tol=1e-6)
    kmeans.fit(features)

    cluster_ids = kmeans.fit_predict(features)
    clusterSizes = np.unique(cluster_ids, return_counts=True)[1]

    print(f'Clusters: {n_clusters}')
    print(f'Cluster Sizes: {clusterSizes}')

    clustersToSmall = np.vectorize(
        lambda x: 1 if x <= 2 else 0)(clusterSizes).sum()

    if clustersToSmall > 0:
        raise Exception('Skipping because cluster is to small!')

    # Create Cluster Objects
    numOfDays = len(dates)
    clusters = [ClusterInfo(i) for i in range(n_clusters)]

    for i in range(numOfDays):
        dayInfo = dates[i]
        cluster_id = cluster_ids[i]
        cluster = clusters[cluster_id]

        cluster.add_date(dayInfo)

    # Sort by Size
    clusters.sort(key=lambda x: x.size(), reverse=True)

    # Sort Dates
    for cluster in clusters:
        cluster.sort_dates()

    return clusters
