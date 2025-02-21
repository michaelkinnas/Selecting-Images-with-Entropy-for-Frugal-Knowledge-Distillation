from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from numpy import argmax


def kmeans_clustering(data, n_clusters, seed) -> dict:
    '''
    Perform clustering on representation vectors and return
    a dictionary of cluster keys and the dataset sample indices that belong in each cluster.
    '''
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=seed)
    clusters = kmeans.fit_predict(X=data)
    clustered_data_indices = {x:[] for x in range(n_clusters)}
    for index, element, in enumerate(clusters):
        clustered_data_indices[element].append(index)
    return clustered_data_indices


def silhouete_score(data, seed):
    '''
    Calculate optimal number of kmeans clusters for given data
    '''
    sil = []
    start = 10
    kmax = 30
    # dissimilarity would not be defined for a single cluster, thus, 
    # minimum number of clusters should be 2
    for k in range(start, kmax+1, 2):
        kmeans = KMeans(n_clusters = k, n_init="auto", random_state=seed).fit(data)
        labels = kmeans.labels_
        sil.append(silhouette_score(data, labels, metric = 'euclidean'))
    # print(sil)
    return argmax(sil).item() + 2