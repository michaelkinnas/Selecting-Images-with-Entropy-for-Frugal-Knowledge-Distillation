def top_n_per_category(data, labels, n_samples, representations_fn=None, evaluator_fn=None, selector_fn=None):
    '''
    Calculate scores and select the top n samples per category from a given dataset.
    '''
    print("METHODS: Top N per category")
    if representations_fn is not None:
        print("Calculating representations...")
        data = representations_fn(data)
    print("Calculating scores...")
    scores = evaluator_fn(data)
    print("Selecting images...")
    selected_indices = selector_fn(scores, labels, n_samples // 10)
    return selected_indices


def top_n(data, labels, n_samples, representations_fn=None, evaluator_fn=None, selector_fn=None):
    '''
    Calculate scores and select the top n samples from a given dataset.
    '''
    print("METHODS: Top N from dataset")
    if representations_fn is not None:
        print("Calculating representations...")
        data = representations_fn(data)
    print("Calculating scores...")
    scores = evaluator_fn(data)
    print("Selecting images...")
    print(n_samples)
    selected_indices = selector_fn(scores, labels, n_samples)
    return selected_indices


def kmeans_clustering(data, n_samples, representations_fn, evaluator_fn, selector_fn, n_clusters=None, seed=None):
    from utils.clustering import kmeans_clustering, silhouete_score
    print("METHODS: KMeans Clustering")
    print("Calculating representations...")
    representations = representations_fn(data)
    print("Calculating scores...")
    scores = evaluator_fn(representations)

    if n_clusters is None:
        print("Calculating silhouete score...")
        n_clusters = silhouete_score(representations, seed)
        print(f"Calculated {n_clusters} clusters for kmeans.")

    print("Clustering data...")
    clustered_data_indices = kmeans_clustering(data=representations, n_clusters=n_clusters, seed=seed)
    # Number of samples per cluster is representative of cluster size

    print("Selecting images...")
    selected_indices = selector_fn(scores=scores,
                                    clustered_indices=clustered_data_indices,
                                    n_samples=n_samples)


    return selected_indices, n_clusters

 
def kmeans_clustering_per_category(data, labels, n_samples, evaluator_fn, selector_fn, representations_fn=None,
                                  n_clusters=None, seed=None):
    from utils.clustering import kmeans_clustering, silhouete_score
    from torch import Tensor
    print("METHODS: KMeans Clustering per Category")

    if representations_fn is not None:
        print("Calculating representations...")
        data = representations_fn(data)
    print("Calculating scores...")
    scores = evaluator_fn(data)

    selected_samples = []
    clusters_per_label = []

    if isinstance(labels[0], Tensor):
        labels = [x.item() for x in labels]

    label_indices = {x:[] for x in set(labels)}
    for i, label in enumerate(labels):
        label_indices[label].append(i)

    for label in label_indices.keys():
        print(f"Processing label {label}")

        label_representations = [data[x] for x in label_indices[label]]
        label_scores = [scores[x] for x in label_indices[label]]

        if n_clusters is None:
            print("Calculating silhouete score...")
            calculated_clusters = silhouete_score(label_representations, seed)
            print(f"Calculated {calculated_clusters} clusters for label {label}.")
        else:
            calculated_clusters = n_clusters

        clusters_per_label.append(calculated_clusters)

        print("Clustering data...")
        clustered_label_data_indices = kmeans_clustering(data=label_representations, n_clusters=calculated_clusters, seed=seed)

        print("Selecting images...")
        selected_label_indices = selector_fn(scores=label_scores,
                                    clustered_indices=clustered_label_data_indices,
                                    n_samples=n_samples // len(set(labels)))

        selected_samples.extend([label_indices[label][x] for x in selected_label_indices])

    return selected_samples, clusters_per_label

# Manifold learning
# Step 1: Manifold Learning
def _manifold_learning(data, n_components):
    from sklearn.manifold import TSNE
    # Perform t-SNE manifold learning
    tsne = TSNE(n_components=n_components)
    embedded_data = tsne.fit_transform(data)
    return embedded_data

# Step 2: K-Means Clustering
def _kmeans_clustering(data, n_clusters):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(data)
    return clusters

# Step 3: Iteratively Sample Images
def _iterative_sampling(clusters, num_samples, seed):
    from numpy import where, unique, random
    random.seed(seed)
    k = 0  # cluster index
    sampled_indices = set()
    samples = []
    for i in range(num_samples):
        cluster_indices = where(clusters == k)[0]
        available_indices = list(set(cluster_indices) - sampled_indices)
        if not available_indices:
            # If no available indices in the cluster, skip to the next cluster
            k = (k + 1) % len(unique(clusters))
            continue
        sample_index = random.choice(available_indices)
        samples.append(sample_index)
        sampled_indices.add(sample_index)
        k = (k + 1) % len(unique(clusters))  # move to next cluster or wrap around
    return samples

def manifold_learning(images, n_samples, tsne_n_components, kmeans_clusters, seed):
    from numpy import array
    '''
    Select subset of images using manifold-based sampling
    '''
    print("Method: Manifold Learning")
    data_flat = array([array(image).flatten() for image in images]) / 255.0
    embedded_data = _manifold_learning(data_flat, n_components=tsne_n_components)
    clusters = _kmeans_clustering(embedded_data, kmeans_clusters)
    sampled_indices = _iterative_sampling(clusters, n_samples, seed)
    return sampled_indices
