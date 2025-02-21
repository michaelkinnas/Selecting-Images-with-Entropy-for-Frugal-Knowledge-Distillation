from numpy import unique, where, argsort

def highest_score(scores, _labels, num_images):
    '''
    Selects the top N samples based on scores and returns their corresponding indices in the dataset.
    '''
    print("SELECTION: Highest Score from dataset")
    sorted_indices = argsort(scores)        
    selected_indices = sorted_indices[-num_images:]
    return selected_indices


def highest_score_per_category(scores, labels, num_images_per_category):
    '''
    Selects the top N samples based on scores for each label and returns their corresponding indices in the dataset.
    '''
    print("SELECTION: Highest Score per Category")
    unique_labels = unique(labels)
    selected_indices = []
    for label in unique_labels:
        label_indices = where(labels == label)[0]
        label_entropies = [scores[i] for i in label_indices]
        sorted_indices = argsort(label_entropies)
        selected_label_indices = label_indices[sorted_indices[-num_images_per_category:]]
        selected_indices.extend(selected_label_indices)
    return selected_indices



def lowest_score_per_category(scores, labels, num_images_per_category):
    '''
    Selects the bottom N samples based on scores for each label and returns their corresponding indices in the dataset.
    '''
    print("SELECTION: Lowest Score per category")
    unique_labels = unique(labels)
    selected_indices = []
    for label in unique_labels:
        label_indices = where(labels == label)[0]
        label_entropies = [scores[i] for i in label_indices]
        sorted_indices = argsort(label_entropies)
        selected_label_indices = label_indices[sorted_indices[:num_images_per_category]]
        selected_indices.extend(selected_label_indices)
    return selected_indices



def highest_score_per_cluster(scores, clustered_indices, n_samples):
    '''
    Selects the top N samples based on scores for each cluster and returns their corresponding indices in the dataset.
    '''
    print("SELECTION: Highest Score per Cluster")
    selected_indices = []
    for cluster_index in clustered_indices.keys():
        cluster_scores = [scores[i] for i in clustered_indices[cluster_index]]
        sorted_indices = argsort(cluster_scores)
        selection = round(len(clustered_indices[cluster_index]) / len(scores) * n_samples)
        if selection > 0:
            selected_cluster_indices = [clustered_indices[cluster_index][x] for x in [sorted_indices[-selection:]][0]]
            selected_indices.extend(selected_cluster_indices)
    return selected_indices



def lowest_score_per_cluster(scores, clustered_indices, n_samples):
    '''
    Selects the top N samples based on scores for each cluster and returns their corresponding indices in the dataset.
    '''
    print("SELECTION: Lowest Score per Cluster")
    selected_indices = []
    for cluster_index in clustered_indices.keys():
        cluster_scores = [scores[i] for i in clustered_indices[cluster_index]]
        sorted_indices = argsort(cluster_scores)
        selection = round(len(clustered_indices[cluster_index]) / len(scores) * n_samples)
        if selection > 0:
            selected_cluster_indices = [clustered_indices[cluster_index][x] for x in [sorted_indices[:selection]][0]]
            selected_indices.extend(selected_cluster_indices)
    return selected_indices