import numpy as np
from sklearn.metrics import silhouette_score


def count_clusters(labels):
    return len(set(labels) - {-1})


def count_noise(labels):
    return int(np.sum(labels == -1))


def compute_silhouette(X, labels):
    n_clusters = count_clusters(labels)
    if n_clusters < 2:
        return -1.0
    mask = labels != -1
    if mask.sum() < 2:
        return -1.0
    try:
        return float(silhouette_score(X[mask], labels[mask]))
    except ValueError:
        return -1.0


def compute_all_metrics(X, labels):
    return {
        "silhouette": compute_silhouette(X, labels),
        "n_clusters": count_clusters(labels),
        "n_noise": count_noise(labels),
    }
