from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import numpy as np


def kmeans_clustering(esm_embeddings, k=2, random_state=42):
    """
    Cluster ESM embeddings with k-means.
    """
    km = KMeans(n_clusters=k, random_state=random_state)
    labels = km.fit_predict(np.array(esm_embeddings))
    return labels


def tsne_dim_reduction(esm_embeddings, dim=2, random_state=42):
    """
    Reduce high-dimensional ESM embeddings with t-SNE.
    """
    tsne = TSNE(n_components=dim, random_state=random_state)
    coords = tsne.fit_transform(np.array(esm_embeddings))
    return coords

