from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


def knn_graph(features: np.ndarray, k: int = 12) -> sparse.csr_matrix:
    n = features.shape[0]
    k = min(k, max(1, n - 1))
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(features)
    distances, indices = nn.kneighbors(features)
    rows = np.repeat(np.arange(n), k)
    cols = indices[:, 1:].reshape(-1)
    sigma = np.median(distances[:, 1:]) or 1.0
    weights = np.exp(-((distances[:, 1:].reshape(-1) ** 2) / (2 * sigma**2)))
    adj = sparse.coo_matrix((weights, (rows, cols)), shape=(n, n))
    adj = adj.maximum(adj.T).tocsr()
    return adj


def normalized_laplacian(adj: sparse.csr_matrix) -> sparse.csr_matrix:
    degree = np.asarray(adj.sum(axis=1)).ravel()
    degree[degree == 0] = 1.0
    inv_sqrt = sparse.diags(1.0 / np.sqrt(degree))
    identity = sparse.identity(adj.shape[0], format="csr")
    return identity - inv_sqrt @ adj @ inv_sqrt

