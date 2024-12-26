import numpy as np
import concurrent.futures
import warnings
warnings.filterwarnings('ignore') 


def kmeans_seq(X, k, max_iter=300):
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    for _ in range(max_iter):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
    return centroids, labels


def parallel_assignment_future(X_chunk, centroids):
    distances = np.linalg.norm(X_chunk[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)


def kmeans_par(X, k, max_iter=300, n_jobs=None):
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    for _ in range(max_iter):
        chunk_size = X.shape[0] // n_jobs
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(
                lambda chunk: parallel_assignment_future(chunk, centroids), 
                [X[i:i + chunk_size] for i in range(0, X.shape[0], chunk_size)])
            )
        labels = np.concatenate(results)
        centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
    return centroids, labels

