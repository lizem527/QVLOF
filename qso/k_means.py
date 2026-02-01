import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances


class EqualSizeKMeans:
    def __init__(
        self,
        n_clusters,
        size,
        n_init=10,
        random_state=None,
        batch_size=5000,
    ):
        self.n_clusters = n_clusters
        self.size = size
        self.n_init = n_init
        self.random_state = random_state
        self.batch_size = batch_size

        self.cluster_centers_ = None
        self.labels_ = None

    def fit_predict(self, X):
        N, D = X.shape
        assert N == self.n_clusters * self.size, \
            "N must equal n_clusters * size"

        # --------------------------------------------------
        # Step 1: learn centroids
        # --------------------------------------------------
        kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            n_init=self.n_init,
            batch_size=self.batch_size,
            random_state=self.random_state,
        )
        kmeans.fit(X)
        centers = kmeans.cluster_centers_
        self.cluster_centers_ = centers

        # --------------------------------------------------
        # Step 2: first-pass greedy assignment (batched)
        # --------------------------------------------------
        labels = -np.ones(N, dtype=np.int32)
        cluster_sizes = np.zeros(self.n_clusters, dtype=np.int32)

        # process points in batches
        order = np.arange(N)
        np.random.shuffle(order)

        for start in range(0, N, self.batch_size):
            end = min(start + self.batch_size, N)
            idx = order[start:end]
            Xb = X[idx]

            dists = pairwise_distances(Xb, centers)
            pref = np.argsort(dists, axis=1)

            for i, prefs in zip(idx, pref):
                for k in prefs:
                    if cluster_sizes[k] < self.size:
                        labels[i] = k
                        cluster_sizes[k] += 1
                        break

        # --------------------------------------------------
        # Step 3: global repair (guarantees success)
        # --------------------------------------------------
        unassigned = np.where(labels < 0)[0]
        if len(unassigned) > 0:
            free_clusters = []
            for k in range(self.n_clusters):
                free_clusters.extend(
                    [k] * (self.size - cluster_sizes[k])
                )
            free_clusters = np.array(free_clusters)

            assert len(free_clusters) == len(unassigned)

            # compute distances for repair
            dists = pairwise_distances(
                X[unassigned],
                centers[free_clusters]
            )

            best = np.argmin(dists, axis=1)

            for i, k_idx in enumerate(best):
                k = free_clusters[k_idx]
                labels[unassigned[i]] = k
                cluster_sizes[k] += 1

        # --------------------------------------------------
        # Final sanity check
        # --------------------------------------------------
        assert np.all(labels >= 0)
        assert np.all(cluster_sizes == self.size)

        self.labels_ = labels
        return labels
def main():
    np.random.seed(0)

    N = 100_000   # 先测试 10w
    D = 128
    K = 100
    SIZE = N // K

    X = np.random.randn(N, D).astype(np.float32)

    km = EqualSizeKMeans(
        n_clusters=K,
        size=SIZE,
        n_init=5,
        random_state=42,
        batch_size=5000,
    )

    labels = km.fit_predict(X)

    unique, counts = np.unique(labels, return_counts=True)
    print("min/max cluster size:", counts.min(), counts.max())


if __name__ == "__main__":
    main()
