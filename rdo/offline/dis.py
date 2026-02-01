import numpy as np
import threading
from offline.layout import *
from offline.transform import *
from utils.hot_start_query import *
from scipy.stats import gaussian_kde
from scipy.spatial import KDTree
import math
import collections
from tqdm import tqdm


def decimalToBinary(n, max_bits):
    raw = bin(n).replace("0b", "")
    return '0' * (max_bits - len(raw)) + raw


def get_represent(df, queries):
    # 1. Data preparation: Ensure that the df matches the queries dimension
    df_array = df.values
    queries_array = np.array(queries)

    # 2. Calculate the KDE of each column in df
    kde_list = []
    for col in range(df_array.shape[1]):
        kde = gaussian_kde(df_array[:, col])
        kde_list.append(kde)

    # 3. calculate KDE
    kde_values = []
    print("calculate KDE...")
    for query in tqdm(queries_array, desc="calculate process"):
        total_kde = 0
        for col, kde in enumerate(kde_list):
            total_kde += kde.evaluate(query[col])
        kde_values.append(total_kde)

    # 4. Select the index corresponding to the highest value of KDE
    max_index = np.argmax(kde_values)

    max_query = queries[max_index]
    max_query = np.array(max_query)

    return max_query


def code_to_label(distances, k):
    indices = np.argsort(distances)
    part_size = len(distances) // k + 1
    bids = np.zeros(len(distances))
    for i in range(k):
        if i == k - 1:
            bids[indices[i * part_size:]] = i
        else:
            bids[indices[i * part_size:(i + 1) * part_size]] = i

    return bids


class DIS(Layout):
    def __init__(self, df, cfg, k, partial=0.06, workload=None, verbose=True, init=None, center=None):
        super().__init__(df, cfg, k)
        self.meta = None
        self.codes = None
        self.init = init
        self.workload = workload
        self.verbose = verbose
        self.part = int(partial * self.k)
        self.scaler = []
        if self.workload is not None:
            if center is not None:
                self.center = center
            else:
                self.center = get_represent(self.df, self.workload)
        else:
            self.center = self._get_kcenter()

        self.R = self._calculate_R()

    def _calculate_R(self):
        # Calculate the R value: Randomly select 1000 points from data and calculate the minimum Euclidean distance between each point and the others
        data = self.df.values

        # Randomly select 1,000 points
        n_samples = min(1000, self.N)
        random_indices = np.random.choice(self.N, n_samples, replace=False)
        sample_points = data[random_indices]

        tree = KDTree(data)
        distances, indices = tree.query(sample_points, k=2)
        # Take the distance of the second nearest neighbor (excluding oneself)
        min_distances = distances[:, 1]

        # Calculate the average minimum distance
        R = np.mean(min_distances)

        return R

    def _get_kcenter(self):

        X = self.df
        center = np.mean(X, axis=0)
        center = np.array(center).flatten()
        return center

    def _get_codes(self):
        # Calculate the dis of data
        print("computing distance")
        distances = np.linalg.norm(self.df - self.center, axis=1)

        return distances

    def _get_labels(self):
        if self.workload is not None:
            self.codes = np.array(self._get_codes())
        else:
            hot_query = generate_high_dataset(self.df)
            processor = DataProcessor()
            self.codes = processor.method3(self.df, hot_query)
        self.labels = code_to_label(self.codes, self.k)

    def compute_meta(self):
        metas = []
        for i in range(self.k):
            pid = int(i)
            filtered_codes = self.codes[self.labels == pid]
            max_part = np.max(filtered_codes)
            min_part = np.min(filtered_codes)
            meta = [max_part, min_part, filtered_codes.size]
            metas.append(meta)
        return metas

    def make_partitions(self):
        self._get_labels()
        self.load_by_labels(self.labels)

    def save_by_path(self, path):
        self.path = path
        pickle.dump(self.labels, open(self.path, "wb"))
        if self.meta is None:
            if self.codes is None:
                self.codes = np.array(self._get_codes())
            self.meta = self.compute_meta()
            # print("len meta:", len(self.meta))

    def load_by_path(self, path):
        labels = pickle.load(open(path, "rb"))
        self.path = path
        self.load_by_labels(labels)

        if self.meta is None:
            if self.codes is None:
                self.codes = np.array(self._get_codes())
            self.meta = self.compute_meta()

    def eval(self, queries, avg=True):
        read = [0] * len(queries)
        read_pids = []

        R = self.R

        for i, query in enumerate(queries):

            q1 = np.array(query)
            pids = []

            if self.init is not None:
                dis = np.linalg.norm(self.center - q1)
                d_max = R + dis
                if R < dis:
                    d_min = dis - R
                else:
                    d_min = 0
                # part can search
                if d_max <= self.meta[self.part - 1][0]:
                    for t in range(self.part - 1):
                        pid = int(t)
                        # print("pid,",pid,self.meta[pid])
                        if self.meta[pid][0] >= d_min and self.meta[pid][1] <= d_max:
                            read[i] += self.meta[pid][2]
                    read_pids.append(pids)
                # part can't, use full(init)
                else:
                    dis2 = np.linalg.norm(self.init.center - q1)
                    d_max = R + dis2
                    if R < dis2:
                        d_min = dis2 - R
                    else:
                        d_min = 0
                    for t in range(self.k):
                        pid = int(t)
                        if self.init.meta[pid][0] >= d_min and self.init.meta[pid][1] <= d_max:
                            read[i] += self.init.meta[pid][2]
                    read_pids.append(pids)
            else:
                dis = np.linalg.norm(self.center - q1)
                d_max = R + dis
                if R < dis:
                    d_min = dis - R
                else:
                    d_min = 0
                for t in range(self.k):
                    pid = int(t)
                    if self.meta[pid][0] >= d_min and self.meta[pid][1] <= d_max:
                        read[i] += self.meta[pid][2]
                        pids.append(pid)
                read_pids.append(pids)

        read = np.array(read) * 1.0 / self.N
        if avg:
            return np.average(read), read_pids
        else:
            return read, read_pids
