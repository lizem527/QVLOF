import hashlib

from offline.layout import *
from sklearn.preprocessing import MinMaxScaler
import math
import collections
from tqdm import tqdm


def decimalToBinary(n, max_bits):
    raw = bin(n).replace("0b", "")
    return '0' * (max_bits - len(raw)) + raw


def get_top_columns3(cfg, workload, cutoff=2):
    cnt = {}
    for q in workload:
        leaves = q.get_leaves()
        for leaf in leaves:
            col = leaf.col
            if not col in cnt:
                cnt[col] = 0
            cnt[col] += 1
    # Heuristic: date and numeric columns before categorical columns
    for col in cnt:
        if col in cfg["date_cols"]:
            cnt[col] += 0.2
        elif col in cfg["num_cols"]:
            cnt[col] += 0.1
        if col in cfg["sort_cols"]:
            cnt[col] += 0.1
    sorted_cnt = dict(sorted(cnt.items(), key=lambda item: item[1], reverse=True))
    return list(sorted_cnt.keys())[:min(cutoff, len(sorted_cnt))]


class Hash(Layout):
    def __init__(self, df, cfg, k, max_bits=10, workload=None, verbose=True, cols=''):
        super().__init__(df, cfg, k)
        self.max_bits = max_bits
        self.bits = self.max_bits
        self.verbose = verbose
        if workload is not None:
            self.sort_cols = get_top_columns3(cfg, workload)
        elif len(cols) != 0:
            self.sort_cols = cols
        else:
            self.sort_cols = cfg["sort_cols"]

    def _remove_unused_cols(self, cfg, df):
        # Extract columns that are involved in the query
        cols = []
        idx = []
        for i, col in enumerate(df.columns):
            if not col in self.sort_cols:
                continue
            cols.append(col)
            idx.append(i)
        df = df[cols]
        # Store column types
        types = []
        for col in df.columns:
            if col in cfg["num_cols"]:
                types.append("num")
            elif col in cfg["cat_cols"]:
                types.append("cat")
            else:
                types.append("date")
        df = df.reindex(columns=self.sort_cols)
        self.d = len(cols)
        self.types = types
        return df

    def _transform(self, df):
        """Prepare columns values for hash"""
        new_df = collections.OrderedDict()
        for i, col in enumerate(df.columns):
            vals = df[col].values
            # Normalize numeric values to integers in [0, 2^(max_bits)]
            if self.types[i] == "num":

                scaler = MinMaxScaler()
                vals = vals.reshape(-1, 1)
                scaler.fit(vals)
                new_vals = np.squeeze(scaler.transform(vals)) * (math.pow(2, self.max_bits) - 1)
                new_vals = new_vals.astype(int)
            # Transform strings into binary by keeping track of unique values
            else:
                # Integer encode strings via alphabetical order
                # This is needed since date columns have orders
                mapping = {}
                unique = sorted(list(set(vals)))
                for val in unique:
                    mapping[val] = len(mapping)
                new_vals = []
                for val in vals:
                    new_vals.append(mapping[val])
                num_bits = int(math.log2(len(mapping)))
                if math.pow(2, num_bits) < len(mapping):
                    num_bits += 1
                self.bits = max(self.max_bits, num_bits)
            new_df[col] = new_vals
        new_dataframe = pd.DataFrame.from_dict(new_df)
        return new_dataframe.values

    def _get_hash(self):
        if self.verbose:
            print("Computing hash...")
        hash_order = []
        partitions = [[] for _ in range(self.k)]
        avg_count = self.N // self.k + 1
        if self.verbose:
            progress = tqdm(total=self.N, miniters=1000)
        for i in range(self.N):
            value = self.vals[i][0].astype(str)
            hash_value = int(hashlib.md5(value.encode()).hexdigest(), 16) % self.k
            partitions[hash_value].append(i)
            hash_order.append(hash_value)
            if self.verbose:
                progress.update()

        # 第二步：处理超出平均数据数的分区
        if self.verbose:
            progress = tqdm(total=self.k)
        for i in range(self.k):
            while len(partitions[i]) > avg_count:
                # 取出超出的部分
                overflow_index = partitions[i].pop()

                # 尝试根据第二列的哈希值重新分配
                if len(self.sort_cols) != 1:
                    second_col_hash = int(hashlib.md5(str(self.vals[overflow_index][1]).encode()).hexdigest(), 16)
                    second_col_partition = second_col_hash % self.k
                else:
                    second_col_partition = hash_order[overflow_index]

                if len(partitions[second_col_partition]) < avg_count:
                    # 如果第二列哈希值对应的分区未满，则分配到该分区
                    partitions[second_col_partition].append(overflow_index)
                    hash_order[overflow_index] = second_col_partition
                else:
                    # 否则，按顺序逐个寻找未满的分区
                    for j in range(self.k):
                        if len(partitions[j]) < avg_count:
                            partitions[j].append(overflow_index)
                            hash_order[overflow_index] = j
                            break
            if self.verbose:
                progress.update()
        return hash_order

    def _get_labels(self):
        df = self._remove_unused_cols(self.cfg, self.df)
        self.vals = self._transform(df)
        self.labels = self._get_hash()

    def make_partitions(self):
        self._get_labels()
        self.load_by_labels(self.labels)

    def save_by_path(self, path):
        self.path = path
        pickle.dump(self.labels, open(self.path, "wb"))

    def load_by_path(self, path):
        labels = pickle.load(open(path, "rb"))
        self.path = path
        self.load_by_labels(labels)
