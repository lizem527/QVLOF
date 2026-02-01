from utils.data import *
from utils.predicates import *
from utils.workload import *
import json
import os


def part_split(files, fnames, k, equal=False):
    """Split partition count budget across files"""
    n = len(files)
    if not equal:
        sizes = []
        for f in files:
            sizes.append(os.path.getsize(f))
            print("file", f, sizes)
        sizes = np.array(sizes) * 1. / sum(sizes) * k
        parts = [int(round(s)) for s in sizes]
    else:
        parts = [k // n] * n
    if sum(parts) < k:
        delta = k - sum(parts)
        indices = np.random.choice(range(n), delta, replace=False)
        for i in indices:
            parts[i] += 1
    elif sum(parts) > k:
        delta = sum(parts) - k
        indices = np.random.choice(range(n), delta, replace=False)
        for i in indices:
            parts[i] -= 1
    ans = {}
    for i, fname in enumerate(fnames):
        ans[fname] = parts[i]
    return ans


def get_workload(config, df, fname, args, tot_queries=10000):
    """Load query workload"""
    #gen = PredicateGenerator(config, fname)
    # Randomly generating queries according to schema
    if "random" in args.q:
        queries = gen_with_clustering(df, tot_queries, n_clusters=100, noise_level=0.01)

    # Pickled query file
    else:
        queries = pickle.load(open("resources/query/%s.p" % args.q, "rb"))
    return queries


def get_workload_perfile(config, files, fnames, args):
    """Load query workload per input file"""
    if "random" in args.q:
        counts = part_split(files, fnames, args.k, True)
        queries = []
        for i, file in enumerate(files):
            df = load_csv(file, config)
            qs = get_workload(config, df, fnames[i], args, counts[fnames[i]])
            queries.extend(qs)
    else:
        qfile, ext = os.path.splitext(fnames)
        queries_temp = pickle.load(open("resources/pre_query/%s.p" % qfile, "rb"))
        queries = new_gen_workload(config, queries_temp)

    return queries


def new_get_workload(config, files, fnames, args):
    """Load query workload per input file"""
    dim=config["dim"]
    if "random" in args.q:
        counts = part_split(files, fnames, args.k, True)
        queries = []
        for i, file in enumerate(files):
            df = load_csv(file, config)
            qs = get_workload(config, df, fnames[i], args, counts[fnames[i]])
            queries.extend(qs)
    else:
        qfile, ext = os.path.splitext(fnames)
        query_file = "resources/pre_query/%s_q.csv" % qfile
        df = pd.read_csv(
            query_file,
            delimiter=config["delimiter"],
            header=None,
            usecols=range(dim),
            dtype=float
        )

        queries = df.values.tolist()

    return queries


def get_data(config, args, parts, file, fname, frac=0.001):
    """Load query workload per input file"""
    df = load_csv(file, config)
    k = parts[fname]
    sample_size = int(min(max(k * 10000, frac * len(df)), len(df)))
    df_sample = df.sample(n=sample_size, random_state=args.seed).reset_index(drop=True)
    return df, df_sample, k


def setup_perfile(args):
    with open("resources/config/%s.json" % args.config, mode="r") as f:
        config = json.load(f)
    fnames = get_filenames(config["path"])
    files = [join(config["path"], f) for f in fnames]
    print(args.k, args.equal)
    parts = part_split(files, fnames, args.k, args.equal)
    return fnames, files, parts, config


def setup(args, frac=0.001, tot_queries=10000):
    with open("resources/config/%s.json" % args.config, mode="r") as f:
        config = json.load(f)
    df, fname = load_df(config)
    sample_size = int(min(max(args.k * 10000, frac * len(df)), len(df)))
    df_sample = df.sample(n=sample_size, random_state=args.seed).reset_index(drop=True)

    # Generate workload
    queries = get_workload(config, df, fname, args, tot_queries)
    return df, df_sample, queries, config
