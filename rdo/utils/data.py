import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join


def get_query_cols(q_config):
    cols = []
    for id in q_config:
        cols.extend(list(q_config[id].keys()))
    return list(set(cols))


def get_filenames(path):
    files = []
    for f in listdir(path):
        if isfile(join(path, f)) and ('csv' in f or 'tbl' in f):
            files.append(f)
    return files


def load_csv(file, config):
    dim = config["dim"]
    df = pd.read_csv(
        file,
        delimiter=config["delimiter"],
        header=None,
        usecols=range(dim),
        dtype=float
    )
    return df


def load_df(config, concat=True):
    fnames = get_filenames(config["path"])
    files = [join(config["path"], f) for f in fnames]
    dim = config["dim"]

    dfs = []
    for file in files:
        df = pd.read_csv(
            file,
            delimiter=config["delimiter"],
            header=None,
            usecols=range(dim),
            dtype=float
        )

        dfs.append(df)
    if concat:
        return df, config["ds"]
    else:
        return dfs, fnames
