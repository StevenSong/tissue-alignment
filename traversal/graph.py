import os

import numpy as np
import pandas as pd
from pqdm.processes import pqdm
from scipy.sparse.csgraph import shortest_path
from sklearn.metrics import pairwise_distances


def get_hex_grid_adj_matrix(pos_df):
    # TODO can probably precompute this whole matrix and simply sort by spot ordering...
    n = len(pos_df)
    rows = np.asarray(pos_df[["row"]])
    cols = np.asarray(pos_df[["col"]])
    rows_mx = np.broadcast_to(rows, (n, n))
    cols_mx = np.broadcast_to(cols, (n, n))
    adj_spots = (
        # spots 1 row away are 1 col away
        ((rows_mx == (rows_mx - 1).T) | (rows_mx == (rows_mx + 1).T))
        & ((cols_mx == (cols_mx - 1).T) | (cols_mx == (cols_mx + 1).T))
    ) | (
        # spots in same row are 2 cols away
        (rows_mx == rows_mx.T)
        & ((cols_mx == (cols_mx - 2).T) | (cols_mx == (cols_mx + 2).T))
    )
    assert adj_spots.sum(axis=0).max() <= 6
    return adj_spots


def compute_distance_matrix(embeddings, pos_df):
    # computes both kinds of adjacency in case expression averaging needs
    distances = pairwise_distances(embeddings, metric="euclidean")

    # hex grid adjacency
    hex_adj_mx = get_hex_grid_adj_matrix(pos_df=pos_df)

    return distances, hex_adj_mx


def compute_path_idxs(distances, hex_adj_mx, start_idx, end_idx):
    # 0 is unreachable by algorithm
    _distances = np.where(hex_adj_mx, distances, 0)

    path_lens, predecessors = shortest_path(
        csgraph=_distances,
        method="D",
        directed=False,
        return_predecessors=True,
        indices=start_idx,
    )
    path_idxs = [end_idx]
    prev_idx = predecessors[end_idx]
    while prev_idx != start_idx:
        path_idxs.append(prev_idx)
        prev_idx = predecessors[prev_idx]
    path_idxs.append(start_idx)
    path_idxs = path_idxs[::-1]
    return path_idxs  # either list or list of lists


def compute_clusters(embeddings, centroids, spot_frac=1):
    assert spot_frac >= 0 and spot_frac <= 1
    n = len(embeddings)
    k = len(centroids)

    distances = np.linalg.norm(
        embeddings - np.repeat(centroids[:, None, :], n, axis=1),
        axis=-1,
    )  # euclidean distance
    clusters = distances.argmin(axis=0)
    if spot_frac != 1:
        spot_n = max(k, int(spot_frac * n))
        dist_to_cluster = distances[clusters, range(len(clusters))]
        clusters[dist_to_cluster.argsort()[spot_n:]] = -1
    return clusters


TEMP = {}


def _get_cluster_count(i):
    pos_df = TEMP["pos_df"]
    counts = TEMP["counts"]
    clusters = TEMP["clusters"]
    cluster_idxs = clusters == i
    counts_idxs = pos_df.loc[cluster_idxs, "barcode"]
    cluster_counts = counts.loc[counts_idxs].sparse.to_dense()
    mean = cluster_counts.mean(axis=0)
    std = cluster_counts.std(axis=0)
    return mean, std

    # total_UMI = cluster_counts.to_numpy().sum()
    # norm_mean = np.log(mean / total_UMI)
    # norm_std = np.log(std / total_UMI)
    # return norm_mean, norm_std


def compute_path_counts(
    pos_df,
    counts,
    path_idxs,
    clusters,
):
    TEMP["pos_df"] = pos_df
    TEMP["counts"] = counts
    TEMP["clusters"] = clusters

    counts = pqdm(
        range(len(path_idxs)),
        _get_cluster_count,
        n_jobs=min(len(path_idxs), os.cpu_count()),
    )
    mean_counts = pd.concat([m for m, s in counts], axis=1).T
    std_counts = pd.concat([s for m, s in counts], axis=1).T

    # mean_counts = []
    # std_counts = []
    # for i in range(len(path_idxs)):
    #     mean_count, std_count = _get_cluster_count(i)
    #     mean_counts.append(mean_count)
    #     std_counts.append(std_count)
    # mean_counts = pd.concat(mean_counts, axis=1).T
    # std_counts = pd.concat(std_counts, axis=1).T

    return mean_counts, std_counts
