import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import colormaps
from matplotlib.collections import PatchCollection
from matplotlib.colors import TABLEAU_COLORS, to_rgba
from matplotlib.patches import Circle
from PIL import Image
from scipy.io import mmread
from scipy.sparse.csgraph import shortest_path
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import PAIRWISE_DISTANCE_FUNCTIONS
from tqdm import tqdm

COLORS = list(TABLEAU_COLORS)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adjacency", choices=["hex", "embed"], default="hex")
    parser.add_argument(
        "--avg-expression",
        choices=["off", "hex", "embed", "path-clusters"],
        default="off",
    )
    parser.add_argument("--num-neighbors", type=int, default=6)
    parser.add_argument("--data-root", default="/mnt/data5/spatial")
    parser.add_argument("--sections", default=["slide3/A1"], nargs="+")
    parser.add_argument("--model", default="triplet-old-all-slides-0999")
    parser.add_argument(
        "--distance-metric",
        choices=sorted(list(PAIRWISE_DISTANCE_FUNCTIONS.keys())),
        default="euclidean",
    )
    parser.add_argument(
        "--path-alg",
        choices=["FW", "D", "BF", "J"],
        default="D",
        help="see available methods at https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.shortest_path.html",
    )
    parser.add_argument("--fullres", action="store_true")
    parser.add_argument("--genes", nargs="+", default=["EPCAM", "ACTA2"])
    parser.add_argument("--figsize", type=int, default=8)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cluster-frac", type=float, default=1)
    parser.add_argument("--window-size", type=int, default=5)

    args = parser.parse_args()
    return args


def read_spatial_data(count_path, fullres):
    pos_df = pd.read_csv(
        os.path.join(count_path, "spatial/tissue_positions_list.csv"),
        header=None,
        names=[
            "barcode",
            "in_tissue",
            "row",
            "col",
            "y_fullres",
            "x_fullres",
        ],
    )
    pos_df = pos_df.loc[pos_df["in_tissue"] == 1].reset_index(drop=True)
    pos_df[["y", "x"]] = pos_df[["y_fullres", "x_fullres"]].copy()
    with open(os.path.join(count_path, "spatial/scalefactors_json.json")) as f:
        scale_factors = json.loads(f.read())
    spot_radius = int(round(scale_factors["spot_diameter_fullres"] / 2))
    if not fullres:
        hires_scale = scale_factors["tissue_hires_scalef"]
        spot_radius = round(spot_radius * hires_scale)
        pos_df[["y", "x"]] *= hires_scale
    return pos_df, spot_radius


def read_transcription_data(count_path, pos_df, genes):
    features = pd.read_csv(
        os.path.join(count_path, "filtered_feature_bc_matrix/features.tsv.gz"),
        sep="\t",
        names=["id", "gene", "type"],
    )
    barcodes = pd.read_csv(
        os.path.join(count_path, "filtered_feature_bc_matrix/barcodes.tsv.gz"),
        sep="\t",
        names=["barcode"],
    )

    target = 10_000  # 10_000 counts per spot (or per slide)
    mat = mmread(os.path.join(count_path, "filtered_feature_bc_matrix/matrix.mtx.gz"))
    t = mat.shape[1]  # number of spots in slide
    z = mat.sum()  # sum of all counts in slide
    # t = t if use_slide_size else 1
    mat = mat * t / z * target  # (z/t) is average count per spot for the slide
    mat = mat.log1p()

    counts = pd.DataFrame.sparse.from_spmatrix(mat).T
    counts.index = barcodes["barcode"]
    counts.columns = features["gene"]
    # use pos_df ordering of spots
    counts = counts.loc[pos_df["barcode"]].reset_index()
    assert counts["barcode"].equals(pos_df["barcode"])

    for gene in genes:
        assert gene in counts.columns

    return counts


def read_embedding_data(data_root, model, section):
    return torch.load(os.path.join(data_root, f"embeddings/{model}/embeddings.pt"))[
        section
    ].numpy()


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


def get_embedding_adj_matrix(distances, num_neighbors):
    n = len(distances)
    neighbors = distances.argsort(axis=1)[:, :num_neighbors]
    mask = np.zeros_like(distances, dtype=bool)
    mask[np.arange(n)[:, None], neighbors] = 1
    assert mask.sum() == n * num_neighbors
    return mask


def compute_distance_matrix(embeddings, distance_metric, pos_df, num_neighbors):
    # computes both kinds of adjacency in case expression averaging needs
    distances = pairwise_distances(embeddings, metric=distance_metric)

    # hex grid adjacency
    hex_adj_mx = get_hex_grid_adj_matrix(pos_df=pos_df)

    # embedding space adjacency
    embed_adj_mx = get_embedding_adj_matrix(
        distances=distances, num_neighbors=num_neighbors
    )
    return distances, hex_adj_mx, embed_adj_mx


def _get_clusters(embeddings, centroids, cluster_frac=1):
    assert cluster_frac >= 0 and cluster_frac <= 1
    n = len(embeddings)
    k = len(centroids)

    distances = np.linalg.norm(
        embeddings - np.repeat(centroids[:, None, :], n, axis=1),
        axis=-1,
    )  # euclidean distance
    clusters = distances.argmin(axis=0)

    if cluster_frac != 1:
        for i in range(k):
            cluster_idxs = (clusters == i).nonzero()[0]
            cluster_n = len(cluster_idxs)
            cluster_end = max(1, int(cluster_n * cluster_frac))
            cluster_idxs_rank = distances[
                i, cluster_idxs
            ].argsort()  # get sorted order of cluster idxs
            cluster_idxs = cluster_idxs[
                cluster_idxs_rank
            ]  # get sorted order of global idxs
            clusters[cluster_idxs[cluster_end:]] = -1  # set outliers to no cluster
    return clusters


def compute_path_idxs(distances, path_alg, start_idx, end_idx):
    path_lens, predecessors = shortest_path(
        csgraph=distances,
        method=path_alg,
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


def get_path_counts(
    avg_expression,
    counts,
    path_idxs,
    hex_adj_mx,
    embed_adj_mx,
    genes,
    clusters,
):
    if avg_expression == "hex":
        adj_idxs_fn = lambda i_idx: hex_adj_mx[i_idx[1]]
    elif avg_expression == "embed":
        adj_idxs_fn = lambda i_idx: embed_adj_mx[i_idx[1]]
    elif avg_expression == "path-clusters":
        adj_idxs_fn = lambda i_idx: np.nonzero(clusters == i_idx[0])[0]
    else:  # avg_expression == 'off'
        return counts.loc[path_idxs, genes]

    path_counts = {}
    for path_i, path_idx in enumerate(path_idxs):
        adj_idxs = adj_idxs_fn((path_i, path_idx))
        avg = counts.loc[adj_idxs, genes].mean(axis=0)
        path_counts[path_idx] = avg
    path_counts = pd.DataFrame(path_counts).T
    return path_counts


def read_image(data_root, section, fullres):
    if fullres:
        im = Image.open(os.path.join(data_root, "data", section, f"{section[-2:]}.tif"))
    else:
        im = Image.open(
            os.path.join(
                data_root, "count", section, "outs/spatial/tissue_hires_image.png"
            )
        )
    return im


def show_slide(ax, im, pos_df, spot_radius):
    ax.imshow(im)
    circs = PatchCollection(
        [Circle((x, y), spot_radius) for x, y in pos_df[["x", "y"]].to_numpy()],
        picker=True,
    )
    n = len(pos_df)
    facecolors = np.asarray([list(to_rgba("lightgray"))] * n)
    edgecolors = np.asarray([list(to_rgba("lightgray"))] * n)
    alphas = np.full(n, 0.4)
    circs.set_facecolor(facecolors)
    circs.set_edgecolor(edgecolors)
    circs.set_alpha(alphas)
    ax.add_collection(circs)
    return circs, facecolors, edgecolors, alphas


def select_start(ax, pos_df, start_idx, circs, edgecolors, alphas):
    edgecolors[start_idx] = to_rgba(COLORS[0])
    alphas[start_idx] = 1
    ax.annotate(str(0), pos_df.loc[start_idx, ["x", "y"]], color=COLORS[0])
    circs.set_edgecolor(edgecolors)
    circs.set_alpha(alphas)


def select_end(
    ax,
    pos_df,
    start_idx,
    end_idx,
    genes,
    path_alg,
    avg_expression,
    counts,
    distances,
    hex_adj_mx,
    embed_adj_mx,
    adjacency,
    circs,
    facecolors,
    edgecolors,
    alphas,
    embeddings,
    cluster_frac,
    section,
    window_size,
):
    # 0 is unreachable by algorithm
    _distances = np.where(
        hex_adj_mx if adjacency == "hex" else embed_adj_mx,
        distances,
        0,
    )
    path_idxs = compute_path_idxs(
        distances=_distances,
        path_alg=path_alg,
        start_idx=start_idx,
        end_idx=end_idx,
    )

    # get layer crossings
    win_dists = []
    for win_start, win_end in zip(
        path_idxs[: -window_size + 1], path_idxs[window_size - 1 :]
    ):
        curr_win_dist = distances[win_start, win_end]
        win_dists.append(curr_win_dist)
    win_dists = np.asarray(win_dists)
    crossings = []
    win_tol = win_dists.mean() + win_dists.std()
    win_mid = int(np.ceil(window_size / 2))
    contiguous = False
    for i, win_dist in enumerate(win_dists):
        print(win_dist)
        if win_dist > win_tol and not contiguous:
            crossings.append(path_idxs[i + win_mid])
            contiguous = True
        else:
            contiguous = False
    print(crossings)
    print(win_tol)
    # edgecolor for path
    count = 1
    curr_layer = 0
    for path_idx in path_idxs[1:]:  # last index includes end_idx
        if curr_layer < len(crossings) and path_idx == crossings[curr_layer]:
            curr_layer += 1
        edgecolors[path_idx] = to_rgba(COLORS[curr_layer])
        alphas[path_idx] = 1
        ax.annotate(
            str(count), pos_df.loc[path_idx, ["x", "y"]], color=COLORS[curr_layer]
        )
        count += 1

    # facecolor for clusters
    if avg_expression == "path-clusters":
        clusters = _get_clusters(
            embeddings=embeddings,
            centroids=embeddings[path_idxs],
            cluster_frac=cluster_frac,
        )
        cmap = colormaps["gist_rainbow"]
        cmap_interp = np.linspace(0, 1, len(path_idxs))
        facecolors = np.asarray(
            [
                list(cmap(cmap_interp[i])) if i != -1 else facecolors[idx]
                for idx, i in enumerate(clusters)
            ]
        )

    circs.set_facecolor(facecolors)
    circs.set_edgecolor(edgecolors)
    circs.set_alpha(alphas)

    path_counts = get_path_counts(
        avg_expression=avg_expression,
        counts=counts,
        path_idxs=path_idxs,
        hex_adj_mx=hex_adj_mx,
        embed_adj_mx=embed_adj_mx,
        genes=genes,
        clusters=clusters,
    )
    print(f"{section}: Calculated expression along path")
    return path_counts
    # for gene in genes:
    #     ax2.scatter(x=range(len(path_counts)), y=list(path_counts[gene]), label=gene)
    # ax2.set_title("Gene expression along path")
    # ax2.set_ylabel("LogNorm Expression")
    # ax2.set_xlabel("Path Index")
    # ax2.set_xticks(list(range(len(path_counts))))
    # ax2.legend()


def main(args):
    all_path_counts = []
    for section in args.sections:
        print(f"----- {section} -----")
        count_path = os.path.join(args.data_root, "count", section, "outs")
        pos_df, spot_radius = read_spatial_data(
            count_path=count_path, fullres=args.fullres
        )
        print(f"{section}: Loaded spot positions")
        counts = read_transcription_data(
            count_path=count_path, pos_df=pos_df, genes=args.genes
        )
        print(f"{section}: Loaded transcription counts")
        embeddings = read_embedding_data(
            data_root=args.data_root, model=args.model, section=section
        )
        print(f"{section}: Loaded embeddings")
        distances, hex_adj_mx, embed_adj_mx = compute_distance_matrix(
            embeddings=embeddings,
            distance_metric=args.distance_metric,
            pos_df=pos_df,
            num_neighbors=args.num_neighbors,
        )
        print(f"{section}: Computed distances")
        im = read_image(data_root=args.data_root, section=section, fullres=args.fullres)
        print(f"{section}: Loaded image")

        def onpick(event):
            nonlocal start_set, start_idx
            idx = event.ind[0]
            if not start_set:
                start_idx = idx
                start_set = True
                select_start(
                    ax=ax,
                    pos_df=pos_df,
                    start_idx=start_idx,
                    circs=circs,
                    edgecolors=edgecolors,
                    alphas=alphas,
                )
                ax.set_title("Select end spot")
            else:
                end_idx = idx
                fig.canvas.mpl_disconnect(cid)
                path_counts = select_end(
                    ax=ax,
                    pos_df=pos_df,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    genes=args.genes,
                    path_alg=args.path_alg,
                    avg_expression=args.avg_expression,
                    counts=counts,
                    distances=distances,
                    hex_adj_mx=hex_adj_mx,
                    embed_adj_mx=embed_adj_mx,
                    adjacency=args.adjacency,
                    circs=circs,
                    facecolors=facecolors,
                    edgecolors=edgecolors,
                    alphas=alphas,
                    embeddings=embeddings,
                    cluster_frac=args.cluster_frac,
                    section=section,
                    window_size=args.window_size,
                )
                ax.set_title(section)
                print(f"{section}: Close the window")
                all_path_counts.append(path_counts)

        start_idx = -1
        start_set = False

        plt.ion()
        fig, ax = plt.subplots(figsize=(args.figsize, args.figsize))
        circs, facecolors, edgecolors, alphas = show_slide(
            ax=ax,
            im=im,
            pos_df=pos_df,
            spot_radius=spot_radius,
        )
        ax.set_title("Select start spot")
        cid = fig.canvas.mpl_connect("pick_event", onpick)

        plt.show(block=True)

        fig.savefig(os.path.join(args.output_dir, section.replace("/", "-") + ".png"))
        plt.close(fig)
        print(f"{section}: Saved figure")

    breakpoint()


# start_idx = 2352
# end_idx = 727

if __name__ == "__main__":
    args = parse_args()
    main(args)
