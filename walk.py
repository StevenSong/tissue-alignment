import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
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
        "--avg-expression", choices=["off", "hex", "embed"], default="off"
    )
    parser.add_argument("--num-neighbors", type=int, default=6)
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=1,
        help="enable kmeans clustering by setting num clusters > 1",
    )
    parser.add_argument("--data-root", default="/mnt/data5/spatial")
    parser.add_argument("--section", default="slide3/A1")
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
    parser.add_argument("--genes", nargs="+", default=["EPCAM", "CEACAM7"])
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--fit-all-centroids", action="store_true")
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


def read_embedding_data(data_root, section):
    return torch.load(
        os.path.join(data_root, "embeddings/all-slides-model/embeddings.pt")
    )[section].numpy()


def compute_distance_matrix(embeddings, distance_metric, pos_df, num_neighbors):
    def get_adj_spots(pos_df, spot_row, spot_col):
        adj_idxs = pos_df[
            (
                # spots 1 row away are 1 col away
                ((pos_df["row"] == spot_row - 1) | (pos_df["row"] == spot_row + 1))
                & ((pos_df["col"] == spot_col - 1) | (pos_df["col"] == spot_col + 1))
            )
            | (
                # spots in same row are 2 cols away
                (pos_df["row"] == spot_row)
                & ((pos_df["col"] == spot_col - 2) | (pos_df["col"] == spot_col + 2))
            )
        ].index
        return adj_idxs

    distances = pairwise_distances(embeddings, metric=distance_metric)
    # compute both kinds of adjacency in case expression averaging needs
    distances_hex = np.empty_like(distances)
    distances_embed = np.empty_like(distances)
    for idx, spot in pos_df.iterrows():
        spot_row, spot_col = spot["row"], spot["col"]

        # hex grid adjacency
        hex_adj_idxs = get_adj_spots(
            pos_df=pos_df, spot_row=spot_row, spot_col=spot_col
        )
        # distance 0 is treated as unreachable by path alg
        distances_hex[idx] = np.where(
            pos_df.index.isin(hex_adj_idxs), distances[idx], 0
        )

        # embedding space adjacency
        embed_adj_idxs = np.argsort(distances[idx])[:num_neighbors]
        distances_embed[idx] = np.where(
            pos_df.index.isin(embed_adj_idxs), distances[idx], 0
        )
    return distances_hex, distances_embed


def compute_clusters(
    embeddings, num_clusters, max_iter, tol, start_idx, end_idx, fit_all_centroids
):
    print("Computer clusters")
    if fit_all_centroids:
        skip = 0
        idxs = np.random.choice(len(embeddings), size=num_clusters, replace=False)
    else:
        skip = 2
        idxs = np.random.choice(
            [x for x in range(len(embeddings)) if x not in {start_idx, end_idx}],
            size=num_clusters - skip,
            replace=False,
        )
        idxs = np.concatenate([[start_idx, end_idx], idxs])

    centroids = embeddings[idxs]
    last_centroids = np.full_like(centroids, 9999)
    for _ in tqdm(enumerate(range(max_iter))):
        distances = np.linalg.norm(
            embeddings - np.repeat(centroids[:, None, :], len(embeddings), axis=1),
            axis=-1,
        )  # euclidean distance
        clusters = distances.argmin(axis=0)

        # frobenius norm
        if np.sqrt(np.power(centroids - last_centroids, 2).sum()) < tol:
            break

        last_centroids = centroids.copy()
        for i in range(skip, num_clusters):
            centroids[i] = embeddings[clusters == i].mean(axis=0)
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
    return path_idxs


def get_path_counts(
    avg_expression, counts, path_idxs, distances_hex, distances_embed, genes
):
    if avg_expression == "hex":
        distances = distances_hex
    elif avg_expression == "embed":
        distances = distances_embed
    else:  # avg_expression == 'off'
        return counts.loc[path_idxs, genes]

    path_counts = {}
    for path_idx in path_idxs:
        adj_idxs = distances[path_idx].nonzero()[0]
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


def show_slide(ax1, im, pos_df, spot_radius, clusters=None):
    ax1.imshow(im)
    circs = PatchCollection(
        [Circle((x, y), spot_radius) for x, y in pos_df[["x", "y"]].to_numpy()],
        picker=True,
    )
    n = len(pos_df)
    facecolors = np.asarray([list(to_rgba("lightgray"))] * n)
    edgecolors = np.asarray([list(to_rgba("lightgray"))] * n)
    if clusters is not None:
        facecolors = np.asarray([list(to_rgba(COLORS[i])) for i in clusters])
    alphas = np.full(n, 0.4)
    circs.set_facecolor(facecolors)
    circs.set_edgecolor(edgecolors)
    circs.set_alpha(alphas)
    ax1.add_collection(circs)
    return circs, facecolors, edgecolors, alphas


def select_start(ax1, pos_df, start_idx, circs, edgecolors, alphas):
    edgecolors[start_idx] = to_rgba("tab:red")
    alphas[start_idx] = 1
    ax1.annotate(str(0), pos_df.loc[start_idx, ["x", "y"]], color="tab:red")
    circs.set_edgecolor(edgecolors)
    circs.set_alpha(alphas)


def select_end(
    ax1,
    ax2,
    pos_df,
    start_idx,
    end_idx,
    genes,
    path_alg,
    avg_expression,
    counts,
    distances_hex,
    distances_embed,
    adjacency,
    circs,
    facecolors,
    edgecolors,
    alphas,
    clusters=None,
):
    # facecolor for clusters
    # edgecolor for path
    if clusters is not None:
        facecolors = np.asarray([list(to_rgba(COLORS[i])) for i in clusters])
    path_idxs = compute_path_idxs(
        distances=(distances_hex if adjacency == "hex" else distances_embed),
        path_alg=path_alg,
        start_idx=start_idx,
        end_idx=end_idx,
    )
    count = 1
    for path_idx in path_idxs[1:]:  # last index includes end_idx
        edgecolors[path_idx] = to_rgba("tab:red")
        alphas[path_idx] = 1
        ax1.annotate(str(count), pos_df.loc[path_idx, ["x", "y"]], color="tab:red")
        count += 1

    circs.set_facecolor(facecolors)
    circs.set_edgecolor(edgecolors)
    circs.set_alpha(alphas)

    path_counts = get_path_counts(
        avg_expression=avg_expression,
        counts=counts,
        path_idxs=path_idxs,
        distances_hex=distances_hex,
        distances_embed=distances_embed,
        genes=genes,
    )
    for gene in genes:
        ax2.scatter(x=range(len(path_idxs)), y=list(path_counts[gene]), label=gene)
    ax2.set_title("Gene expression along path")
    ax2.set_ylabel("LogNorm Expression")
    ax2.set_xlabel("Path Index")
    ax2.set_xticks(list(range(len(path_idxs))))
    ax2.legend()


def main(args):
    count_path = os.path.join(args.data_root, "count", args.section, "outs")
    pos_df, spot_radius = read_spatial_data(count_path=count_path, fullres=args.fullres)
    print("Loaded spot positions")
    counts = read_transcription_data(
        count_path=count_path, pos_df=pos_df, genes=args.genes
    )
    print("Loaded transcription counts")
    embeddings = read_embedding_data(data_root=args.data_root, section=args.section)
    print("Loaded embeddings")
    distances_hex, distances_embed = compute_distance_matrix(
        embeddings=embeddings,
        distance_metric=args.distance_metric,
        pos_df=pos_df,
        num_neighbors=args.num_neighbors,
    )
    print("Computed distances")
    im = read_image(
        data_root=args.data_root, section=args.section, fullres=args.fullres
    )
    print("Loaded image")

    def onpick(event):
        nonlocal start_set, start_idx
        idx = event.ind[0]
        if not start_set:
            start_idx = idx
            start_set = True
            select_start(
                ax1=ax1,
                pos_df=pos_df,
                start_idx=start_idx,
                circs=circs,
                edgecolors=edgecolors,
                alphas=alphas,
            )
        else:
            end_idx = idx
            fig.canvas.mpl_disconnect(cid)
            clusters = None
            if args.num_clusters > 1 and not args.fit_all_centroids:
                clusters = compute_clusters(
                    embeddings=embeddings,
                    num_clusters=args.num_clusters,
                    max_iter=args.max_iter,
                    tol=args.tol,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    fit_all_centroids=False,
                )
            select_end(
                ax1=ax1,
                ax2=ax2,
                pos_df=pos_df,
                start_idx=start_idx,
                end_idx=end_idx,
                genes=args.genes,
                path_alg=args.path_alg,
                avg_expression=args.avg_expression,
                counts=counts,
                distances_hex=distances_hex,
                distances_embed=distances_embed,
                adjacency=args.adjacency,
                circs=circs,
                facecolors=facecolors,
                edgecolors=edgecolors,
                alphas=alphas,
                clusters=clusters,
            )

    start_idx = -1
    start_set = False

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    clusters = None
    if args.num_clusters > 1 and args.fit_all_centroids:
        clusters = compute_clusters(
            embeddings=embeddings,
            num_clusters=args.num_clusters,
            max_iter=args.max_iter,
            tol=args.tol,
            start_idx=-1,
            end_idx=-1,
            fit_all_centroids=True,
        )
    circs, facecolors, edgecolors, alphas = show_slide(
        ax1=ax1, im=im, pos_df=pos_df, spot_radius=spot_radius, clusters=clusters
    )

    cid = fig.canvas.mpl_connect("pick_event", onpick)

    plt.show(block=True)


# start_idx = 2352
# end_idx = 727
# select_start(pos_df=pos_df, start_idx=start_idx)
# select_end(
#     pos_df=pos_df,
#     start_idx=start_idx,
#     end_idx=end_idx,
#     genes=args.genes,
#     path_alg=args.path_alg,
#     avg_expression=args.avg_expression,
#     counts=counts,
#     distances_hex=distances_hex,
#     distances_embed=distances_embed,
#     adjacency=args.adjacency,
# )

if __name__ == "__main__":
    args = parse_args()
    main(args)
