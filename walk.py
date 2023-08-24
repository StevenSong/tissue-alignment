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
        choices=["off", "hex", "embed", "path-clusters", "orthogonal-paths"],
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

    # k means
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=1,
        help="enable kmeans clustering by setting num clusters > 1",
    )
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--fit-all-centroids", action="store_true")

    # orthogonal paths
    parser.add_argument("--max-depth", type=int, default=10)

    # path clusters
    parser.add_argument("--cluster-frac", type=float, default=1)

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
    # distance 0 is treated as unreachable by path alg
    distances_hex = np.where(hex_adj_mx, distances, 0)

    # embedding space adjacency
    embed_adj_mx = get_embedding_adj_matrix(
        distances=distances, num_neighbors=num_neighbors
    )
    distances_embed = np.where(embed_adj_mx, distances, 0)
    return (distances_hex, hex_adj_mx), (distances_embed, embed_adj_mx)


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


def compute_clusters(
    section,
    embeddings,
    num_clusters,
    max_iter,
    tol,
    start_idx,
    end_idx,
    fit_all_centroids,
):
    print(f"{section}: Computing clusters")
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
        clusters = _get_clusters(embeddings=embeddings, centroids=centroids)

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
    return path_idxs  # either list or list of lists


def get_path_counts(
    avg_expression,
    counts,
    path_idxs,
    distances_hex,
    distances_embed,
    genes,
    clusters,
):
    if avg_expression == "hex":
        adj_idxs_fn = lambda i_idx: distances_hex[i_idx[1]].nonzero()[0]
    elif avg_expression == "embed":
        adj_idxs_fn = lambda i_idx: distances_embed[i_idx[1]].nonzero()[0]
    elif avg_expression == "path-clusters":
        adj_idxs_fn = lambda i_idx: np.nonzero(clusters == i_idx[0])[0]
    elif avg_expression == "orthogonal-paths":
        # path_idxs is list of list of ints
        x = np.linspace(0, 1, 100)
        path_counts = {gene: np.zeros(100, dtype=float) for gene in genes}
        for _path_idxs in path_idxs:
            xp = np.linspace(0, 1, len(_path_idxs))
            _path_counts = counts.loc[_path_idxs, genes]
            for gene in genes:
                path_counts[gene] += np.interp(x=x, xp=xp, fp=_path_counts[gene])
        path_counts = pd.DataFrame(path_counts) / len(path_idxs)
        return path_counts
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


def show_slide(ax, im, pos_df, spot_radius, clusters=None):
    ax.imshow(im)
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
    ax.add_collection(circs)
    return circs, facecolors, edgecolors, alphas


def select_start(ax, pos_df, start_idx, circs, edgecolors, alphas):
    edgecolors[start_idx] = to_rgba("tab:red")
    alphas[start_idx] = 1
    ax.annotate(str(0), pos_df.loc[start_idx, ["x", "y"]], color="tab:red")
    circs.set_edgecolor(edgecolors)
    circs.set_alpha(alphas)


def get_orthogonal_spot(embeddings, adj_mx, curr_idx, next_idx):
    adj_idxs = adj_mx[curr_idx].copy()
    adj_idxs[next_idx] = 0
    adj_diffs = embeddings[adj_idxs] - embeddings[curr_idx]
    next_diff = embeddings[next_idx] - embeddings[curr_idx]
    adj_norm = adj_diffs / np.linalg.norm(adj_diffs, axis=1)[:, None]
    next_norm = next_diff / np.linalg.norm(next_diff)
    dot_prods = np.abs(adj_norm.dot(next_norm))
    orthogonal_idx = adj_idxs.nonzero()[0][dot_prods.argmin()]
    return orthogonal_idx


def get_orthogonal_paths(
    section,
    adjacency,
    hex_adj_mx,
    embed_adj_mx,
    path_idxs,
    start_idx,
    end_idx,
    max_depth,
    embeddings,
    path_alg,
    distances_hex,
    distances_embed,
):
    print(f"{section}: Computing orthogonal paths")
    if adjacency == "hex":
        _adj_mx = hex_adj_mx
        _distances = distances_hex
    else:  # adjacency == 'embed'
        _adj_mx = embed_adj_mx
        _distances = distances_embed

    # 1. for start/end spots:
    #    a: get adjacent spot embeddings
    #    b: get norm of vector from spot to adj spots
    #    c: get dot product of each vector to vector of path direction
    #    d: pick most orthogonal vector, define corresponding spot as new start/end spot
    # 2: get path between new start/end spot
    # 3: recurse, stop at depth 10
    all_path_idxs = [path_idxs]
    stack = [
        (
            0,  # recursion depth
            start_idx,
            end_idx,
            path_idxs,
        )
    ]

    def loop_gen(stack):
        while len(stack):
            yield

    for _ in tqdm(loop_gen(stack)):
        _depth, _start_idx, _end_idx, _path_idxs = stack.pop()
        if _depth >= max_depth:
            continue
        _new_depth = _depth + 1
        _new_start_idx = get_orthogonal_spot(
            embeddings=embeddings,
            adj_mx=_adj_mx,
            curr_idx=_start_idx,
            next_idx=_path_idxs[1],
        )
        _new_end_idx = get_orthogonal_spot(
            embeddings=embeddings,
            adj_mx=_adj_mx,
            curr_idx=_end_idx,
            next_idx=_path_idxs[-2],
        )
        # prevent using the same path as the previous path
        # also prevent next iteration from choosing current spot as orthogonal
        for mx in [hex_adj_mx, embed_adj_mx, distances_hex, distances_embed]:
            mx[_start_idx, _new_start_idx] = 0
            mx[_new_start_idx, _start_idx] = 0
            mx[_end_idx, _new_end_idx] = 0
            mx[_new_end_idx, _end_idx] = 0
            for i, j in zip(_path_idxs[:-1], _path_idxs[1:]):
                mx[i, j] = 0
                mx[j, i] = 0
        _new_path_idxs = compute_path_idxs(
            distances=_distances,
            path_alg=path_alg,
            start_idx=_new_start_idx,
            end_idx=_new_end_idx,
        )
        all_path_idxs.append(_new_path_idxs)
        stack.append((_new_depth, _new_start_idx, _new_end_idx, _new_path_idxs))
    return all_path_idxs


def select_end(
    ax,
    pos_df,
    start_idx,
    end_idx,
    genes,
    path_alg,
    avg_expression,
    counts,
    distances_hex,
    hex_adj_mx,
    distances_embed,
    embed_adj_mx,
    adjacency,
    circs,
    facecolors,
    edgecolors,
    alphas,
    embeddings,
    max_depth,
    cluster_frac,
    section,
    clusters=None,
):
    path_idxs = compute_path_idxs(
        distances=(distances_hex if adjacency == "hex" else distances_embed),
        path_alg=path_alg,
        start_idx=start_idx,
        end_idx=end_idx,
    )

    # edgecolor for path
    count = 1
    for path_idx in path_idxs[1:]:  # last index includes end_idx
        edgecolors[path_idx] = to_rgba("tab:red")
        alphas[path_idx] = 1
        ax.annotate(str(count), pos_df.loc[path_idx, ["x", "y"]], color="tab:red")
        count += 1

    # facecolor for clusters
    if clusters is not None:
        facecolors = np.asarray([list(to_rgba(COLORS[i])) for i in clusters])
    elif avg_expression == "path-clusters":
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
    elif avg_expression == "orthogonal-paths":
        all_path_idxs = get_orthogonal_paths(
            section=section,
            adjacency=adjacency,
            hex_adj_mx=hex_adj_mx,
            embed_adj_mx=embed_adj_mx,
            path_idxs=path_idxs,
            start_idx=start_idx,
            end_idx=end_idx,
            max_depth=max_depth,
            embeddings=embeddings,
            path_alg=path_alg,
            distances_hex=distances_hex,
            distances_embed=distances_embed,
        )
        cmap = colormaps["viridis"]
        for _path_idxs in all_path_idxs:
            cmap_interp = np.linspace(0, 1, len(_path_idxs))
            for i, _idx in enumerate(_path_idxs):
                facecolors[_idx] = cmap(cmap_interp[i])
        path_idxs = all_path_idxs

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
        (distances_hex, hex_adj_mx), (
            distances_embed,
            embed_adj_mx,
        ) = compute_distance_matrix(
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
                clusters = None
                if args.num_clusters > 1 and not args.fit_all_centroids:
                    clusters = compute_clusters(
                        section=section,
                        embeddings=embeddings,
                        num_clusters=args.num_clusters,
                        max_iter=args.max_iter,
                        tol=args.tol,
                        start_idx=start_idx,
                        end_idx=end_idx,
                        fit_all_centroids=False,
                    )
                path_counts = select_end(
                    ax=ax,
                    pos_df=pos_df,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    genes=args.genes,
                    path_alg=args.path_alg,
                    avg_expression=args.avg_expression,
                    counts=counts,
                    distances_hex=distances_hex,
                    hex_adj_mx=hex_adj_mx,
                    distances_embed=distances_embed,
                    embed_adj_mx=embed_adj_mx,
                    adjacency=args.adjacency,
                    circs=circs,
                    facecolors=facecolors,
                    edgecolors=edgecolors,
                    alphas=alphas,
                    embeddings=embeddings,
                    max_depth=args.max_depth,
                    clusters=clusters,
                    cluster_frac=args.cluster_frac,
                    section=section,
                )
                ax.set_title("Close this window")
                all_path_counts.append(path_counts)

        start_idx = -1
        start_set = False

        plt.ion()
        fig, ax = plt.subplots(figsize=(args.figsize, args.figsize))
        clusters = None
        if args.num_clusters > 1 and args.fit_all_centroids:
            clusters = compute_clusters(
                section=section,
                embeddings=embeddings,
                num_clusters=args.num_clusters,
                max_iter=args.max_iter,
                tol=args.tol,
                start_idx=-1,
                end_idx=-1,
                fit_all_centroids=True,
            )
        circs, facecolors, edgecolors, alphas = show_slide(
            ax=ax, im=im, pos_df=pos_df, spot_radius=spot_radius, clusters=clusters
        )
        ax.set_title("Select start spot")
        cid = fig.canvas.mpl_connect("pick_event", onpick)

        plt.show(block=True)
        ax.set_title(section)
        fig.savefig(os.path.join(args.output_dir, section.replace("/", "-") + ".png"))
        plt.close(fig)
        print(f"{section}: Saved figure")

    breakpoint()


# start_idx = 2352
# end_idx = 727

if __name__ == "__main__":
    args = parse_args()
    main(args)
