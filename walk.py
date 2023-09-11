import argparse
import json
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp_sparse
import torch
from dtw import dtw, warp
from matplotlib import colormaps
from matplotlib.collections import PatchCollection
from matplotlib.colors import to_rgba
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
from matplotlib.ticker import MaxNLocator
from PIL import Image
from scipy.sparse.csgraph import shortest_path
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import PAIRWISE_DISTANCE_FUNCTIONS

epilog = """\
Usage tips:
1) Pick start points in the same tissue layer across slides
2) Pick start points close to the middle of the edge of the tissue layer
3) Beware of tissue layer differences
"""


def parse_args():
    parser = argparse.ArgumentParser(epilog=epilog)
    parser.add_argument("--adjacency", choices=["hex", "embed"], default="hex")
    parser.add_argument(
        "--avg-expression",
        choices=["off", "hex", "embed", "path-clusters"],
        default="path-clusters",
    )
    parser.add_argument("--num-neighbors", type=int, default=6)
    parser.add_argument("--sections", nargs="+", required=True)
    parser.add_argument("--data_root", default="/mnt/data5/spatial/data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="triplet-all-slides-0999")
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
    parser.add_argument("--alignment-genes", nargs="+", default=["EPCAM", "ACTA2"])
    parser.add_argument("--figsize", type=int, default=8)
    parser.add_argument("--spot-frac", type=float, default=1)

    args = parser.parse_args()
    return args


def read_spatial_data(section_path, fullres):
    pos_df = pd.read_csv(
        os.path.join(section_path, "outs/spatial/tissue_positions_list.csv"),
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
    with open(os.path.join(section_path, "outs/spatial/scalefactors_json.json")) as f:
        scale_factors = json.loads(f.read())
    spot_radius = int(round(scale_factors["spot_diameter_fullres"] / 2))
    if not fullres:
        hires_scale = scale_factors["tissue_hires_scalef"]
        spot_radius = round(spot_radius * hires_scale)
        pos_df[["y", "x"]] *= hires_scale
    return pos_df, spot_radius


def read_transcription_data(section_path, genes):
    with h5py.File(
        os.path.join(section_path, "outs/filtered_feature_bc_matrix.h5"), "r"
    ) as f:
        barcodes = f["matrix/barcodes"][:]
        data = f["matrix/data"][:]
        indices = f["matrix/indices"][:]
        indptr = f["matrix/indptr"][:]
        shape = f["matrix/shape"][:]
        names = f["matrix/features/name"][:]
        mat = sp_sparse.csc_matrix((data, indices, indptr), shape=shape)

    target = 10_000  # 10_000 counts per spot (or per slide)
    t = mat.shape[1]  # number of spots in slide
    z = mat.sum()  # sum of all counts in slide
    # t = t if use_slide_size else 1
    mat = mat * t / z * target  # (z/t) is average count per spot for the slide
    mat = mat.log1p()

    df = pd.DataFrame.sparse.from_spmatrix(mat).T
    df.index = pd.Series(barcodes).str.decode("utf-8")
    df.columns = pd.Series(names).str.decode("utf-8")

    # reindexing is really slow, instead of below, index using barcodes
    # (see get_path_counts fn)
    # use pos_df ordering of spots
    # counts = counts.loc[pos_df["barcode"]].reset_index()
    # assert counts["barcode"].equals(pos_df["barcode"])

    for gene in genes:
        assert gene in df.columns

    return df


def read_embedding_data(section_path, model):
    return torch.load(os.path.join(section_path, f"embeddings/{model}.pt")).numpy()


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


def _get_clusters(embeddings, centroids, spot_frac=1):
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
    pos_df,
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
        counts_idxs = pos_df.loc[path_idxs, "barcode"]
        return counts.loc[counts_idxs, genes]

    path_counts = {}
    for path_i, path_idx in enumerate(path_idxs):
        adj_idxs = adj_idxs_fn((path_i, path_idx))
        counts_idxs = pos_df.loc[adj_idxs, "barcode"]
        avg = counts.loc[counts_idxs, genes].mean(axis=0)
        path_counts[path_idx] = avg
    path_counts = pd.DataFrame(path_counts).T
    return path_counts


def read_image(section_path, fullres):
    if fullres:
        slide = os.path.basename(section_path.rstrip("/"))
        im = Image.open(os.path.join(section_path, f"{slide}.tif"))
    else:
        im = Image.open(
            os.path.join(section_path, "outs/spatial/tissue_hires_image.png")
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
    edgecolors[start_idx] = to_rgba("tab:red")
    alphas[start_idx] = 1
    ax.annotate(str(0), pos_df.loc[start_idx, ["x", "y"]], color="tab:red")
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
    spot_frac,
    section,
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

    # edgecolor for path
    count = 1
    for path_idx in path_idxs[1:]:  # last index includes end_idx
        edgecolors[path_idx] = to_rgba("tab:red")
        alphas[path_idx] = 1
        ax.annotate(str(count), pos_df.loc[path_idx, ["x", "y"]], color="tab:red")
        count += 1

    # facecolor for clusters
    if avg_expression == "path-clusters":
        clusters = _get_clusters(
            embeddings=embeddings,
            centroids=embeddings[path_idxs],
            spot_frac=spot_frac,
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
        pos_df=pos_df,
        counts=counts,
        path_idxs=path_idxs,
        hex_adj_mx=hex_adj_mx,
        embed_adj_mx=embed_adj_mx,
        genes=genes,
        clusters=clusters,
    )
    print(f"{section}: Calculated expression along path")
    return path_counts


def main(args):
    print(f"----- Section Selection -----")
    base_section = None

    def pick_section(event):
        for ax, section in zip(axs, args.sections):
            if event.artist == ax:
                nonlocal base_section
                base_section = section
                fig.canvas.mpl_disconnect(cid)
                break

    fig = plt.figure(figsize=(args.figsize, args.figsize))
    n = int(np.ceil(np.sqrt(len(args.sections))))
    gs = GridSpec(n, n, figure=fig)
    axs = []
    for i, section in enumerate(args.sections):
        ax = fig.add_subplot(gs[i // n, i % n])
        ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        section_path = os.path.join(args.data_root, section)
        im = read_image(section_path=section_path, fullres=False)
        ax.imshow(im)
        ax.set_title(section)
        ax.set_picker(True)
        axs.append(ax)
    fig.suptitle("Select section to use as base path")
    cid = fig.canvas.mpl_connect("pick_event", pick_section)
    plt.show(block=False)
    while base_section is None:
        plt.pause(0.5)
    plt.close(fig)
    print(f"Selected {base_section} as base section")

    all_path_counts = []
    for section in args.sections:
        print(f"----- {section} -----")
        section_path = os.path.join(args.data_root, section)
        pos_df, spot_radius = read_spatial_data(
            section_path=section_path, fullres=args.fullres
        )
        print(f"{section}: Loaded spot positions")
        # be really careful how you index counts,
        # its ordering is not the same as pos_df
        counts = read_transcription_data(section_path=section_path, genes=args.genes)
        print(f"{section}: Loaded transcription counts")
        embeddings = read_embedding_data(section_path=section_path, model=args.model)
        print(f"{section}: Loaded embeddings")
        distances, hex_adj_mx, embed_adj_mx = compute_distance_matrix(
            embeddings=embeddings,
            distance_metric=args.distance_metric,
            pos_df=pos_df,
            num_neighbors=args.num_neighbors,
        )
        print(f"{section}: Computed distances")
        im = read_image(section_path=section_path, fullres=args.fullres)
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
                    spot_frac=args.spot_frac,
                    section=section,
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

        fig.savefig(
            os.path.join(
                args.output_dir, "walked-" + section.replace("/", "-") + ".png"
            )
        )
        plt.close(fig)
        print(f"{section}: Saved figure")
    plt.ioff()

    print("----- Path Alignment -----")

    # do path alignment
    assert all(gene in args.genes for gene in args.alignment_genes)
    alignment_gene_idxs = [
        i for i, gene in enumerate(args.genes) if gene in args.alignment_genes
    ]
    temp = [section == base_section for section in args.sections]
    ref_i = np.asarray(temp).argmax()
    ref_x = all_path_counts[ref_i].to_numpy()
    ref = ref_x[:, alignment_gene_idxs]
    aligned_counts = []
    for i, x in enumerate(all_path_counts):
        if i == ref_i:
            aligned_counts.append(ref_x)
            continue
        x = x.to_numpy()
        qry = x[:, alignment_gene_idxs]
        alignment = dtw(qry, ref, open_end=True)
        x_aligned = x[warp(alignment)]
        x_aligned_padded = np.pad(
            x_aligned,
            pad_width=((0, len(ref) - len(x_aligned)), (0, 0)),
            mode="constant",
            constant_values=np.nan,
        )
        aligned_counts.append(x_aligned_padded)
    aligned_counts = np.asarray(aligned_counts)
    print("Aligned paths")

    # plot all aligned path expressions per gene
    x = np.arange(len(ref))
    for g, gene in enumerate(args.genes):
        fig, ax = plt.subplots(figsize=(args.figsize, args.figsize))
        for p in range(len(aligned_counts)):
            exp = aligned_counts[p, :, g]
            exp = exp[~np.isnan(exp)]
            tag = " (aligned)"
            if p == ref_i:
                tag = " (reference)"
            ax.plot(exp, label=args.sections[p] + tag)
        mean_exp = np.nanmean(aligned_counts[:, :, g], axis=0)
        ax.plot(mean_exp, label="average expression")
        ax.set_title(f"{gene} expression along aligned path")
        ax.set_ylabel("LogNorm Expression")
        ax.set_xlabel("Aligned Path Index")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        fig.savefig(os.path.join(args.output_dir, "aligned-" + gene + ".png"))
        plt.close(fig)
        print(f"Saved {gene} expression")

    # plot average expression
    fig, ax = plt.subplots(figsize=(args.figsize, args.figsize))
    lines_other = []
    genes_other = []
    lines_align = []
    genes_align = []
    for g, gene in enumerate(args.genes):
        mean_exp = np.nanmean(aligned_counts[:, :, g], axis=0)
        err_exp = np.nanstd(aligned_counts[:, :, g], axis=0)
        (line,) = ax.plot(x, mean_exp)
        if gene in args.alignment_genes:
            lines_align.append(line)
            genes_align.append(gene)
        else:
            lines_other.append(line)
            genes_other.append(gene)
        ax.fill_between(x=x, y1=mean_exp - err_exp, y2=mean_exp + err_exp, alpha=0.25)
    line_dummy = [ax.plot([0], marker="None", linestyle="None")[0]]
    gene_dummy = [("---------\n" + "Alignment\n" + "Genes\n" + "---------")]

    ax.set_title("Average gene expression along aligned path")
    ax.set_ylabel("LogNorm Expression")
    ax.set_xlabel("Aligned Path Index")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(
        lines_other + line_dummy + lines_align,
        genes_other + gene_dummy + genes_align,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )
    fig.savefig(os.path.join(args.output_dir, "average-aligned-expressions.png"))
    print("Saved average gene expression")
    print("Close window when done")
    plt.show(block=True)


if __name__ == "__main__":
    args = parse_args()
    main(args)
