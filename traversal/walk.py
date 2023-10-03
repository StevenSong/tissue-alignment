import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

# isort: off
from graph import (
    compute_clusters,
    compute_distance_matrix,
    compute_path_counts,
    compute_path_idxs,
)
from io_utils import (
    read_embedding_data,
    read_image,
    read_spatial_data,
    read_transcription_data,
)
from plot import LABEL_DIRS, draw_clusters, draw_path, draw_start, pick_idx, show_slide

# isort: on


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--section", required=True)
    parser.add_argument("--data_root", default="/mnt/data1/spatial/data")
    parser.add_argument("--start_idx", required=False, type=int)
    parser.add_argument("--end_idx", required=False, type=int)
    parser.add_argument("--start_label", choices=LABEL_DIRS, default=LABEL_DIRS[0])
    parser.add_argument("--end_label", choices=LABEL_DIRS, default=LABEL_DIRS[1])
    parser.add_argument("--fig_labels", nargs=2, default=["(a)", "(b)"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--fullres", action="store_true")
    parser.add_argument("--figsize", type=int, default=8)

    args = parser.parse_args()
    return args


def main(args):
    section = args.section.replace("/", "-")
    section_path = os.path.join(args.data_root, args.section)

    # ----- Data Loading -----
    pos_df, spot_radius = read_spatial_data(
        section_path=section_path, fullres=args.fullres
    )
    print(f"Loaded spot positions")
    # be really careful how you index counts,
    # its ordering is not the same as pos_df
    counts = read_transcription_data(section_path=section_path)
    print(f"Loaded transcription counts")
    embeddings = read_embedding_data(section_path=section_path, model=args.model)
    print(f"Loaded embeddings")
    im = read_image(section_path=section_path, fullres=args.fullres)
    print(f"Loaded image")

    # ----- Start/End Selection -----
    fig, ax, circs, facecolors, edgecolors = show_slide(
        figsize=args.figsize,
        im=im,
        pos_df=pos_df,
        spot_radius=spot_radius,
        fig_labels=args.fig_labels,
    )

    start_idx = args.start_idx
    if start_idx is None:
        ax.set_title("Select start spot")
        start_idx = pick_idx(fig)
    print(f"Selected start spot: {start_idx}")
    draw_start(
        ax=ax,
        pos_df=pos_df,
        start_idx=start_idx,
        start_label=args.start_label,
        circs=circs,
        edgecolors=edgecolors,
    )

    end_idx = args.end_idx
    if end_idx is None:
        ax.set_title("Select end spot")
        end_idx = pick_idx(fig)
    print(f"Selected end spot: {end_idx}")

    # ----- Path Traversal -----
    distances, hex_adj_mx = compute_distance_matrix(
        embeddings=embeddings,
        pos_df=pos_df,
    )
    print(f"Computed distances")
    path_idxs = compute_path_idxs(
        distances=distances,
        hex_adj_mx=hex_adj_mx,
        start_idx=start_idx,
        end_idx=end_idx,
    )
    print(f"Computed path")
    ax.set_title("")
    draw_path(
        ax=ax,
        pos_df=pos_df,
        path_idxs=path_idxs,
        end_label=args.end_label,
        circs=circs,
        edgecolors=edgecolors,
    )

    # ----- Spot Alignment -----
    clusters = compute_clusters(
        embeddings=embeddings,
        centroids=embeddings[path_idxs],
    )
    print(f"Computed clusters")
    draw_clusters(
        fig=fig,
        ax=ax,
        path_idxs=path_idxs,
        clusters=clusters,
        circs=circs,
        facecolors=facecolors,
    )

    mean_counts, std_counts = compute_path_counts(
        pos_df=pos_df,
        counts=counts,
        path_idxs=path_idxs,
        clusters=clusters,
    )
    print(f"Computed expression along path")

    # ----- Data Saving -----
    pd.DataFrame(
        {
            "cluster": clusters,
            "center_idx": [path_idxs[i] for i in clusters],
        }
    ).to_csv(
        os.path.join(args.output_dir, "clusters-" + section + ".csv"),
        index=False,
    )
    mean_counts.to_csv(
        os.path.join(args.output_dir, "path-mean-counts-" + section + ".csv"),
        index=False,
    )
    std_counts.to_csv(
        os.path.join(args.output_dir, "path-std-counts-" + section + ".csv"),
        index=False,
    )
    fig.savefig(
        os.path.join(args.output_dir, "walked-" + section.replace("/", "-") + ".png")
    )

    # ----- Wait for Close -----
    ax.set_title("Close window when done")
    print("Close window when done")
    plt.show(block=True)


if __name__ == "__main__":
    args = parse_args()
    main(args)
