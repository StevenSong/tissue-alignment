import argparse
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.colors import TABLEAU_COLORS, to_rgba
from matplotlib.patches import Circle

sys.path.append("/mnt/data1/spatial/tissue-alignment/traversal")
from io_utils import read_image, read_spatial_data
from plot import draw_clusters


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layers")
    args = parser.parse_args()
    return args


def main(args):
    plt.ion()
    fullres = False
    im = read_image(args.section, fullres)
    pos_df, spot_radius = read_spatial_data(args.section, fullres)
    layers = pos_df[["barcode"]].copy()
    layers["layer"] = -1
    if args.layers is not None:
        layers = pd.read_csv(args.layers)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(im)
    circs = PatchCollection(
        [Circle((x, y), spot_radius) for x, y in pos_df[["x", "y"]].to_numpy()],
        picker=True,
    )
    n = len(pos_df)
    facecolors = np.asarray([list(to_rgba("lightgray"))] * n)
    edgecolors = np.asarray([list(to_rgba("darkgray"))] * n)
    alphas = np.full(n, 0.5)

    sample = "151673"
    if sample in args.section:
        layer_to_barcode = {}
        for i, l in enumerate(["L1", "L2", "L3", "L4", "L5", "L6", "WM"]):
            with open(f"{sample}/{sample}_{l}_barcodes.txt") as f:
                layer_to_barcode[i] = f.read().splitlines()
        barcode_to_layer = {b: l for l, bs in layer_to_barcode.items() for b in bs}
        num_layers = len(layer_to_barcode)
        pos_df["layer"] = pos_df["barcode"].replace(barcode_to_layer)
        tab10 = list(TABLEAU_COLORS)
        layer_to_color = dict(zip(layer_to_barcode, tab10))
        facecolors = np.asarray(
            [
                list(to_rgba(layer_to_color[l] if l in layer_to_color else "lightgray"))
                for l in pos_df["layer"]
            ]
        )

    circs.set_facecolor(facecolors)
    circs.set_edgecolor(edgecolors)
    circs.set_alpha(alphas)
    ax.add_collection(circs)
    draw_clusters(
        fig=fig,
        ax=ax,
        path_idxs=[-1] * len(layers["layer"].drop_duplicates()),
        clusters=layers["layer"],
        circs=circs,
        facecolors=facecolors,
        edgecolors=edgecolors,
        cm_name="tab10",
        show_cbar=False,
        discrete_cm=True,
    )
    curr = 0
    ax.set_title(f"Boundary #{curr}")

    def onpick(event):
        nonlocal curr
        idx = event.ind[0]
        color = list(TABLEAU_COLORS)[curr]
        facecolors[idx] = list(to_rgba(color))
        edgecolors[idx] = list(to_rgba(color))
        layers.loc[idx, "layer"] = curr
        circs.set_facecolor(facecolors)
        circs.set_edgecolor(edgecolors)

    cid1 = fig.canvas.mpl_connect("pick_event", onpick)

    def onpress(event):
        nonlocal curr
        if event.key.isnumeric():
            curr = int(event.key)
            ax.set_title(f"Boundary #{curr}")

    cid2 = fig.canvas.mpl_connect("key_press_event", onpress)
    plt.show(block=True)
    layers.to_csv(args.output, index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
