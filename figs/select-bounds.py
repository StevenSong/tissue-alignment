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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    return args


def main(args):
    plt.ion()
    fullres = False
    im = read_image(args.section, fullres)
    pos_df, spot_radius = read_spatial_data(args.section, fullres)

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
    circs.set_facecolor(facecolors)
    circs.set_edgecolor(edgecolors)
    circs.set_alpha(alphas)
    ax.add_collection(circs)
    idxs = defaultdict(list)
    curr = 0
    ax.set_title(f"Boundary #{curr}")

    def onpick(event):
        nonlocal curr
        idx = event.ind[0]
        idxs[curr].append(idx)
        color = list(TABLEAU_COLORS)[curr]
        facecolors[idx] = list(to_rgba(color))
        alphas[idx] = 1
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
    _idxs = []
    for i in range(len(idxs)):
        _idxs.append(np.asarray(sorted(list(set(idxs[i])))))
    idxs = np.asarray(_idxs, dtype=object)
    np.save(args.output, idxs)


if __name__ == "__main__":
    args = parse_args()
    main(args)
