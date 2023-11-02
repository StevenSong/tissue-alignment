import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.stats.api as sms
from matplotlib import colormaps
from matplotlib.collections import PatchCollection
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle
from scipy.stats import spearmanr

sys.path.append("/mnt/data1/spatial/tissue-alignment/traversal")
from io_utils import read_image, read_spatial_data
from plot import draw_clusters

sys.path.append("/mnt/data1/spatial/belayer/src")
from dprelated import dp
from harmonic import harmonic
from region_cost_fun import fill_geometry
from spatialcoord import spatialcoord
from utils_IO import read_boundary_list, read_input_10xdirectory

for dz in ["CD", "UC"]:
    for section in "ABCD":
        section_path = f"/mnt/data1/spatial/data/colon/{dz}/{section}"
        count, coords, barcodes, gene_labels = read_input_10xdirectory(section_path)
        F_glmpca = np.load(f"colon/colon-{dz}-{section}-glmpca.npy")

        boundary_file = f"colon/colon-{dz}-{section}-boundaries.npy"

        G, N = count.shape

        # fill in gaps in 10x hexagonal grid
        fullpoints, in_tissue = fill_geometry(coords, is_hexagon=True)

        # get boundary points
        boundary_list = read_boundary_list(boundary_file, fullpoints)

        spos = spatialcoord(x=fullpoints[:, 0], y=fullpoints[:, 1])
        har = harmonic(
            fullpoints, spos.adjacency_mat, np.sqrt(spos.pairwise_squared_dist)
        )

        interpolation = har.interpolation_using_list(boundary_list)
        depth = interpolation[in_tissue]

        bounds = np.load(boundary_file, allow_pickle=True)
        L = len(bounds) - 1
        loss_array, label_dict = dp(F_glmpca.T, depth, L, use_buckets=False)
        belayer_labels = label_dict[L]

        pos_df, spot_radius = read_spatial_data(section_path, False)
        pos_df.loc[pos_df["barcode"].argsort(), "depth"] = depth
        pos_df["depth2"] = pos_df["depth"].round().astype(int)
        pos_df["depth2"] = pos_df["depth2"] - pos_df["depth2"].min()  # make min 0
        pos_df.loc[pos_df["barcode"].argsort(), "belayer-layer"] = belayer_labels
        pos_df[["barcode", "depth", "depth2", "belayer-layer"]].to_csv(
            f"colon/colon-{dz}-{section}-belayer.csv", index=False
        )
