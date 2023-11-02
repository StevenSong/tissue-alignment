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
from glmpca import glmpca
from harmonic import harmonic
from region_cost_fun import fill_geometry
from spatialcoord import spatialcoord
from utils_IO import read_boundary_list, read_input_10xdirectory

for dz in ["CD", "UC"]:
    for section in "ABCD":
        section_path = f"/mnt/data1/spatial/data/colon/{dz}/{section}"
        count, coords, barcodes, gene_labels = read_input_10xdirectory(section_path)
        glmpca_res = glmpca.glmpca(count, 5, fam="poi", penalty=10, verbose=True)
        F_glmpca = glmpca_res["factors"]
        np.save(f"colon/colon-{dz}-{section}-glmpca.npy", F_glmpca)
