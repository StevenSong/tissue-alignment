import json
import os

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp_sparse
import torch
from PIL import Image


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


def read_transcription_data(section_path):
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

    df = pd.DataFrame.sparse.from_spmatrix(mat).T
    df.index = pd.Series(barcodes).str.decode("utf-8")
    df.columns = pd.Series(names).str.decode("utf-8")

    # don't bother reindexing, just use barcodes to index between count and pos

    # only keep genes where at least 15% of spots have nonzero UMI
    # nonzero_UMI_frac = (df != 0).sum(axis=0) / len(df)
    # df = df.loc[:, nonzero_UMI_frac >= 0.15]

    target = 10_000  # 10_000 counts per spot (or per slide)
    t = df.shape[0]  # number of spots in slide
    z = df.to_numpy().sum()  # sum of all counts in slide
    df = df * t / z * target  # (z/t) is average count per spot for the slide
    df = np.log1p(df)

    return df


def read_embedding_data(section_path, model):
    return torch.load(os.path.join(section_path, f"embeddings/{model}.pt")).numpy()


def read_image(section_path, fullres):
    if fullres:
        slide = os.path.basename(section_path.rstrip("/"))
        im = Image.open(os.path.join(section_path, f"{slide}.tif"))
    else:
        im = Image.open(
            os.path.join(section_path, "outs/spatial/tissue_hires_image.png")
        )
    return im
