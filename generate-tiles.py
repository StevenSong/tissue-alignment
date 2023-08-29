import json
import os

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

# prefix = "/mnt/data5/spatial/data/colon"
# prefix = "/mnt/data5/spatial/data/stomach"
# spot_radius = 86 // 2

prefix = "/mnt/data5/spatial/data/dlpfc"
spot_radius = 96 // 2

cols = [
    "barcode",
    "in_tissue",
    "array_row",
    "array_col",
    "pxl_row_in_fullres",
    "pxl_col_in_fullres",
]

tile_shape = (spot_radius * 2, spot_radius * 2, 3)

count = 0
padded = 0

for cat in tqdm(sorted(os.listdir(prefix)), desc="Category"):
    for slide in tqdm(sorted(os.listdir(os.path.join(prefix, cat))), desc="Slide"):
        bpath = os.path.join(prefix, cat, slide)
        spath = os.path.join(bpath, "outs/spatial")
        tpath = os.path.join(bpath, "tiles")
        os.makedirs(tpath, exist_ok=True)
        pos_df = pd.read_csv(
            os.path.join(spath, "tissue_positions_list.csv"),
            header=None,
            names=cols,
        )
        pos_df = pos_df.loc[pos_df["in_tissue"] == 1]
        with open(os.path.join(spath, "scalefactors_json.json")) as f:
            scale_factors = json.loads(f.read())
        # check that the actual spot size is not smaller than tile size - 1
        # ok if actual spot is larger than tile size
        actual_size = scale_factors["spot_diameter_fullres"]
        assert actual_size - spot_radius * 2 > -1

        im = np.asarray(Image.open(os.path.join(bpath, f"{slide}.tif")))
        for i, spot in tqdm(pos_df.iterrows(), desc="Tile"):
            count += 1
            row = spot["pxl_row_in_fullres"]
            col = spot["pxl_col_in_fullres"]
            row_lo, row_hi = row - spot_radius, row + spot_radius
            col_lo, col_hi = col - spot_radius, col + spot_radius
            tile = im[row_lo:row_hi, col_lo:col_hi]
            if tile.shape != tile_shape:
                new_tile = np.full(tile_shape, 255, dtype=tile.dtype)
                new_tile[: tile.shape[0], : tile.shape[1], : tile.shape[2]] = tile
                tile = new_tile
                padded += 1
            Image.fromarray(tile).save(os.path.join(tpath, f'{spot["barcode"]}.png'))

print(f"Padded {padded} tiles")
print(f"Saved {count} tiles")
