import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


Image.MAX_IMAGE_PIXELS = None

prefix = '/mnt/data5/spatial'

cols = [
    'barcode',
    'in_tissue',
    'array_row',
    'array_col',
    'pxl_row_in_fullres',
    'pxl_col_in_fullres',
]

spot_radius = 86 // 2

count = 0
discarded = 0
for slide in tqdm([1, 2, 3, 4], desc='Slide'):
    for section in tqdm(['A', 'B', 'C', 'D'], desc='Section'):
        spath = f'slide{slide}/{section}1'
        ipath = os.path.join(prefix, 'data', spath, f'{section}1.tif')
        cpath = os.path.join(prefix, 'count', spath, 'outs/spatial')
        opath = os.path.join(prefix, 'tiles', spath)
        os.makedirs(opath, exist_ok=True)
        pos_df = pd.read_csv(
            os.path.join(cpath, 'tissue_positions_list.csv'),
            header=None,
            names=cols,
        )
        pos_df = pos_df.loc[pos_df['in_tissue'] == 1]
        with open(os.path.join(cpath, 'scalefactors_json.json')) as f:
            scale_factors = json.loads(f.read())
        # check that the actual spot size is not smaller than tile size - 1
        # ok if actual spot is larger than tile size
        actual_size = scale_factors['spot_diameter_fullres']
        assert actual_size - spot_radius * 2 > -1

        im = np.asarray(Image.open(ipath))
        for i, spot in tqdm(pos_df.iterrows(), desc='Tile'):
            count += 1
            row = spot['pxl_row_in_fullres']
            col = spot['pxl_col_in_fullres']
            row_lo, row_hi = row - spot_radius, row + spot_radius
            col_lo, col_hi = col - spot_radius, col + spot_radius
            tile = im[row_lo:row_hi,col_lo:col_hi]
            if tile.shape != (spot_radius*2, spot_radius*2, 3):
                discarded += 1
                continue
            Image.fromarray(tile).save(os.path.join(opath, f'{spot["barcode"]}.png'))

print(f'Discarded {discarded} tiles')
print(f'Saved {count-discarded} tiles')
