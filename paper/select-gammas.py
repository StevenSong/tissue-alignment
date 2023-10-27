import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--dz")
parser.add_argument("--section")
args = parser.parse_args()

# data_dir = '/mnt/data1/spatial/data/colon'
data_dir = "/Users/steven/temp"
fpaths = []
# for dz in ['CD', 'UC']:
#     for section in ['A', 'B', 'C', 'D']:
#         if dz == 'UC' and section == 'B':
#             continue
for dz in [args.dz]:
    for section in [args.section]:
        fpath = f"{data_dir}/{dz}/{section}/spatial"
        ipath = f"colon-{dz}-{section}-segmented.png"
        opath = f"colon-{dz}-{section}-boundaries.npy"
        fpaths.append((fpath, ipath, opath))
fpaths = sorted(fpaths, key=lambda x: x[1])

gray = (128, 128, 128)
cw = [  # colors in BGR
    (180, 119, 31),  # tab:blue
    (14, 127, 255),  # tab:orange
    (44, 160, 44),  # tab:green
    (40, 39, 214),  # tab:red
    (189, 103, 148),  # tab:purple
]
title = f"Do not release mouse while drawing a boundary. Max boundaries supported: {len(cw)}"


def draw_spots(im, spots: pd.DataFrame, radius, colors=None, alpha=0.1):
    overlay = im.copy()
    _colors = colors
    if colors is None:
        colors = [gray] * len(spots)
    spots = spots[["x", "y"]].to_numpy()
    for (x, y), color in zip(spots, colors):
        if _colors is not None and color == gray:
            continue
        x, y = round(x), round(y)
        cv2.circle(overlay, (x, y), radius, color, -1)
    im2 = cv2.addWeighted(overlay, alpha, im, 1 - alpha, 0)
    return im2


idx = -1


def gamma_selector(fpath):
    print("Select index 1-5")
    with open(f"{fpath}/scalefactors_json.json") as f:
        scales = json.load(f)
        radius = scales["spot_diameter_fullres"] / 2
        hires_scale = scales["tissue_hires_scalef"]
        radius = round(radius * hires_scale)

    spots = pd.read_csv(
        f"{fpath}/tissue_positions_list.csv",
        header=None,
        names=["barcode", "inside", "row", "col", "y", "x"],
    )
    spots = spots[spots["inside"].astype(bool)].reset_index(drop=True)
    spots[["x", "y"]] = spots[["x", "y"]] * hires_scale

    im = cv2.imread(f"{fpath}/tissue_hires_image.png")
    im2 = draw_spots(im, spots, radius)

    drawing = False
    last_x, last_y = -1, -1
    gammas = []

    def draw(event, x, y, flags, param):
        global idx
        nonlocal drawing, last_x, last_y, gammas
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            last_x, last_y = x, y
            # idx += 1
            # gammas.append([])
            gammas[idx].append((x, y))
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                cv2.line(im2, (last_x, last_y), (x, y), (0, 0, 255), 5)
                last_x, last_y = x, y
                gammas[idx].append((x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.line(im2, (last_x, last_y), (x, y), (0, 0, 255), 5)
            gammas[idx].append((x, y))
        return x, y

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 800, 800)
    cv2.setMouseCallback(title, draw)
    while True:
        cv2.imshow(title, im2)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            global idx
            idx = -1
            break
        elif k == ord("1"):
            idx = 0
            gammas.append([])
            print("Index 1")
        elif k == ord("2"):
            idx = 1
            gammas.append([])
            print("Index 2")
        elif k == ord("3"):
            idx = 2
            gammas.append([])
            print("Index 3")
        elif k == ord("4"):
            idx = 3
            gammas.append([])
            print("Index 4")
        elif k == ord("5"):
            idx = 4
            gammas.append([])
            print("Index 5")
    return gammas, spots, radius


def calculate_boundary_spots(gamma, spots):
    spots = spots[["x", "y"]].to_numpy()
    gamma = np.array(gamma)

    temp = np.repeat(
        spots[:, None, :], gamma.shape[0], 1
    )  # expand spot coords to same dimension as number of coords in drawn boundary
    diffs = temp - gamma  # coord diff of all boundary coords for each spot
    dists = np.linalg.norm(
        diffs, axis=2
    )  # get distance between boundary coords and spots
    gamma_spot_idxs = np.argmin(
        dists, axis=0
    )  # find the spot with the smallest dist _for each boundary coord_
    return np.unique(
        gamma_spot_idxs
    )  # return unique spots, may not be same length as original input gamma


for fpath, ipath, opath in fpaths:
    gammas, spots, radius = gamma_selector(fpath)
    colors = [gray] * len(spots)
    boundaries = []
    for gamma, color in zip(gammas, cw):
        gamma_spot_idxs = calculate_boundary_spots(gamma, spots)
        boundary = np.array(
            [
                (row, col)
                for row, col in spots.loc[gamma_spot_idxs, ["row", "col"]].to_numpy()
            ]
        )
        boundaries.append(boundary)
        for idx in gamma_spot_idxs:
            colors[idx] = color
    im = cv2.imread(f"{fpath}/tissue_hires_image.png")
    im2 = draw_spots(im, spots, radius, colors=colors, alpha=1)
    cv2.imshow(title, im2)
    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    cv2.imwrite(ipath, im2)
    cv2.destroyAllWindows()
    np.save(opath, np.array(boundaries, dtype=object))
    print(f"Processed {fpath}")
