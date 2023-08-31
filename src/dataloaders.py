import os
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from tqdm import tqdm

from utils import PARAMS

FILE_EXT = ".png"
DATALOADER_T = Iterable[
    Dict[
        str,  # data key
        torch.Tensor,  # data
    ]
]
GET_DATALOADER_FN = Callable[
    [
        List[str],  # data paths
        List[str],  # eval data paths
        PARAMS,  # loader params
        PARAMS,  # eval loader params
        int,  # batch size
        int,  # num workers
    ],
    Tuple[
        DATALOADER_T,
        Optional[DATALOADER_T],
    ],
]
TRANSFORM_FN = Callable[
    [
        Dataset,  # dataset
        int,  # data index
        torch.Tensor,  # data
    ],
    Dict[
        str,  # data key
        torch.Tensor,  # data
    ],
]


class TileDataset(Dataset):
    def __init__(
        self,
        *,  # enforce kwargs
        name: str,
        tile_dirs: List[str],
        transform: Optional[TRANSFORM_FN] = None,
    ):
        self.tile_paths = []
        for t in tile_dirs:
            for root, dirs, files in os.walk(t):
                for file in files:
                    if file.endswith(FILE_EXT):
                        self.tile_paths.append(os.path.join(root, file))
        self.name = name
        self.__mean = None
        self.__std = None
        self.transform = transform

    def __len__(self):
        return len(self.tile_paths)

    def __getitem__(self, idx):
        tile = self.get_raw_tile(idx)
        if self.transform is not None:
            tile = self.transform(
                ds=self,
                idx=idx,
                x=tile,
            )
        return tile

    def get_raw_tile(self, idx):
        tile_path = self.tile_paths[idx]
        tile = read_image(tile_path)
        if not torch.is_floating_point(tile):
            tile = tile.to(torch.float32) / 255
        return tile

    def get_tile_size(self) -> int:
        return self[0].shape[1]

    def get_mean_std(
        self,
        *,  # enforce kwargs
        recompute: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if recompute or self.__mean is None or self.__std is None:
            means = []
            stds = []
            for i in tqdm(
                range(len(self)), desc=f"Computing {self.name.title()} Dataset Norm"
            ):
                tile = self.get_raw_tile(i).to(float)
                means.append(tile.mean(axis=(1, 2)))
                stds.append(tile.std(axis=(1, 2)))
            self.__mean = np.asarray(means).mean(axis=0)
            self.__std = np.asarray(stds).mean(axis=0)
        return self.__mean, self.__std


def __get_simsiam_transform(
    *,  # enforce kwargs
    size: Union[int, Tuple[int, int]],
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
) -> TRANSFORM_FN:
    augmentation = T.Compose(
        [
            T.RandomResizedCrop(size, scale=(0.2, 1.0), antialias=True),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply(
                [T.GaussianBlur(kernel_size=size // 20 * 2 + 1, sigma=(0.1, 2.0))],
                p=0.5,
            ),
            T.RandomHorizontalFlip(),
            T.Normalize(mean=mean, std=std),
        ]
    )

    def transform(
        *,  # enforce kwargs
        ds: Dataset,
        idx: int,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return {
            "x1": augmentation(x),
            "x2": augmentation(x),
        }

    return transform


def get_pathology_tile_simsiam_loaders(
    *,  # enforce kwargs
    data_paths: List[str],
    eval_data_paths: List[str],
    loader_params: PARAMS,
    eval_loader_params: PARAMS,
    batch_size: int,
    num_workers: int,
) -> Tuple[DATALOADER_T, Optional[DATALOADER_T]]:
    train_ds = TileDataset(
        name="train",
        tile_dirs=data_paths,
    )

    mean, std = train_ds.get_mean_std()
    tile_size = train_ds.get_tile_size()
    transform = __get_simsiam_transform(size=tile_size, mean=mean, std=std)
    train_ds.transform = transform

    train_dl = DataLoader(
        train_ds,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    eval_dl = None
    if eval_data_paths:
        eval_ds = TileDataset(
            name="eval",
            tile_dirs=eval_data_paths,
            transform=transform,
        )
        eval_dl = DataLoader(
            eval_ds,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_dl, eval_dl


def __hex_grid_adjacency_matrix(
    *,  # enforce kwargs
    pos_paths: List[str],
    data_paths: List[str],
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    cols = [
        "barcode",
        "in_tissue",
        "array_row",
        "array_col",
        "pxl_row_in_fullres",
        "pxl_col_in_fullres",
    ]
    dtypes = [
        str,
        bool,
        np.uint8,
        np.uint8,
        int,
        int,
    ]
    pos_dfs = []
    for i, ppath in enumerate(pos_paths):
        pos_df = pd.read_csv(
            ppath, header=None, names=cols, dtype=dict(zip(cols, dtypes))
        )
        pos_df["idx"] = i
        pos_df["tile_path"] = data_paths[i] + "/" + pos_df["barcode"] + FILE_EXT
        pos_df = pos_df[pos_df["in_tissue"]]
        pos_dfs.append(pos_df)
    pos_df = pd.concat(pos_dfs, ignore_index=True)
    n = len(pos_df)
    rows = np.asarray(pos_df[["array_row"]])
    cols = np.asarray(pos_df[["array_col"]])
    idxs = np.asarray(pos_df[["idx"]])
    rows_mx = np.broadcast_to(rows, (n, n))
    cols_mx = np.broadcast_to(cols, (n, n))
    idxs_mx = np.broadcast_to(idxs, (n, n))
    slide_spots = idxs_mx == idxs_mx.T
    adj_spots = (
        # spots 1 row away are 1 col away
        ((rows_mx == (rows_mx - 1).T) | (rows_mx == (rows_mx + 1).T))
        & ((cols_mx == (cols_mx - 1).T) | (cols_mx == (cols_mx + 1).T))
    ) | (
        # spots in same row are 2 cols away
        (rows_mx == rows_mx.T)
        & ((cols_mx == (cols_mx - 2).T) | (cols_mx == (cols_mx + 2).T))
    )
    # adj spots must come from same slide/section
    adj_spots &= slide_spots
    assert adj_spots.sum(axis=0).max() <= 6

    # non-adj spots must also come from same slide
    non_adj_spots = ~adj_spots & slide_spots

    # non-adj spots from any slide in corpus
    # non_adj_spots = ~adj_spots

    return adj_spots, non_adj_spots, pos_df


def __get_adj_tile_triplet_transform(
    *,  # enforce kwargs
    size: int,
    adj_spots: np.ndarray,
    non_adj_spots: np.ndarray,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    augment: bool,
) -> TRANSFORM_FN:
    if augment:
        xform = T.Compose(
            [
                T.RandomResizedCrop(size, scale=(0.2, 1.0)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.4)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.5),
                T.RandomSolarize(0.5, p=0.2),
                T.Normalize(mean=mean, std=std),
            ]
        )
    else:
        xform = T.Normalize(mean=mean, std=std)

    def transform(
        *,  # enforce kwargs
        ds: Dataset,
        idx: int,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        assert isinstance(ds, TileDataset)
        pos_idxs = adj_spots[idx].nonzero()[0]
        if len(pos_idxs) == 0:
            # if there are no adjacent spots, just use self as positive
            # not ideal but there's very few of these in the dataset (~0.1%)
            pos_idxs = [idx]
        pos_idx = np.random.choice(pos_idxs, 1)[0]
        neg_idxs = non_adj_spots[idx].nonzero()[0]
        neg_idx = np.random.choice(neg_idxs, 1)[0]
        pos = ds.get_raw_tile(pos_idx)
        neg = ds.get_raw_tile(neg_idx)
        return {
            "x": xform(x),
            "pos": xform(pos),
            "neg": xform(neg),
        }

    return transform


def get_pathology_adj_tile_triplet_loaders(
    *,  # enforce kwargs
    data_paths: List[str],
    eval_data_paths: List[str],
    loader_params: PARAMS,
    eval_loader_params: PARAMS,
    batch_size: int,
    num_workers: int,
) -> Tuple[DATALOADER_T, Optional[DATALOADER_T]]:
    assert "position_table" in loader_params
    if not isinstance(loader_params["position_table"], list):
        loader_params["position_table"] = [loader_params["position_table"]]
    assert len(loader_params["position_table"]) == len(data_paths)
    print("Computing Hex Grid Adjacency Matrix")
    adj_spots, non_adj_spots, pos_df = __hex_grid_adjacency_matrix(
        pos_paths=loader_params["position_table"],
        data_paths=data_paths,
    )
    train_ds = TileDataset(
        name="train",
        tile_dirs=data_paths,
    )
    assert len(train_ds.tile_paths) == len(pos_df)
    train_ds.tile_paths = pos_df["tile_path"].to_list()
    size = train_ds.get_tile_size()
    mean, std = train_ds.get_mean_std()
    augment = bool(loader_params["augment"]) if "augment" in loader_params else False
    transform = __get_adj_tile_triplet_transform(
        size=size,
        adj_spots=adj_spots,
        non_adj_spots=non_adj_spots,
        mean=mean,
        std=std,
        augment=augment,
    )
    train_ds.transform = transform
    train_dl = DataLoader(
        train_ds,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    eval_dl = None
    if eval_data_paths:
        assert "position_table" in eval_loader_params
        assert len(eval_loader_params["position_table"]) == len(eval_data_paths)
        eval_adj_spots, eval_non_adj_spots, eval_pos_df = __hex_grid_adjacency_matrix(
            pos_paths=eval_loader_params["position_table"],
            data_paths=eval_data_paths,
        )
        eval_transform = __get_adj_tile_triplet_transform(
            size=size,
            adj_spots=eval_adj_spots,
            non_adj_spots=eval_non_adj_spots,
            mean=mean,
            std=std,
            augment=False,
        )
        eval_ds = TileDataset(
            name="eval",
            tile_dirs=eval_data_paths,
            transform=eval_transform,
        )
        assert len(eval_ds.tile_paths) == len(eval_pos_df)
        eval_ds.tile_paths = eval_pos_df["tile_path"].to_list()
        eval_dl = DataLoader(
            eval_ds,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_dl, eval_dl


DATALOADERS: Dict[str, GET_DATALOADER_FN] = {
    "pathology/tile-simsiam": get_pathology_tile_simsiam_loaders,
    "pathology/adj-tile-triplet": get_pathology_adj_tile_triplet_loaders,
}
