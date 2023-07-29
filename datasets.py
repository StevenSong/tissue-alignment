import os
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from tqdm import tqdm


class TileDataset(Dataset):
    def __init__(
        self,
        *,
        tile_dir: Union[str, List[str]],
        file_ext: str,
        name: Optional[str] = None,
        transform: Optional[Callable] = None,
    ):
        if not isinstance(tile_dir, list):
            tile_dir = [tile_dir]

        self.tile_paths = []
        for t in tile_dir:
            for root, dirs, files in os.walk(t):
                for file in files:
                    if file.endswith(file_ext):
                        self.tile_paths.append(os.path.join(root, file))
        self.name = name
        self.__mean = None
        self.__std = None
        self.transform = transform

    def __len__(self):
        return len(self.tile_paths)

    def __getitem__(self, idx):
        tile_path = self.tile_paths[idx]
        tile = read_image(tile_path)
        if not torch.is_floating_point(tile):
            tile = tile.to(torch.float32) / 255
        if self.transform is not None:
            tile = self.transform(tile)
        return tile

    def get_tile_size(self):
        return self[0].shape[1]

    def get_mean_std(self, *, recompute: bool = False):
        if recompute or self.__mean is None or self.__std is None:
            means = []
            stds = []
            name = ""
            if self.name is not None and self.name:
                name = f"{self.name.title()} "
            for i in tqdm(range(len(self)), desc=f"Computing {name}Dataset Norm"):
                tile = self[i].to(float)
                means.append(tile.mean(axis=(1, 2)))
                stds.append(tile.std(axis=(1, 2)))
            self.__mean = np.asarray(means).mean(axis=0)
            self.__std = np.asarray(stds).mean(axis=0)
        return self.__mean, self.__std
