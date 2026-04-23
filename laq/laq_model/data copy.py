from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader as PytorchDataLoader

from torchvision import transforms as T

import os
import random
import cv2

import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import json



def exists(val):
    return val is not None

def identity(t, *args, **kwargs):
    return t

def pair(val):
    return val if isinstance(val, tuple) else (val, val)

'''
This is the dataset class for Sthv2 dataset.
The dataset is a list of folders, each folder contains a sequence of frames.
You have to change the dataset class to fit your dataset for custom training.
'''

class ImageVideoDataset(Dataset):
    def __init__(
        self,
        folder,
        depth_folder,
        image_size,
        offset=5,
        repeat_depth_to_3ch=True,
    ):
        super().__init__()
        
        self.folder = folder
        self.depth_folder = depth_folder

        rgb_folders = set(os.listdir(folder))
        depth_folders = set(os.listdir(depth_folder))
        self.folder_list = sorted(list(rgb_folders & depth_folders))

        self.image_size = image_size
        self.offset = offset
        self.repeat_depth_to_3ch = repeat_depth_to_3ch

        self.rgb_transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize(image_size),
            T.ToTensor(),
        ])

        self.resize_depth = T.Resize(
            image_size,
            interpolation=T.InterpolationMode.NEAREST
        )

    def __len__(self):
        return len(self.folder_list)

    def _load_depth(self, path):
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if depth is None:
            raise RuntimeError(f"Cannot read depth image: {path}")

        if depth.ndim != 2:
            raise RuntimeError(f"Depth image not single-channel: {path}")

        depth = depth.astype(np.float32) / 65535.0
        depth = torch.from_numpy(depth).unsqueeze(0)  # [1, H, W]

        depth = self.resize_depth(depth)

        if self.repeat_depth_to_3ch:
            depth = depth.repeat(3, 1, 1)

        return depth

    def __getitem__(self, index):
        try:
            folder = self.folder_list[index]

            rgb_path = os.path.join(self.folder, folder)
            depth_path = os.path.join(self.depth_folder, folder)

            rgb_list = sorted(os.listdir(rgb_path), key=lambda x: int(x.split('.')[0][4:]))
            depth_list = sorted(os.listdir(depth_path), key=lambda x: int(x.split('.')[0][4:]))

            num_frames = min(len(rgb_list), len(depth_list))

            first_idx = random.randint(0, num_frames - 1)
            second_idx = min(first_idx + self.offset, num_frames - 1)

            # RGB
            img1 = Image.open(os.path.join(rgb_path, rgb_list[first_idx]))
            img2 = Image.open(os.path.join(rgb_path, rgb_list[second_idx]))

            rgb1 = self.rgb_transform(img1).unsqueeze(1)
            rgb2 = self.rgb_transform(img2).unsqueeze(1)
            cat_img = torch.cat([rgb1, rgb2], dim=1)

            # DEPTH (your correct pipeline)
            depth1 = self._load_depth(os.path.join(depth_path, depth_list[first_idx])).unsqueeze(1)
            depth2 = self._load_depth(os.path.join(depth_path, depth_list[second_idx])).unsqueeze(1)
            cat_depth = torch.cat([depth1, depth2], dim=1)

            return cat_img, cat_depth

        except Exception as e:
            print("error", index, e)
            if index < self.__len__() - 1:
                return self.__getitem__(index + 1)
            else:
                return self.__getitem__(random.randint(0, self.__len__() - 1))