from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader as PytorchDataLoader

from torchvision import transforms as T

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
        npz_folder,
        image_size,
        offset=5,
        repeat_depth_to_3ch=True,
    ):
        super().__init__()
        
        self.folder = folder
        self.depth_folder = depth_folder
        self.npz_folder = npz_folder

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

    def _load_z_rgb_from_npz(self, folder, frame_name):
        npz_path = os.path.join(self.npz_folder, f"{folder}.npz")

        if not os.path.isfile(npz_path):
            raise RuntimeError(f"Cannot find npz file: {npz_path}")

        data = np.load(npz_path, allow_pickle=True)

        # your saved format:
        # frame_files, z_rgb_indices
        if "frame_files" in data and "z_rgb_indices" in data:
            frame_files = data["frame_files"].tolist()
            z_rgb_indices = data["z_rgb_indices"]

            for i, f in enumerate(frame_files):
                if str(f) == frame_name:
                    if i >= len(z_rgb_indices):
                        raise RuntimeError(f"z_rgb index out of range for frame {frame_name} in {npz_path}")
                    return torch.tensor(z_rgb_indices[i], dtype=torch.long)

            raise RuntimeError(f"Frame {frame_name} not found in {npz_path}")

        # old supported format 1
        elif "data" in data:
            frame_data = data["data"].tolist()

            for item in frame_data:
                if item["frame"] == frame_name:
                    if "z_rgb" not in item:
                        raise RuntimeError(f"z_rgb not found for frame {frame_name} in {npz_path}")
                    return torch.tensor(item["z_rgb"], dtype=torch.long)

            raise RuntimeError(f"Frame {frame_name} not found in {npz_path}")

        # old supported format 2
        elif "frames" in data and "z_rgb" in data:
            frames = data["frames"].tolist()
            z_rgb_all = data["z_rgb"].tolist()

            for i, f in enumerate(frames):
                if str(f) == frame_name:
                    return torch.tensor(z_rgb_all[i], dtype=torch.long)

            raise RuntimeError(f"Frame {frame_name} not found in {npz_path}")

        else:
            raise RuntimeError(f"NPZ format not supported: {npz_path}, keys={list(data.keys())}")

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
        max_retry = 20

        for trial in range(max_retry):
            try:
                cur_index = (index + trial) % self.__len__()
                folder = self.folder_list[cur_index]

                rgb_path = os.path.join(self.folder, folder)
                depth_path = os.path.join(self.depth_folder, folder)
                npz_path = os.path.join(self.npz_folder, f"{folder}.npz")

                # skip if 1 of the 3 main sources is missing
                if not os.path.isdir(rgb_path):
                    print(f"skip {folder}: missing rgb folder")
                    continue

                if not os.path.isdir(depth_path):
                    print(f"skip {folder}: missing depth folder")
                    continue

                if not os.path.isfile(npz_path):
# The line `# print(f"skip {folder}: missing npz file")` is a commented-out print statement in the
# code. This line is meant to be used for debugging purposes to indicate that the dataset is skipping
# a particular folder because the corresponding NPZ file is missing.
                    # print(f"skip {folder}: missing npz file")
                    continue

                rgb_list = sorted(os.listdir(rgb_path), key=lambda x: int(x.split('.')[0][4:]))
                depth_list = sorted(os.listdir(depth_path), key=lambda x: int(x.split('.')[0][4:]))

                num_frames = min(len(rgb_list), len(depth_list))

                if num_frames == 0:
                    print(f"skip {folder}: no frames found")
                    continue

                first_idx = random.randint(0, num_frames - 1)
                second_idx = min(first_idx + self.offset, num_frames - 1)

                rgb1_path = os.path.join(rgb_path, rgb_list[first_idx])
                rgb2_path = os.path.join(rgb_path, rgb_list[second_idx])
                depth1_path = os.path.join(depth_path, depth_list[first_idx])
                depth2_path = os.path.join(depth_path, depth_list[second_idx])

                # skip if one selected file is missing
                if not os.path.isfile(rgb1_path):
                    print(f"skip {folder}: missing rgb1 file")
                    continue

                if not os.path.isfile(rgb2_path):
                    print(f"skip {folder}: missing rgb2 file")
                    continue

                if not os.path.isfile(depth1_path):
                    print(f"skip {folder}: missing depth1 file")
                    continue

                if not os.path.isfile(depth2_path):
                    print(f"skip {folder}: missing depth2 file")
                    continue

                # skip if z_rgb is missing/invalid for rgb1
                try:
                    z_rgb = self._load_z_rgb_from_npz(folder, rgb_list[first_idx])
                except Exception as e:
                    print(f"skip {folder}: missing invalid z_rgb for frame {rgb_list[first_idx]} - {e}")
                    continue

                # RGB
                img1 = Image.open(rgb1_path)
                img2 = Image.open(rgb2_path)

                rgb1 = self.rgb_transform(img1).unsqueeze(1)
                rgb2 = self.rgb_transform(img2).unsqueeze(1)
                cat_img = torch.cat([rgb1, rgb2], dim=1)

                # DEPTH
                depth1 = self._load_depth(depth1_path).unsqueeze(1)
                depth2 = self._load_depth(depth2_path).unsqueeze(1)
                cat_depth = torch.cat([depth1, depth2], dim=1)
                # print("cat_img:", cat_img.shape, "cat_depth:", cat_depth.shape, "z_rgb:", z_rgb.shape)
                return cat_img, cat_depth, z_rgb

            except Exception as e:
                print("error", cur_index, e)

        raise RuntimeError(f"Failed to load sample after {max_retry} retries, starting from index {index}")