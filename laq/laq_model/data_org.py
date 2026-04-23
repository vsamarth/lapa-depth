from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader as PytorchDataLoader

from torchvision import transforms as T

import os
import random


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

class ImageVideoDataset_(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        offset=5,
    ):
        super().__init__()
        
        self.folder = folder
        self.folder_list = os.listdir(folder)
        self.image_size = image_size
      
        self.offset = offset

        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize(image_size),
            T.ToTensor(),
        ])


    def __len__(self):
        return len(self.folder_list) ## length of folder list is not exact number of frames; TODO: change this to actual number of frames
    
    def __getitem__(self, index):
        try :
            offset = self.offset
            
            folder = self.folder_list[index]
            img_list = os.listdir(os.path.join(self.folder, folder))

            img_list = sorted(img_list, key=lambda x: int(x.split('.')[0][4:]))
            ## pick random frame 
            first_frame_idx = random.randint(0, len(img_list)-1)
            first_frame_idx = min(first_frame_idx, len(img_list)-1)
            second_frame_idx = min(first_frame_idx + offset, len(img_list)-1)
            
            first_path = os.path.join(self.folder, folder, img_list[first_frame_idx])
            second_path = os.path.join(self.folder, folder, img_list[second_frame_idx])
                    
            img = Image.open(first_path)
            next_img = Image.open(second_path)
            
            transform_img = self.transform(img).unsqueeze(1)
            next_transform_img = self.transform(next_img).unsqueeze(1)
            
            cat_img = torch.cat([transform_img, next_transform_img], dim=1)
            return cat_img
        except :
            print("error", index)
            if index < self.__len__() - 1:
                return self.__getitem__(index + 1)
            else:
                return self.__getitem__(random.randint(0, self.__len__() - 1))


import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class ImageVideoDatasetDepth(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        offset=5,
        repeat_to_3ch=True,
    ):
        super().__init__()
        
        self.folder = folder
        self.folder_list = sorted(
            [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]
        )
        self.image_size = image_size
        self.offset = offset
        self.repeat_to_3ch = repeat_to_3ch

        self.resize = T.Resize(
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
            raise RuntimeError(f"Depth image is not single-channel: {path}, shape={depth.shape}")

        # expect uint16 png
        depth = depth.astype(np.float32) / 65535.0

        # H x W -> 1 x H x W
        depth = torch.from_numpy(depth).unsqueeze(0)

        # resize
        depth = self.resize(depth)

        # keep current model happy if it expects 3 channels
        if self.repeat_to_3ch:
            depth = depth.repeat(3, 1, 1)

        return depth

    def __getitem__(self, index):
        try:
            offset = self.offset
            
            folder = self.folder_list[index]
            folder_path = os.path.join(self.folder, folder)

            img_list = [
                f for f in os.listdir(folder_path)
                if f.lower().endswith(".png")
            ]

            # if your depth filenames are like 0001.png / 1.png
            # img_list = sorted(img_list, key=lambda x: int(os.path.splitext(x)[0]))
            img_list = sorted(img_list, key=lambda x: int(os.path.splitext(x)[0][3:]))

            if len(img_list) == 0:
                raise RuntimeError(f"No depth png found in {folder_path}")

            first_frame_idx = random.randint(0, len(img_list) - 1)
            second_frame_idx = min(first_frame_idx + offset, len(img_list) - 1)
            
            first_path = os.path.join(folder_path, img_list[first_frame_idx])
            second_path = os.path.join(folder_path, img_list[second_frame_idx])
                    
            img = self._load_depth(first_path)
            next_img = self._load_depth(second_path)
            
            transform_img = img.unsqueeze(1)         # C x 1 x H x W
            next_transform_img = next_img.unsqueeze(1)

            cat_img = torch.cat([transform_img, next_transform_img], dim=1)  # C x 2 x H x W
            return cat_img

        except Exception as e:
            print("error", index, e)
            if index < self.__len__() - 1:
                return self.__getitem__(index + 1)
            else:
                return self.__getitem__(random.randint(0, self.__len__() - 1))