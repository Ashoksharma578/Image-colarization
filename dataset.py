import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from skimage.color import rgb2lab
from imageio import imread
import random

class ColorizationDataset(Dataset):
    def __init__(self, img_dir, limit=None):
        self.img_dir = img_dir

        # Only valid image files
        self.img_list = [
            f for f in os.listdir(img_dir)
            if os.path.isfile(os.path.join(img_dir, f))
            and f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

        random.shuffle(self.img_list)

        #  APPLY LIMIT PROPERLY
        if limit is not None:
            self.img_list = self.img_list[:limit]

        print(f"[INFO] Loaded {len(self.img_list)} images from {img_dir}")

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_list[idx])

        try:
            image = imread(img_path)

            #  HANDLE NON RGB IMAGES
            if len(image.shape) != 3 or image.shape[2] != 3:
                print(f"[WARN] Non-RGB image skipped: {img_path}")
                return self.__getitem__((idx + 1) % len(self))

            image = cv2.resize(image, (128, 128))
            lab = rgb2lab(image).astype("float32")

            L = lab[:, :, 0] / 100.0
            ab = lab[:, :, 1:] / 128.0

            L = torch.from_numpy(L).unsqueeze(0).float()
            ab = torch.from_numpy(ab.transpose(2, 0, 1)).float()

            return L, ab

        except Exception as e:
            print(f"[ERROR] Failed loading {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))
