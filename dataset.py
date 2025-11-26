import os
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.color import rgb2lab
from imageio import imread
import cv2
import random

class ColorizationDataset(Dataset):
    def __init__(self, img_dir, limit=None):
        self.img_dir = img_dir
        
        # Load only valid image files
        self.img_list = [
            f for f in os.listdir(img_dir)
            if os.path.isfile(os.path.join(img_dir, f))
        ]

        # Optional limit for training speed
        if limit:
            self.img_list = self.img_list[:limit]

        random.shuffle(self.img_list)
        print(f"[INFO] Loaded {len(self.img_list)} images from {img_dir}")

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_list[idx])

        try:
            # Read
            image = imread(img_path)

            # Skip None images
            if image is None:
                print(f"[WARN] Failed to read {img_path}, skipping...")
                return self.__getitem__((idx + 1) % len(self))

            # Ensure 3-channel image only
            if len(image.shape) != 3 or image.shape[2] != 3:
                print(f"[WARN] Non-RGB image skipped: {img_path}")
                return self.__getitem__((idx + 1) % len(self))

            # Resize all images
            image = cv2.resize(image, (256, 256))

            # Convert to LAB
            lab = rgb2lab(image).astype("float32")

            # Normalize
            L = lab[:, :, 0] / 100.0
            ab = lab[:, :, 1:] / 128.0

            # Convert to tensors
            L = torch.from_numpy(L).unsqueeze(0).float()
            ab = torch.from_numpy(ab.transpose(2, 0, 1)).float()

            return L, ab

        except Exception as e:
            print(f"[ERROR] Could not process {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))
