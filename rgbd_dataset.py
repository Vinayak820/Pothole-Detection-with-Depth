import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

class RGBDDataset(Dataset):
    def __init__(self, manifest_csv, transform=None):
        import pandas as pd
        self.df = pd.read_csv(manifest_csv)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def load_mask(self, label_path, size):
        h, w = size
        mask = np.zeros((h, w), dtype=np.uint8)

        if not os.path.exists(label_path):
            return mask

        with open(label_path, "r") as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                if len(parts) < 7:
                    continue

                coords = np.array(parts[1:]).reshape(-1, 2)
                coords[:, 0] *= w
                coords[:, 1] *= h
                pts = coords.astype(np.int32)
                cv2.fillPoly(mask, [pts], 1)

        return mask

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        rgb = cv2.imread(row["image"])
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = rgb.astype(np.float32) / 255.0

        depth = np.load(row["depth"]).astype(np.float32)
        if depth.max() > 1:
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        h, w = rgb.shape[:2]
        mask = self.load_mask(row["label"], (h, w))

        rgb = torch.tensor(rgb).permute(2, 0, 1)
        depth = torch.tensor(depth).unsqueeze(0)
        mask = torch.tensor(mask).unsqueeze(0).float()

        return rgb, depth, mask