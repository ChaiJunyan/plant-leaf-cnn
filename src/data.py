"""Manifest-driven data loading. No split generation or test selection here."""
import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class LeafDataset(Dataset):
    def __init__(self, data_root, manifest, image_size, mean, std, training=False):
        self.data_root = Path(data_root)
        with Path(manifest).open(encoding="utf-8", newline="") as f:
            self.rows = list(csv.DictReader(f))
        ops = [transforms.Resize((image_size, image_size))]
        if training:
            ops += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(20),
                    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10)]
        self.transform = transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize(mean, std)])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.data_root / row["path"]) as image:
            x = self.transform(image.convert("RGB"))
        return x, int(row["label"]), index


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
