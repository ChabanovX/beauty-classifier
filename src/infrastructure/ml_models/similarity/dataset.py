from __future__ import annotations

import os
from typing import Optional, Tuple, Dict, Any, List

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from src.config import config


class CelebrityDataset(Dataset):
    def __init__(
        self,
        annotations_file: str,
        images_dir: Optional[str] = None,
        labels_dir: Optional[str] = None,
        transform: Optional[Any] = None,
    ) -> None:
        self.images_dir = (
            images_dir
            or os.path.join(config.ml.scut_data_path, "images")
        )
        self.labels_dir = (
            labels_dir
            or os.path.join(config.ml.scut_data_path, "labels/processed")
        )

        # Allow passing full path as annotations_file
        if os.path.isabs(annotations_file) or os.path.exists(annotations_file):
            csv_path = annotations_file
        else:
            csv_path = os.path.join(self.labels_dir, annotations_file)

        self.df = pd.read_csv(csv_path)
        if "image" not in self.df.columns:
            # assume first column is image name
            self.df.rename(columns={self.df.columns[0]: "image"}, inplace=True)

        # accept 'label' column or the second column
        if "label" not in self.df.columns:
            if len(self.df.columns) >= 2:
                self.df.rename(columns={self.df.columns[1]: "label"}, inplace=True)
            else:
                raise ValueError("CSV must contain at least two columns: image and label")

        self.transform = transform

        # Build label -> index mapping if labels are strings
        sample_label = self.df["label"].iloc[0]
        if isinstance(sample_label, str):
            # map unique names to indices in sorted order to have deterministic mapping
            unique_names = sorted(self.df["label"].unique().tolist())
            self.name2idx: Dict[str, int] = {n: i for i, n in enumerate(unique_names)}
            self.idx2name: Dict[int, str] = {i: n for n, i in self.name2idx.items()}
            # replace column with integer indices
            self.df["label_idx"] = self.df["label"].map(self.name2idx).astype(int)
            self._has_name_labels = True
        else:
            # assume numeric labels (int-like)
            self.df["label_idx"] = self.df["label"].astype(int)
            self.name2idx = {}
            self.idx2name = {}
            self._has_name_labels = False

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_name = row["image"]
        img_path = os.path.join(self.images_dir, img_name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        with Image.open(img_path) as img:
            img = img.convert("RGB")
            if self.transform is not None:
                img = self.transform(img)

        label = int(row["label_idx"])  # guaranteed int
        label = torch.tensor(label, dtype=torch.long)

        return img, label


def get_data_loaders(
    batch_size: int = 32,
    train_file: str = "train.csv",
    val_file: str = "val.csv",
    test_file: Optional[str] = "test.csv",
    images_dir: Optional[str] = None,
    labels_dir: Optional[str] = None,
    num_workers: int = 2,
    image_size: int = 224,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], Dict[int, str]]:

    # Transforms
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Datasets
    train_ds = CelebrityDataset(
        annotations_file=train_file,
        images_dir=images_dir,
        labels_dir=labels_dir,
        transform=train_transform,
    )

    val_ds = CelebrityDataset(
        annotations_file=val_file,
        images_dir=images_dir,
        labels_dir=labels_dir,
        transform=val_transform,
    )

    test_ds = None
    if test_file is not None:
        try:
            test_ds = CelebrityDataset(
                annotations_file=test_file,
                images_dir=images_dir,
                labels_dir=labels_dir,
                transform=val_transform,
            )
        except Exception:
            test_ds = None

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers) if test_ds is not None else None

    # Build class_map (index -> name) — prefer val_ds.idx2name if available
    class_map: Dict[int, str] = {}
    if getattr(train_ds, "idx2name", None):
        class_map = train_ds.idx2name
    elif getattr(val_ds, "idx2name", None):
        class_map = val_ds.idx2name
    elif test_ds is not None and getattr(test_ds, "idx2name", None):
        class_map = test_ds.idx2name

    return train_loader, val_loader, test_loader, class_map


# Example usage:
# train_loader, val_loader, test_loader, class_map = get_data_loaders(batch_size=32)
# print('num classes:', len(class_map) if class_map else 'unknown (numeric labels)')
