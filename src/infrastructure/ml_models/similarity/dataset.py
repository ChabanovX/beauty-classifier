import os
import json
import pandas as pd
from typing import Optional, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


CELEB_DATA_PATH = "datasets/scut"


class CelebrityDataset(Dataset):
    """
    CSV формат: image,label
    - image: относительный путь от datasets/celebs/images/
    - label: строка с именем знаменитости (например 'Tom_Hanks')
    """

    def __init__(
        self,
        data_file: str,
        transform=None,
        classes_file: Optional[str] = None,
        label2id: Optional[Dict[str, int]] = None,
    ):
        self.df = pd.read_csv(
            os.path.join(CELEB_DATA_PATH, "labels/processed", data_file)
        )
        self.df["label"] = self.df["label"].astype(str)
        self.transform = transform

        # Куда и откуда читать словари соответствий
        self._classes_file = classes_file or os.path.join(
            CELEB_DATA_PATH, "labels/processed", "classes.json"
        )

        if label2id is not None:
            self.label2id = label2id
        else:
            # если classes.json уже есть — читаем; иначе создадим по train.txt при первом использовании
            if os.path.exists(self._classes_file):
                with open(self._classes_file, "r", encoding="utf-8") as f:
                    id2label = json.load(f)
                self.label2id = {v: int(k) for k, v in id2label.items()}
            else:
                # строим из текущего файла (ожидается, что это train.txt)
                unique_labels = sorted(self.df["label"].unique().tolist())
                self.label2id = {lbl: i for i, lbl in enumerate(unique_labels)}
                # сохраним id2label = {id: label}
                id2label = {str(i): lbl for lbl, i in self.label2id.items()}
                os.makedirs(os.path.dirname(self._classes_file), exist_ok=True)
                with open(self._classes_file, "w", encoding="utf-8") as f:
                    json.dump(id2label, f, ensure_ascii=False, indent=2)

        self.id2label = {i: lbl for lbl, i in self.label2id.items()}

    def __len__(self) -> int:
        return len(self.df)

    @property
    def num_classes(self) -> int:
        return len(self.label2id)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(CELEB_DATA_PATH, "images", row["image"])

        with Image.open(img_path) as image:
            image = image.convert("RGB")
            if self.transform:
                image = self.transform(image)

        label_str = row["label"]
        label_idx = self.label2id[label_str]
        label_tensor = torch.tensor(label_idx, dtype=torch.long)  # для CrossEntropyLoss

        return image, label_tensor


def get_data_loaders(batch_size: int = 32):
    # Трансформации — как в аттрактивности (аугментации только на train)
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    classes_file = os.path.join(CELEB_DATA_PATH, "labels/processed", "classes.json")

    train_ds = CelebrityDataset(
        data_file="train.txt",
        transform=train_transform,
        classes_file=classes_file,
    )
    # test/val должен использовать тот же label2id, чтобы индексы совпадали
    val_ds = CelebrityDataset(
        data_file="test.txt",
        transform=test_transform,
        classes_file=classes_file,
        label2id=train_ds.label2id,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader
