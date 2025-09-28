from ...config import config

import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os


class AttractivenessDataset(Dataset):
    def __init__(self, annotations_file: str, transform=None):
        self.img_labels = pd.read_csv(
            os.path.join(config.data_path, "labels/processed", annotations_file),
        )
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        label = self.img_labels.iloc[idx, 1]
        label = torch.tensor(
            label, dtype=torch.float32
        )  # Regression target must be float32
        
        img_path = os.path.join(
            config.data_path, "images", self.img_labels.iloc[idx, 0]
        )
        
        with Image.open(img_path) as image:
            if self.transform:
                image = self.transform(image)

        return image, label


def get_data_loaders(batch_size=32):
    # Define data transformations
    # 1. For training: Augmentation + Normalization
    # 2. For testing: Only Normalization
    train_transform = transforms.Compose(
        [
            transforms.Resize(
                (224, 224)
            ),  # Resize to input size expected by most pre-trained models
            transforms.RandomHorizontalFlip(p=0.5),  # Data augmentation: random flip
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2
            ),  # Slight color variation
            transforms.ToTensor(),  # Convert PIL Image to Tensor (0-1 range)
            transforms.Normalize(  # Normalize with ImageNet stats (standard practice)
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create Datasets
    train_dataset = AttractivenessDataset(
        annotations_file="train.txt", transform=train_transform
    )

    test_dataset = AttractivenessDataset(
        annotations_file="test.txt", transform=test_transform
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Crucial for training to shuffle the data
        num_workers=2,  # Number of subprocesses for data loading
        pin_memory=True,  # Faster data transfer to GPU
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle for testing
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, test_loader
