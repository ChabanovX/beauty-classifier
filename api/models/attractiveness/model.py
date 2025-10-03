import io
import logging

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from ..base_model import ModelBase
from .dataset import get_data_loaders


class AttractivenessModel(ModelBase):
    _name: str = "attractiveness_classifier"

    def __init__(self):
        super().__init__()
        self._transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Lambda(lambda x: x.convert("RGB") if x.mode != "RGB" else x),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        model = models.resnet50(weights="IMAGENET1K_V2")
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 512),  # 512 features in the last layer
            nn.ReLU(),
            nn.Dropout(0.3),  # Regularization to prevent overfitting
            nn.Linear(512, 1),  # Final output: 1 value (the attractiveness score)
            nn.Sigmoid(),
        )
        self._model = model
        self._model.to(self._device)
        self.loaded = False

    def train(
        self,
        epochs: int = 10,
        lr: float = 0.001,
        criterion: nn.Module = nn.MSELoss(),
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    ) -> None:
        if not self._model:
            raise ValueError("Model not initialized")

        # Default components if not provided
        if optimizer is None:
            optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=5
            )

        train_loader, val_loader = get_data_loaders()

        best_loss = float("inf")
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            print("-" * 10)

            # Training and validation phases
            for phase, dataloader in [("train", train_loader), ("val", val_loader)]:
                if phase == "train":
                    self._model.train()
                else:
                    self._model.eval()

                running_loss = 0.0
                running_mae = 0.0

                for inputs, labels in dataloader:
                    inputs = inputs.to(self._device)
                    labels = labels.to(self._device).view(-1, 1)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = self._model(inputs)
                        loss = criterion(outputs, labels)
                        mae = nn.L1Loss()(outputs, labels)

                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_mae += mae.item() * inputs.size(0)

                epoch_loss = running_loss / len(dataloader.dataset)
                epoch_mae = running_mae / len(dataloader.dataset)

                print(
                    f"{phase.capitalize()} Loss: {epoch_loss:.4f} MAE: {epoch_mae:.4f}"
                )

                # Save best model
                if phase == "val" and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    self.save()
                    print(f"New best model saved with loss: {best_loss:.4f}")

            # Scheduler step
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_loss)
            else:
                scheduler.step()

            print()

        print(f"Training complete. Best val loss: {best_loss:.4f}")

    def predict(self, image: bytes) -> float:
        """
        Predict face attractiveness from image

        Args:
            image (bytes): face image as bytes

        Raises:
            ValueError: if image is not a valid image

        Returns:
            float: predicted attractiveness score [0, 1]
        """
        self._model.eval()
        try:
            # Convert bytes to PIL Image
            with Image.open(io.BytesIO(image)) as image:
                image_tensor = self._transform(image).unsqueeze(0).to(self._device)

            # Make prediction
            with torch.no_grad():
                output = self._model(image_tensor)

            return 1 + output.item() * 4
        except Exception as e:
            print(f"Error processing image: {e}")
            raise ValueError(f"Could not process image: {e}")

    def evaluate(self) -> tuple[float, float]:
        if not self._model:
            raise ValueError("Model not loaded")

        self._model.eval()
        total_mae = 0.0
        total_mse = 0.0
        mae = nn.L1Loss()
        mse = nn.MSELoss()

        _, test_loader = get_data_loaders()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self._device)
                labels = labels.to(self._device).view(-1, 1)

                outputs = self._model(inputs)
                mae_loss = mae(outputs, labels)
                mse_loss = mse(outputs, labels)

                total_mae += mae_loss.item() * inputs.size(0)
                total_mse += mse_loss.item() * inputs.size(0)

        mae = total_mae / len(test_loader.dataset)
        rmse = (total_mse / len(test_loader.dataset)) ** 0.5

        print(f"Test MAE: {mae:.4f}")
        print(f"Test RMSE: {rmse:.4f}")

        return {"mae": mae, "rmse": rmse}


attractiveness_model = AttractivenessModel()
