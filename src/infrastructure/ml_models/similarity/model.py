import io
import logging
import os
from typing import List, Tuple, Dict, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm
from PIL import Image

from facenet_pytorch import MTCNN
from ..base_model import ModelBase
from .dataset import get_data_loaders


class CelebrityLookalikeModel(ModelBase):

    _name: str = "celebrity_lookalike"

    def __init__(
        self,
        num_classes: Optional[int] = None,
        class_map: Optional[Dict[int, str]] = None,
        freeze_backbone: bool = True,
        mtcnn_device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        # Keep or override device from ModelBase
        self._device = getattr(self, "_device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self._mtcnn_device = mtcnn_device or self._device

        # image transform applied to face crops
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

        # face detector (optional)
        if MTCNN is not None:
            self._mtcnn = MTCNN(keep_all=False, device=self._mtcnn_device)
        else:
            self._mtcnn = None

        # backbone
        backbone = models.resnet50(weights="IMAGENET1K_V2")
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        in_features = backbone.fc.in_features
        # placeholder head; will be replaced when num_classes known
        self._num_classes = num_classes or 1000
        backbone.fc = nn.Linear(in_features, self._num_classes)

        self._model = backbone.to(self._device)

        # class index -> name mapping
        self.class_map = class_map

        self.loaded = False

    # ------------------ compatibility wrappers ------------------
    def save(self) -> None:
        """Save model and metadata to the default models/<name>.pt path.

        This overrides ModelBase.save() to store a payload containing
        `model_state`, `num_classes` and `class_map` for easier reloading.
        """
        if self._model is None:
            raise RuntimeError("No model to save")

        os.makedirs(self._models_dir, exist_ok=True)
        save_path = os.path.join(self._models_dir, f"{self._name}.pt")

        payload = {
            "model_state": self._model.state_dict(),
            "num_classes": self._num_classes,
            "class_map": self.class_map,
        }
        torch.save(payload, save_path)
        logging.info(f"Model saved to {save_path}")

    def load(self, path: Optional[str] = None) -> None:
        """Load model state from given path or default models/<name>.pt."""
        if self._model is None:
            raise RuntimeError("Model architecture not initialized before load()")

        load_path = path or os.path.join(self._models_dir, f"{self._name}.pt")
        logging.debug(f"Loading {self._name} from {load_path}")

        state = torch.load(load_path, map_location=self._device)

        # Support payload with metadata or raw state_dict
        if isinstance(state, dict) and "model_state" in state:
            model_state = state["model_state"]
            file_num_classes = state.get("num_classes")
            file_class_map = state.get("class_map")
        else:
            model_state = state
            file_num_classes = None
            file_class_map = None

        # If payload contains num_classes and differs from current, recreate head
        if file_num_classes and file_num_classes != self._num_classes:
            self.set_num_classes(file_num_classes)

        self._model.load_state_dict(model_state)
        # Restore class_map if present
        if file_class_map is not None:
            self.class_map = file_class_map

        self._model.to(self._device)
        self._model.eval()
        self.loaded = True
        logging.info(f"{self._name} loaded from {load_path} to device {self._device}")

    # ------------------ model utilities ------------------
    def set_num_classes(self, num_classes: int) -> None:
        """(Re)create classifier head for given number of classes."""
        in_features = self._model.fc.in_features
        self._model.fc = nn.Linear(in_features, num_classes).to(self._device)
        self._num_classes = num_classes

    @staticmethod
    def _topk_accuracy(output: torch.Tensor, target: torch.Tensor, k: int = 5) -> float:
        """Compute top-k accuracy for one batch."""
        with torch.no_grad():
            _, preds = output.topk(k, dim=1, largest=True, sorted=True)
            target = target.view(-1, 1).expand_as(preds)
            correct = (preds == target).any(dim=1).float().sum().item()
            return correct / output.size(0)

    # ------------------ training loop (compatible with ModelBase.train signature) ------------------
    def train(
        self,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        lr: float = 1e-3,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> None:
        """
        Train the model. Signature matches ModelBase expectations but keeps the
        convenience of falling back to internal `get_data_loaders()` when
        loaders are not provided.
        """
        if not self._model:
            raise ValueError("Model not initialized")

        # If loaders not provided, try to get them from dataset helper
        if train_loader is None or val_loader is None:
            loaders = get_data_loaders()
            if len(loaders) >= 2:
                train_loader, val_loader = loaders[0], loaders[1]
            else:
                raise ValueError("train_loader and val_loader must be provided")

        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        if optimizer is None:
            optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=5
            )

        # Attempt to infer classes from dataset if not provided
        if hasattr(train_loader.dataset, "classes") and self._num_classes == 1000:
            self._num_classes = len(train_loader.dataset.classes)
            self.set_num_classes(self._num_classes)

        best_val_loss = float("inf")

        for epoch in range(epochs):
            logging.info(f"Epoch {epoch + 1}/{epochs}")

            for phase, loader in [("train", train_loader), ("val", val_loader)]:
                if phase == "train":
                    self._model.train()
                else:
                    self._model.eval()

                running_loss = 0.0
                running_acc = 0.0
                running_top5 = 0.0
                total = 0

                progress_bar = tqdm(loader, desc=f"{phase.capitalize()} Epoch {epoch + 1}")

                for inputs, labels in progress_bar:
                    inputs = inputs.to(self._device)
                    labels = labels.to(self._device).long()
                    batch_size = inputs.size(0)
                    total += batch_size

                    if phase == "train":
                        optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = self._model(inputs)  # [batch, num_classes]
                        loss = criterion(outputs, labels)

                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * batch_size

                    # accuracy
                    preds = outputs.argmax(dim=1)
                    running_acc += (preds == labels).sum().item()

                    # top-5
                    running_top5 += self._topk_accuracy(outputs, labels, k=5) * batch_size

                    avg_loss = running_loss / total
                    avg_acc = running_acc / total
                    avg_top5 = running_top5 / total

                    progress_bar.set_postfix(
                        {"loss": f"{avg_loss:.4f}", "acc": f"{avg_acc:.4f}", "top5": f"{avg_top5:.4f}"}
                    )

                epoch_loss = running_loss / total
                epoch_acc = running_acc / total
                epoch_top5 = running_top5 / total

                logging.info(f"{phase.capitalize()} Loss: {epoch_loss:.6f}")
                logging.info(f"{phase.capitalize()} Acc:  {epoch_acc:.4f}")
                logging.info(f"{phase.capitalize()} Top-5: {epoch_top5:.4f}")

                # save best validation model
                if phase == "val" and epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    self.save()
                    logging.info(f"New best model saved with val loss: {best_val_loss:.4f}")

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_loss)

        logging.info(f"Training complete. Best val loss: {best_val_loss:.4f}")

    # ------------------ evaluation (compatible with ModelBase.evaluate) ------------------
    def evaluate(self, test_loader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """Evaluate using the provided test_loader. If not given, tries to
        get test loader from `get_data_loaders()`.
        """
        if not self._model:
            raise ValueError("Model not loaded")

        if test_loader is None:
            loaders = get_data_loaders()
            if len(loaders) >= 3:
                test_loader = loaders[2]
            else:
                logging.warning("get_data_loaders did not return a test loader. Skipping evaluation.")
                return {}

        self._model.eval()

        running_acc = 0.0
        running_top5 = 0.0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Evaluating"):
                inputs = inputs.to(self._device)
                labels = labels.to(self._device).long()
                batch_size = inputs.size(0)
                total += batch_size

                outputs = self._model(inputs)
                preds = outputs.argmax(dim=1)

                running_acc += (preds == labels).sum().item()
                running_top5 += self._topk_accuracy(outputs, labels, k=5) * batch_size

        metrics = {
            "accuracy": running_acc / total if total > 0 else 0.0,
            "top5": running_top5 / total if total > 0 else 0.0,
        }

        logging.info("Test Results:")
        for k, v in metrics.items():
            logging.info(f"  {k}: {v:.4f}")

        return metrics

    # ------------------ inference (compatible with ModelBase.predict) ------------------
    def _detect_and_crop(self, image: Image.Image) -> Image.Image:
        """Detect a single face and return cropped PIL image. If no detector, returns original image."""
        if self._mtcnn is None:
            # No detector installed — assume image already is a face
            return image

        try:
            face = self._mtcnn(image)
            if face is None:
                raise ValueError("No face detected")

            face_pil = transforms.ToPILImage()(face)
            return face_pil
        except Exception as e:
            raise ValueError(f"Face detection failed: {e}") from e

    def predict(self, input_data: Any, topk: int = 5) -> List[Tuple[str, float]]:
        """Return top-k (name, probability) pairs.

        input_data may be:
          - raw bytes of an image
          - PIL.Image.Image
          - path to image file (str)

        If class_map is not available, returns indices as strings.
        """
        if not self._model:
            raise ValueError("Model not loaded")

        # Normalize input types into bytes
        try:
            if isinstance(input_data, bytes):
                img_bytes = input_data
            elif isinstance(input_data, Image.Image):
                buf = io.BytesIO()
                input_data.save(buf, format="PNG")
                img_bytes = buf.getvalue()
            elif isinstance(input_data, str):
                with open(input_data, "rb") as f:
                    img_bytes = f.read()
            else:
                raise ValueError("Unsupported input_data type for predict()")

            with Image.open(io.BytesIO(img_bytes)) as img:
                img = img.convert("RGB")

                # detect & crop
                try:
                    face = self._detect_and_crop(img)
                except Exception as e:
                    logging.warning(f"Face detection failed, using full image: {e}")
                    face = img

                tensor = self._transform(face).unsqueeze(0).to(self._device)

            self._model.eval()
            with torch.no_grad():
                logits = self._model(tensor)
                probs = F.softmax(logits, dim=1).cpu().numpy().flatten()

            topk_idx = probs.argsort()[::-1][:topk]
            results: List[Tuple[str, float]] = []
            for idx in topk_idx:
                name = self.class_map.get(int(idx), str(int(idx))) if self.class_map else str(int(idx))
                results.append((name, float(probs[int(idx)])))

            return results

        except Exception as e:
            logging.error(f"Error while predicting: {e}")
            raise


