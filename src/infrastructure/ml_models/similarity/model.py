import io
import os
import json
import logging
import warnings
from datetime import datetime
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

from ..base_model import ModelBase

try:
    import mlflow
    import torchmetrics
    from tqdm import tqdm
    from .dataset import get_data_loaders  # <- этот get_data_loaders из файла выше
except ImportError:
    pass

warnings.simplefilter(action="ignore", category=FutureWarning)
logger = logging.getLogger(__name__)


class CelebrityMatcherModel(ModelBase):
    """
    Классификация на множество знаменитостей.
    - Бекбон ResNet50 с предобученными весами (замораживаем параметры)
    - Кастомный голова: Linear -> ReLU -> Dropout -> Linear(num_classes), Softmax в инференсе
    - Лосс: CrossEntropyLoss
    - Метрики: accuracy@1, accuracy@5, macro-F1, пер-класс accuracy (логгинг в mlflow)
    """

    _name: str = "celebrity_matcher"

    def __init__(self, out_features: int = 512, top_k: int = 5):
        super().__init__()
        self.top_k = top_k

        # Трансформации на инференсе
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

        self._classes_file = os.path.join(
            "datasets/celebs", "labels/processed", "classes.json"
        )
        self.id2label = None
        if os.path.exists(self._classes_file):
            with open(self._classes_file, "r", encoding="utf-8") as f:
                _id2label: Dict[str, str] = json.load(f)
            self.id2label = {int(k): v for k, v in _id2label.items()}
        self.num_classes = len(self.id2label) if self.id2label else None

        backbone = models.resnet50(weights="IMAGENET1K_V2")
        for p in backbone.parameters():
            p.requires_grad = False

        in_features = backbone.fc.in_features
        classifier_out = self.num_classes if self.num_classes else 2

        backbone.fc = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(out_features, classifier_out),
        )

        self._model = backbone
        self._model.to(self._device)
        self.loaded = False

    # ---- helpers ----
    def _ensure_head(self, num_classes: int, out_features: int = 512):
        """Переинициализировать голову, если число классов стало известно/изменилось."""
        if self.num_classes == num_classes:
            return
        in_features = (
            self._model.fc[0].in_features
            if isinstance(self._model.fc, nn.Sequential)
            else self._model.fc.in_features
        )
        self._model.fc = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(out_features, num_classes),
        )
        self._model.to(self._device)
        self.num_classes = num_classes
        logger.info(f"Classifier head reinitialized for {num_classes} classes.")

    def _setup_metrics(self, device, num_classes: int):
        acc1 = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(
            device
        )
        acc5 = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, top_k=5
        ).to(device)
        f1m = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        ).to(device)
        return nn.ModuleDict({"acc1": acc1, "acc5": acc5, "f1_macro": f1m})

    def train(
        self,
        epochs: int = 10,
        lr: float = 1e-3,
        criterion: nn.Module = nn.CrossEntropyLoss(),
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    ) -> None:
        if not self.loaded:
            try:
                self.load()
            except Exception:
                logger.warning("No existing checkpoint. Training from scratch.")

        train_loader, val_loader = get_data_loaders()
        num_classes = train_loader.dataset.num_classes

        self._ensure_head(num_classes=num_classes)

        if hasattr(train_loader.dataset, "id2label"):
            self.id2label = train_loader.dataset.id2label

        if optimizer is None:
            optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.1, patience=5
            )

        metrics = self._setup_metrics(self._device, num_classes)

        best_acc = 0.0
        mlflow.start_run(
            run_name=f"CelebrityMatcher {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        mlflow.log_params(
            {
                "epochs": epochs,
                "learning_rate": lr,
                "criterion": criterion.__class__.__name__,
                "optimizer": optimizer.__class__.__name__,
                "scheduler": scheduler.__class__.__name__ if scheduler else "None",
                "model_name": "resnet50",
                "out_features": 512,
                "fine_tuning": "feature_extraction",
                "num_classes": num_classes,
                "top_k": self.top_k,
            }
        )
        mlflow.set_tag("model_architecture", "resnet50_frozen_fc_custom")
        mlflow.log_artifact(__file__, artifact_path="code")

        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            for phase, loader in [("train", train_loader), ("val", val_loader)]:
                if phase == "train":
                    self._model.train()
                else:
                    self._model.eval()

                # reset metrics
                for m in metrics.values():
                    m.reset()

                running_loss = 0.0
                total = 0

                pbar = tqdm(loader, desc=f"{phase.capitalize()} Epoch {epoch + 1}")
                for images, labels in pbar:
                    images = images.to(self._device)
                    labels = labels.to(self._device)
                    bs = images.size(0)
                    total += bs

                    if phase == "train":
                        optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        logits = self._model(images)
                        loss = criterion(logits, labels)
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * bs

                    preds = torch.argmax(logits, dim=1)
                    # обновим метрики
                    metrics["acc1"].update(preds, labels)
                    # для top-5 передадим логиты, torchmetrics сам применит top_k
                    metrics["acc5"].update(logits, labels)
                    metrics["f1_macro"].update(preds, labels)

                    pbar.set_postfix(
                        {
                            "loss": f"{loss.item():.4f}",
                            "acc1": f"{metrics['acc1'].compute().item():.4f}",
                        }
                    )

                epoch_loss = running_loss / max(1, total)
                computed = {name: m.compute().item() for name, m in metrics.items()}

                # Логгинг
                mlflow.log_metrics(
                    {
                        f"{phase}_loss": epoch_loss,
                        f"{phase}_acc1": computed["acc1"],
                        f"{phase}_acc5": computed["acc5"],
                        f"{phase}_f1_macro": computed["f1_macro"],
                    },
                    step=epoch,
                )

                logger.info(
                    f"{phase}: loss={epoch_loss:.4f} acc1={computed['acc1']:.4f} "
                    f"acc5={computed['acc5']:.4f} f1_macro={computed['f1_macro']:.4f}"
                )

                if phase == "val":
                    # сохраняем лучший по acc1
                    if computed["acc1"] > best_acc:
                        best_acc = computed["acc1"]
                        self.save()
                        try:
                            mlflow.pytorch.log_model(
                                self._model,
                                "best_model",
                                registered_model_name=self._name,
                            )
                        except Exception:
                            # На локалке без MLflow Model Registry — просто пропустим
                            pass
                        mlflow.log_metric("best_val_acc1", best_acc)
                        logger.info(
                            f"New best model saved with val acc1: {best_acc:.4f}"
                        )

                    # шаг планировщика по валидационной метрике
                    if isinstance(
                        scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                    ):
                        scheduler.step(computed["acc1"])
                        mlflow.log_metric(
                            "learning_rate", optimizer.param_groups[0]["lr"], step=epoch
                        )

        mlflow.log_metric("final_best_val_acc1", best_acc)
        mlflow.set_tag("best_model", "True")
        mlflow.end_run()
        logger.info(f"Training complete. Best val acc@1: {best_acc:.4f}")

    def evaluate(self) -> dict:
        if not self.loaded:
            self.load()

        self._model.eval()
        _, test_loader = get_data_loaders()
        num_classes = test_loader.dataset.num_classes
        metrics = self._setup_metrics(self._device, num_classes)

        running_loss = 0.0
        total = 0
        criterion = nn.CrossEntropyLoss()

        pbar = tqdm(test_loader, desc="Evaluating", leave=True)
        with torch.no_grad():
            for images, labels in pbar:
                images = images.to(self._device)
                labels = labels.to(self._device)
                bs = images.size(0)
                total += bs

                logits = self._model(images)
                loss = criterion(logits, labels)
                running_loss += loss.item() * bs

                preds = torch.argmax(logits, dim=1)
                metrics["acc1"].update(preds, labels)
                metrics["acc5"].update(logits, labels)
                metrics["f1_macro"].update(preds, labels)

                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "acc1": f"{metrics['acc1'].compute().item():.4f}",
                    }
                )

        final = {name: m.compute().item() for name, m in metrics.items()}
        final["loss"] = running_loss / max(1, total)

        logger.info("Test Results:")
        for k, v in final.items():
            logger.info(f"{k}: {v:.4f}")

        return final

    def predict(self, image: bytes, top_k: int | None = None) -> List[Dict[str, float]]:
        """
        Возвращает top-k [(label, prob)], отсортированные по убыванию вероятности.
        """
        if not self.loaded:
            self.load()

        if self.id2label is None:
            # попробуем прочитать classes.json (нужно для маппинга id->имя)
            if os.path.exists(self._classes_file):
                with open(self._classes_file, "r", encoding="utf-8") as f:
                    _id2label = json.load(f)
                self.id2label = {int(k): v for k, v in _id2label.items()}
            else:
                raise ValueError("id2label mapping not found. Did you train the model?")

        self._model.eval()
        k = top_k or self.top_k
        try:
            with Image.open(io.BytesIO(image)) as img:
                img_t = self._transform(img).unsqueeze(0).to(self._device)

            with torch.no_grad():
                logits = self._model(img_t)
                probs = F.softmax(logits, dim=1)

            values, indices = torch.topk(probs, k=min(k, probs.size(1)), dim=1)
            values = values.squeeze(0).cpu().tolist()
            indices = indices.squeeze(0).cpu().tolist()

            results = [
                {"label": self.id2label[idx], "prob": float(p)}
                for idx, p in zip(indices, values)
            ]
            return results
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise ValueError(f"Could not process image: {e}") from e
