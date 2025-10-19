import logging
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class ModelBase(ABC):
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _models_dir = "models"

    @property
    @abstractmethod
    def _name(self) -> str:
        raise NotImplementedError

    def __init__(self):
        self._model: torch.nn.Module = None
        self.loaded = False

    def save(self) -> None:
        torch.save(self._model.state_dict(), f"{self._models_dir}/{self._name}.pt")

    def load(self, path: str | None = None):
        """Load model weights"""
        if self.loaded:
            return
        load_path = f"{self._models_dir}/{self._name}.pt" or path
        if not load_path:
            raise ValueError(f"Model {self._name} not found")
        logger.debug(f"Loading {self._name} from {load_path}")
        self._model.load_state_dict(torch.load(load_path, map_location=self._device))
        self._model.to(self._device)
        self._model.eval()
        self.loaded = True
        logger.debug(f"{self._name} loaded from {load_path} to device {self._device}")

    @abstractmethod
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        lr: float,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, test_loader: DataLoader) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def predict(self, input_data):
        raise NotImplementedError
