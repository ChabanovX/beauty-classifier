from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
import os
from pathlib import Path

# ДОЛЖНО стоять до импорта torch (если ты хочешь скрыть GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Явно работаем только на CPU
device = torch.device("cpu")


class ModelBase(ABC):
    # используем заранее определённый device (CPU)
    _device: torch.device = device
    _models_dir: Path = Path("models")

    @property
    @abstractmethod
    def _name(self) -> str:
        """Уникальное имя модели (для файлов сохранения)"""
        raise NotImplementedError

    def __init__(self):
        self._model: Optional[nn.Module] = None
        self.loaded: bool = False

    def _get_model_path(self, path: Optional[str]) -> Path:
        """Вернуть полный путь к файлу: аргумент path имеет приоритет, иначе models/{name}.pt"""
        if path:
            return Path(path)
        return self._models_dir / f"{self._name}.pt"

    def save(self, path: Optional[str] = None) -> None:
        """Сохранить state_dict модели. Если путь не указан — сохраняем в models/{name}.pt"""
        if self._model is None:
            raise RuntimeError("Model is not initialized. Set self._model before saving.")
        save_path = self._get_model_path(path)
        # Создаём директорию, если нужно
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def load(self, path: Optional[str] = None) -> None:
        """Загрузить state_dict. При необходимости используется map_location=self._device"""
        if self._model is None:
            raise RuntimeError("Model is not initialized. Set self._model before loading weights.")
        load_path = self._get_model_path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
        state = torch.load(load_path, map_location=self._device)
        # Если state — полный checkpoint со словарём ['model_state_dict'], попробуй автоматически поддержать это
        if isinstance(state, dict) and "model_state_dict" in state:
            state_dict = state["model_state_dict"]
        else:
            state_dict = state
        self._model.load_state_dict(state_dict)
        self._model.to(self._device)
        self._model.eval()
        self.loaded = True
        print(f"Model loaded from {load_path}")
        print(f"Model device: {self._device}")

    @abstractmethod
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        epochs: int,
        lr: float,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> None:
        """Реализация обучения в подписи класса-потомка"""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Оценка модели на тестовом датасете"""
        raise NotImplementedError

    @abstractmethod
    def predict(self, input_data: Any) -> Any:
        """Прогноз: конкретная логика реализуется в подклассах"""
        pass
