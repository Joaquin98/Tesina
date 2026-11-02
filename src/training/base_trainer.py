"""
Base trainer class for consistent training across different models.

This class provides common training loop functionality, checkpointing,
logging, and evaluation to ensure fair model comparisons.
"""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
logger = logging.getLogger(__name__)

class BaseTrainer(ABC):
    """Base trainer class for rugby play classification models."""

    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, optimizer: Optimizer, criterion: nn.Module, device: torch.device, config: Dict[str, Any], scheduler: Optional[_LRScheduler]=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        self.scheduler = scheduler
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.output_dir = Path(config['experiment']['output_dir'])
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f'Trainer initialized. Output: {self.output_dir}')

    @abstractmethod
    def train_epoch(self) -> Dict[str, float]:
        """
        Execute one training epoch.
        
        Returns:
            Dictionary with training metrics (loss, accuracy, etc.)
        """
        pass

    @abstractmethod
    def validate(self) -> Dict[str, float]:
        """
        Run validation.
        
        Returns:
            Dictionary with validation metrics (loss, accuracy, etc.)
        """
        pass

    def train(self, num_epochs: int) -> None:
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
        """
        logger.info(f'Starting training for {num_epochs} epochs')
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            train_metrics = self.train_epoch()
            logger.info(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
            val_metrics = self.validate()
            logger.info(f"Epoch {epoch + 1}/{num_epochs} - Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
            if self.scheduler is not None:
                self.scheduler.step()
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.save_checkpoint('best_model.pth', val_metrics)
            if self._should_early_stop(val_metrics):
                logger.info(f'Early stopping triggered at epoch {epoch + 1}')
                break
        logger.info('Training completed!')

    def save_checkpoint(self, filename: str, metrics: Dict[str, float]) -> None:
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename
        checkpoint = {'epoch': self.current_epoch, 'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'metrics': metrics, 'config': self.config}
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        torch.save(checkpoint, checkpoint_path)
        logger.info(f'Checkpoint saved: {checkpoint_path}')

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(_resolve_model_artifact(checkpoint_path), map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(f'Checkpoint loaded from: {checkpoint_path}')

    def _should_early_stop(self, val_metrics: Dict[str, float]) -> bool:
        """Check if early stopping criteria are met."""
        return False
