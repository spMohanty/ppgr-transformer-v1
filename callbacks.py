"""
Custom callbacks for PyTorch Lightning.
"""
from typing import Any, Dict, Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import OneCycleLR


class LRSchedulerCallback(Callback):
    """
    Callback for learning rate scheduling using OneCycleLR.
    """
    def __init__(
        self, 
        optimizer: Optimizer, 
        base_lr: float, 
        total_steps: int, 
        pct_start: float = 0.1, 
        max_lr_factor: float = 1.5
    ):
        """
        Initialize the LR scheduler callback.
        
        Args:
            optimizer: PyTorch optimizer
            base_lr: Base learning rate
            total_steps: Total number of training steps
            pct_start: Percentage of training to increase learning rate
            max_lr_factor: Factor to multiply base_lr by for max_lr
        """
        super().__init__()
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.max_lr_factor = max_lr_factor
        
        # Create scheduler
        self.scheduler = OneCycleLR(
            optimizer,
            max_lr=base_lr * max_lr_factor,
            total_steps=total_steps,
            pct_start=pct_start,
            div_factor=max_lr_factor,
            final_div_factor=1000.0,
            anneal_strategy='cos'
        )
        
        self.last_lr = None

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Dict[str, Any], 
        batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None
    ):
        """
        Step the scheduler after each batch and log the learning rate.
        """
        self.scheduler.step()
        current_lr = self.scheduler.get_last_lr()[0]
        
        # Only log if LR has changed significantly
        # if self.last_lr is None or abs(current_lr - self.last_lr) / self.base_lr > 0.001:
        pl_module.log("learning_rate", current_lr, on_step=True, on_epoch=False, prog_bar=True)
        self.last_lr = current_lr
            