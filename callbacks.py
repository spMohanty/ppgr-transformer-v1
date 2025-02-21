import pytorch_lightning as pl
from torch.optim.lr_scheduler import OneCycleLR


class LRSchedulerCallback(pl.Callback):
    def __init__(self, optimizer, base_lr, total_steps, pct_start=0.1, anneal_strategy='cos', cycle_momentum=False):
        """
        Consolidates all learning rate scheduling logic.
        
        Args:
            optimizer: The optimizer to schedule
            base_lr: Base learning rate (will scale up to 1.5x this value)
            total_steps: Total training steps
            pct_start: Percentage of steps for increasing phase
            anneal_strategy: Annealing strategy ('cos' or 'linear')
            cycle_momentum: Whether to cycle momentum (False for AdamW)
        """
        self.optimizer = optimizer
        self.max_lr = base_lr * 1.5  # Scale up to 1.5x the base learning rate
        self.scheduler = OneCycleLR(
            optimizer,
            max_lr=self.max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy=anneal_strategy,
            cycle_momentum=cycle_momentum
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Step the scheduler
        self.scheduler.step()
        
        # Get the current learning rate
        lr = self.optimizer.param_groups[0]['lr']
        
        # Log to wandb
        trainer.logger.experiment.log({'learning_rate': lr})