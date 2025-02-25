#!/usr/bin/env python
"""
Main training script for the Meal Glucose Forecast model.
"""
import os
import logging
import warnings

# Suppress the nested tensors prototype warning from PyTorch
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="The PyTorch API of nested tensors is in prototype stage"
)

import click
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import RichModelSummary, RichProgressBar
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch import seed_everything
from dataclasses import asdict

from loguru import logger

from config import ExperimentConfig, generate_experiment_name
from callbacks import LRSchedulerCallback
from dataset import create_cached_dataset
from models.forecast_model import MealGlucoseForecastModel
from utils import create_click_options


def get_dataloaders(config: ExperimentConfig):
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, training_dataset)
    """
    # Create or load cached dataset
    (training_dataset, validation_dataset, test_dataset, categorical_encoders, continuous_scalers) = create_cached_dataset(
        dataset_version=config.dataset_version,
        debug_mode=config.debug_mode,
        validation_percentage=config.validation_percentage,
        test_percentage=config.test_percentage,
        min_encoder_length=config.min_encoder_length,
        max_encoder_length=config.max_encoder_length,
        prediction_length=config.prediction_length,
        encoder_length_randomization=config.encoder_length_randomization,
        is_food_anchored=config.is_food_anchored,
        sliding_window_stride=config.sliding_window_stride,
        use_meal_level_food_covariates=config.use_meal_level_food_covariates,
        use_microbiome_embeddings=config.use_microbiome_embeddings,
        use_bootstraped_food_embeddings=config.use_bootstraped_food_embeddings,
        group_by_columns=config.group_by_columns,
        temporal_categoricals=config.temporal_categoricals,
        temporal_reals=config.temporal_reals,
        user_static_categoricals=config.user_static_categoricals,
        user_static_reals=config.user_static_reals,
        food_categoricals=config.food_categoricals,
        food_reals=config.food_reals,
        targets=config.targets,
        cache_dir=config.cache_dir,
        use_cache=config.use_cache,
    )
    
    from dataset import meal_glucose_collate_fn
    # Create data loaders
    train_loader = DataLoader(
        training_dataset, 
        batch_size=config.batch_size, 
        num_workers=config.dataloader_num_workers, 
        pin_memory=True, 
        persistent_workers=True, 
        shuffle=True,
        collate_fn=meal_glucose_collate_fn
    )
    val_loader = DataLoader(
        validation_dataset, 
        batch_size=config.batch_size, 
        num_workers=config.dataloader_num_workers, 
        pin_memory=True, 
        persistent_workers=True,
        shuffle=False,
        collate_fn=meal_glucose_collate_fn
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        num_workers=config.dataloader_num_workers, 
        pin_memory=True, 
        persistent_workers=True,
        shuffle=False,
        collate_fn=meal_glucose_collate_fn
    )
        
    return train_loader, val_loader, test_loader, training_dataset


def get_trainer(config: ExperimentConfig, callbacks):
    """
    Create a PyTorch Lightning trainer.
    
    Args:
        config: Experiment configuration
        callbacks: List of callbacks
        
    Returns:
        PyTorch Lightning Trainer
    """
    # Initialize WandB logger
    wandb_logger = WandbLogger(
        project=config.wandb_project,
        name=config.wandb_run_name,
        config=asdict(config),
        log_model=True,
    )
    
    # Determine precision
    precision_value = int(config.precision) if config.precision == "32" else "bf16"
    
    # Create trainer
    trainer = pl.Trainer(
        profiler="simple",
        max_epochs=config.max_epochs,
        enable_checkpointing=True,
        logger=wandb_logger,
        callbacks=callbacks,
        precision=precision_value,
        gradient_clip_val=config.gradient_clip_val
    )
    
    return trainer


def prepare_callbacks(config: ExperimentConfig, model: MealGlucoseForecastModel, train_loader: DataLoader):
    """
    Prepare callbacks for training.
    
    Args:
        config: Experiment configuration
        model: Model instance
        train_loader: Training data loader
        
    Returns:
        List of callbacks
    """
    # Initialize optimizer
    optimizer = model.configure_optimizers()
    
    # Standard callbacks
    rich_model_summary = RichModelSummary(max_depth=2)
    rich_progress_bar = RichProgressBar()
    
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = os.path.join(config.checkpoint_base_dir, config.wandb_run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Checkpoint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor=config.checkpoint_monitor,
        mode=config.checkpoint_mode,
        save_top_k=config.checkpoint_top_k,
        filename="best-{epoch}-{val_q_loss:.2f}",
        save_last=True
    )    
    
    # Learning rate scheduler callback
    lr_scheduler = LRSchedulerCallback(
        optimizer=optimizer,
        base_lr=config.optimizer_lr,
        total_steps=config.max_epochs * len(train_loader),
        pct_start=config.optimizer_lr_scheduler_pct_start,
    )
    
    return [rich_model_summary, rich_progress_bar, lr_scheduler, checkpoint_callback]


@click.command()
@create_click_options(ExperimentConfig)
def main(**kwargs):
    """
    Main entry point for training.
    """
    # Setup logging
    logging.getLogger().setLevel(logging.DEBUG if kwargs["debug_mode"] else logging.INFO)
    config = ExperimentConfig(**kwargs)
    
    # Seed for reproducibility
    seed_everything(config.random_seed)
    
    # Generate experiment name
    experiment_name = generate_experiment_name(config, kwargs)
    config.wandb_run_name = experiment_name
    
    logger.info(f"Starting experiment: {experiment_name}")
    
    # Debug mode adjustments
    if config.debug_mode:
        config.dataloader_num_workers = 1
        logger.warning("Debug mode enabled: reducing dataloaderworker count to 1")

    # Create data loaders
    train_loader, val_loader, test_loader, training_dataset = get_dataloaders(config)
    logger.info(f"Dataset loaded: {len(training_dataset)} training, {len(val_loader.dataset)} validation, {len(test_loader.dataset)} test samples")
    
    # Create model
    model = MealGlucoseForecastModel.from_dataset(training_dataset, config)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Setup callbacks and trainer
    callbacks = prepare_callbacks(config, model, train_loader)
    trainer = get_trainer(config, callbacks)
    
    # Training
    logger.info("Starting training")
    trainer.fit(model, train_loader, val_loader)
    logger.info("Training complete")
    
    # Load best model for evaluation
    best_model_path = trainer.checkpoint_callback.best_model_path
    if best_model_path:
        logger.info(f"Loading best model from: {best_model_path}")
        model = MealGlucoseForecastModel.load_from_checkpoint(
            best_model_path,
            config=config,
            num_foods=training_dataset.num_foods,
            food_macro_dim=training_dataset.num_nutrients,
            food_names=training_dataset.food_names,
            food_group_names=training_dataset.food_group_names,
        )
    
    # Testing
    logger.info("Starting testing")
    test_results = trainer.test(model, test_loader)
    logger.info(f"Test results: {test_results}")
    
    # Example inference
    logger.info("Running example inference")
    model.eval()
    with torch.no_grad():
        batch = next(iter(test_loader))
        
        # Move batch to device
        batch = {k: v.to(model.device) for k, v in batch.items()}
        
        # Unpack batch
        (past_glucose, past_meal_ids, past_meal_macros,
         future_meal_ids, future_meal_macros, future_glucose, target_scale, encoder_lengths, encoder_padding_mask) = batch
        
        preds = model(
            batch,
            return_attn=True,
            return_meal_self_attn=True,
        )
    logger.info("Example inference complete")
    logger.info(f"Experiment {experiment_name} completed successfully")


if __name__ == "__main__":
    main()