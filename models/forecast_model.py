"""
Main meal-based glucose forecasting model.
"""
from dataclasses import asdict
from typing import List, Optional, Tuple, Dict, Any, Union

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import pandas as pd
import matplotlib.pyplot as plt
import wandb
from loguru import logger
from tqdm import tqdm
from torch.utils.data import DataLoader

from .encoders import MealEncoder, PatchedGlucoseEncoder
from .transformer_blocks import TransformerDecoderLayer, TransformerDecoder
from config import ExperimentConfig
from plot_helpers import plot_meal_self_attention, plot_forecast_examples, plot_iAUC_scatter
from dataset import PPGRToMealGlucoseWrapper
from .losses import quantile_loss, compute_iAUC, unscale_tensor


class MealGlucoseForecastModel(pl.LightningModule):
    """
    PyTorch Lightning model for forecasting glucose levels based on meal data.
    """
    def __init__(
        self, 
        config: ExperimentConfig, 
        num_foods: int, 
        food_macro_dim: int, 
        food_names: List[str], 
        food_group_names: List[str]
    ):
        super().__init__()
        # Save hyperparameters correctly
        self.save_hyperparameters({
            'config': asdict(config),
            'num_foods': num_foods,
            'food_macro_dim': food_macro_dim
        })
        self.config = config
        self.num_foods = num_foods
        self.food_macro_dim = food_macro_dim
        self.food_names = food_names
        self.food_group_names = food_group_names
        self.forecast_horizon = config.prediction_length
        self.eval_window = config.eval_window if config.eval_window is not None else config.prediction_length

        # Model configuration
        self.add_residual_connection_before_predictions = config.add_residual_connection_before_predictions
        self.num_quantiles = config.num_quantiles
        self.loss_iauc_weight = config.loss_iauc_weight

        self.hidden_dim = config.hidden_dim
        self.optimizer_lr = config.optimizer_lr
        self.weight_decay = config.weight_decay
        self.gradient_clip_val = config.gradient_clip_val
        self.food_embedding_projection_batch_size = config.food_embedding_projection_batch_size
        self.disable_plots = config.disable_plots
        
        # Initialize component models
        self._init_meal_encoder(config)
        self._init_glucose_encoder(config)
        self._init_decoder(config)
        
        # Future time query embeddings for each forecast horizon step
        self.future_time_queries = nn.Embedding(config.prediction_length, config.hidden_dim)

        # Final projection: hidden_dim -> num_quantiles
        self.forecast_linear = nn.Linear(config.hidden_dim, config.num_quantiles)

        # Register the quantiles as a buffer
        self.register_buffer("quantiles", torch.linspace(0.05, 0.95, steps=config.num_quantiles))
        
        # Global time-based positional embedding
        max_time = config.min_encoder_length + config.prediction_length + 2000  # a safe upper bound
        self.time_emb = nn.Embedding(max_time, config.hidden_dim)
        
        # Initialize tracking variables for plotting
        self.example_forecasts = None
        self.example_attn_weights_past = None
        self.example_attn_weights_future = None
        self.example_meal_self_attn_past = None
        self.example_meal_self_attn_future = None
        
    def _init_meal_encoder(self, config: ExperimentConfig) -> None:
        """Initialize the meal encoder component."""
        self.meal_encoder = MealEncoder(
            food_embed_dim=config.food_embed_dim,
            hidden_dim=config.hidden_dim,
            num_foods=self.num_foods,
            food_macro_dim=self.food_macro_dim,
            food_names=self.food_names,
            food_group_names=self.food_group_names,
            max_meals=config.max_meals,
            num_heads=config.num_heads,
            num_layers=config.transformer_encoder_layers,
            dropout_rate=config.dropout_rate,
            transformer_dropout=config.transformer_dropout,
            ignore_food_macro_features=config.ignore_food_macro_features,
            bootstrap_food_id_embeddings=None,  # This is initialized in the from_dataset method
            freeze_food_id_embeddings=config.freeze_food_id_embeddings,
            aggregator_type=config.meal_aggregator_type
        )
        
    def _init_glucose_encoder(self, config: ExperimentConfig) -> None:
        """Initialize the glucose encoder component."""
        self.glucose_encoder = PatchedGlucoseEncoder(
            embed_dim=config.hidden_dim,
            patch_size=config.patch_size,
            patch_stride=config.patch_stride,
            num_heads=config.num_heads,
            num_layers=config.transformer_encoder_layers,
            max_seq_len=config.min_encoder_length,
            add_causal_mask=config.add_glucose_causal_mask,
            dropout_rate=config.dropout_rate
        )
        
    def _init_decoder(self, config: ExperimentConfig) -> None:
        """Initialize the transformer decoder component."""
        dec_layer = TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 2,
            dropout=config.transformer_dropout,
            activation="relu"
        )
        self.decoder = TransformerDecoder(
            dec_layer,
            num_layers=config.transformer_decoder_layers,
            norm=nn.LayerNorm(config.hidden_dim)
        )
    
    def forward(
        self,
        past_glucose: torch.Tensor,        # [B, T_glucose]
        past_meal_ids: torch.Tensor,       # [B, T_pastMeal, M]
        past_meal_macros: torch.Tensor,    # [B, T_pastMeal, M, food_macro_dim]
        future_meal_ids: torch.Tensor,     # [B, T_futureMeal, M]
        future_meal_macros: torch.Tensor,  # [B, T_futureMeal, M, food_macro_dim]
        target_scales: torch.Tensor,       # [B, 2] for unscale
        encoder_lengths: torch.Tensor,     # [B]
        encoder_padding_mask: torch.Tensor, # [B, T_pastMeal]
        return_attn: bool = False,
        return_meal_self_attn: bool = False
    ) -> Union[torch.Tensor, Tuple]:
        """
        Forward pass of the model.
        
        Returns either a single tensor [B, T_future, Q] with predictions
        or a tuple if return_attn=True with the following elements:
          (pred_future, past_meal_enc, attn_past, future_meal_enc, attn_future, meal_self_attn_past, meal_self_attn_future)
        """        
        device = past_glucose.device
        B = past_glucose.size(0)
        
        # 1) Encode glucose sequence => shape [B, G_patches, hidden_dim]
        glucose_enc, patch_indices = self.glucose_encoder(past_glucose)
        
        # Add positional embeddings to glucose encodings
        glucose_enc = glucose_enc + self.time_emb(patch_indices)
        
        # 2) Encode past & future meals
        past_meal_enc, meal_self_attn_past = self.meal_encoder(
            past_meal_ids, past_meal_macros, return_self_attn=return_meal_self_attn
        )
        future_meal_enc, meal_self_attn_future = self.meal_encoder(
            future_meal_ids, future_meal_macros, return_self_attn=return_meal_self_attn
        )

        T_past = past_meal_enc.size(1)
        T_future = future_meal_enc.size(1)

        # 3) Add positional time embeddings
        # Create indices for each block, placing them contiguously in time
        past_indices = torch.arange(T_past, device=device).unsqueeze(0).expand(B, -1)
        future_indices = torch.arange(T_future, device=device).unsqueeze(0).expand(B, -1) + T_past

        # Add time embeddings
        past_meal_enc = past_meal_enc + self.time_emb(past_indices)
        future_meal_enc = future_meal_enc + self.time_emb(future_indices)

        # 4) Combine all encodings in a single "memory" sequence
        memory = torch.cat([past_meal_enc, future_meal_enc, glucose_enc], dim=1)

        # 5) Prepare query embeddings for each forecast horizon step
        t_future_indices = torch.arange(self.forecast_horizon, device=device).unsqueeze(0).expand(B, -1)
        t_future_global_indices = t_future_indices + T_past

        # Get query embeddings and add positional encoding
        query_emb = self.future_time_queries(t_future_indices)  # [B, T_future, hidden_dim]
        query_emb = query_emb + self.time_emb(t_future_global_indices)

        # 6) Decoder: Process queries with the memory
        decoder_output, cross_attn = self.decoder(
            tgt=query_emb,
            memory=memory,
            return_attn=return_attn
        )

        # 7) Extract attention weights for past and future if needed
        attn_past = None
        attn_future = None
        if cross_attn is not None:
            past_len = past_meal_enc.shape[1]
            future_len = future_meal_enc.shape[1]
            # Split attention weights for past and future meals
            attn_past = cross_attn[:, :, :past_len]  # => [B, T_future, past_len]
            attn_future = cross_attn[:, :, past_len : past_len + future_len]

        # 8) Final projection to get quantile predictions
        pred_future = self.forecast_linear(decoder_output)

        # Add residual connection if configured
        if self.add_residual_connection_before_predictions:
            last_val = past_glucose[:, -1].unsqueeze(1).unsqueeze(-1)
            pred_future = pred_future + last_val

        # Unscale predictions back to original range
        pred_future = unscale_tensor(pred_future, target_scales)

        # Return appropriate outputs based on return_attn flag
        if return_attn:
            return (
                pred_future,
                past_meal_enc, attn_past,
                future_meal_enc, attn_future,
                meal_self_attn_past, meal_self_attn_future
            )
        else:
            return pred_future
            
    def _shared_step(self, batch, batch_idx, phase: str) -> torch.Tensor:
        """
        Shared step function for training, validation and testing.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
            phase: Phase name ('train', 'val', or 'test')
            
        Returns:
            Loss tensor
        """
        # Unpack batch
        (past_glucose, past_meal_ids, past_meal_macros,
         future_meal_ids, future_meal_macros, future_glucose, target_scales, encoder_lengths, encoder_padding_mask) = batch
         
        # Ensure target_scales has the right shape
        if target_scales.dim() > 2:
            target_scales = target_scales.view(target_scales.size(0), -1)
            
        # Forward pass with attention weights
        preds = self(
            past_glucose,
            past_meal_ids,
            past_meal_macros,
            future_meal_ids,
            future_meal_macros,
            target_scales,
            encoder_lengths,
            encoder_padding_mask,
            return_attn=True,
            return_meal_self_attn=True,
        )
        
        # Compute metrics
        metrics = self._compute_forecast_metrics(past_glucose, future_glucose, target_scales, preds)
        
        # Log metrics with phase prefix
        for key, value in metrics["metrics"].items():
            self.log(f"{phase}_{key}", value, on_step=True, on_epoch=True, prog_bar=True)
        
        # Store outputs for phase end processing
        if not hasattr(self, f"{phase}_outputs"):
            setattr(self, f"{phase}_outputs", [])
            
        getattr(self, f"{phase}_outputs").append({
            f"{phase}_q_loss": metrics["metrics"]["q_loss"],
            f"{phase}_pred_iAUC_{self.eval_window}": metrics[f"pred_iAUC_{self.eval_window}"],
            f"{phase}_true_iAUC_{self.eval_window}": metrics[f"true_iAUC_{self.eval_window}"],
        })
        
        # Store example data for plotting on first batch
        if batch_idx == 0:
            (pred_future, _, attn_past, _, attn_future, meal_self_attn_past, meal_self_attn_future) = preds
            self.example_forecasts = {
                "past": unscale_tensor(past_glucose, target_scales).detach().cpu(),
                "pred": pred_future.detach().cpu(),
                "truth": (future_glucose * target_scales[:, 1].unsqueeze(1) + target_scales[:, 0].unsqueeze(1)).detach().cpu(),
                "future_meal_ids": future_meal_ids.detach().cpu(),
                "past_meal_ids": past_meal_ids.detach().cpu(),
            }
            self.example_attn_weights_past = attn_past
            self.example_attn_weights_future = attn_future
            self.example_meal_self_attn_past = meal_self_attn_past
            self.example_meal_self_attn_future = meal_self_attn_future
        
        return metrics["metrics"]["q_loss"]

    def _shared_phase_end(self, phase: str) -> None:
        """
        Shared epoch end processing for all phases.
        
        Args:
            phase: Phase name ('train', 'val', or 'test')
        """
        outputs = getattr(self, f"{phase}_outputs", [])
        if len(outputs) == 0:
            return

        # Generate plots if not disabled and we have examples
        if (not self.disable_plots) and (self.example_forecasts is not None):
            fixed_indices = getattr(self, "fixed_forecast_indices", None)
            
            # Plot meal self-attention if available
            if hasattr(self, "example_meal_self_attn_past") and hasattr(self, "example_meal_self_attn_future"):
                fig_meal = plot_meal_self_attention(
                    self.example_meal_self_attn_past,
                    self.example_forecasts["past_meal_ids"],
                    self.example_meal_self_attn_future,
                    self.example_forecasts["future_meal_ids"],
                    self.logger,
                    self.global_step,
                    fixed_indices=fixed_indices
                )
                plt.close(fig_meal)            
            
            # Plot forecast examples
            fixed_indices, fig = plot_forecast_examples(
                self.example_forecasts,
                self.example_attn_weights_past,
                self.example_attn_weights_future,
                self.quantiles,
                self.logger,
                self.global_step,
                fixed_indices=fixed_indices
            )
            self.fixed_forecast_indices = fixed_indices
            plt.close(fig)
            self.example_forecasts = None
        
        # Plot iAUC scatter and calculate correlation
        all_pred_iAUC = torch.cat([output[f"{phase}_pred_iAUC_{self.eval_window}"] for output in outputs], dim=0)
        all_true_iAUC = torch.cat([output[f"{phase}_true_iAUC_{self.eval_window}"] for output in outputs], dim=0)
        fig_scatter, corr = plot_iAUC_scatter(all_pred_iAUC, all_true_iAUC)
        self.logger.experiment.log({
            f"{phase}_iAUC_eh{self.eval_window}_scatter": wandb.Image(fig_scatter),
            f"{phase}_iAUC_eh{self.eval_window}_correlation": corr.item(),
        })
        plt.close(fig_scatter)
        
        # Clear outputs for the next epoch
        setattr(self, f"{phase}_outputs", [])

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        return self._shared_step(batch, batch_idx, "val")

    def on_validation_epoch_end(self):
        """End of validation epoch processing."""
        self._shared_phase_end("val")

    def test_step(self, batch, batch_idx):
        """Test step."""
        return self._shared_step(batch, batch_idx, "test")

    def on_test_end(self):
        """End of test processing."""
        self._shared_phase_end("test")
        
    def on_fit_start(self):
        """Hook called at the beginning of fit."""
        if self.config.wandb_log_embeddings:
            self.log_food_embeddings("on_fit_start")

    def on_fit_end(self):
        """Hook called at the end of fit."""
        if self.config.wandb_log_embeddings:
            self.log_food_embeddings("on_fit_end")

    def training_step(self, batch, batch_idx):
        """Training step."""
        (past_glucose, past_meal_ids, past_meal_macros,
         future_meal_ids, future_meal_macros, future_glucose, target_scales, encoder_lengths, encoder_padding_mask) = batch
        
        # Forward pass
        preds = self(
            past_glucose,
            past_meal_ids,
            past_meal_macros,
            future_meal_ids,
            future_meal_macros,
            target_scales,
            encoder_lengths,
            encoder_padding_mask
        )
        
        # Compute metrics
        metrics = self._compute_forecast_metrics(past_glucose, future_glucose, target_scales, preds)
        
        # Log metrics
        for key in metrics["metrics"]:
            self.log(f"train_step_{key}", metrics["metrics"][key], on_step=True, on_epoch=True, prog_bar=True)
            
        return metrics["metrics"]["total_loss"]

    def _compute_forecast_metrics(
        self, 
        past_glucose: torch.Tensor, 
        future_glucose: torch.Tensor, 
        target_scales: torch.Tensor, 
        preds: Union[torch.Tensor, Tuple]
    ) -> Dict[str, Any]:
        """
        Compute metrics for forecasting performance.
        
        Args:
            past_glucose: Past glucose values
            future_glucose: Future glucose values (target)
            target_scales: Scaling factors
            preds: Model predictions (either tensor or tuple with tensor as first element)
            
        Returns:
            Dictionary of metrics
        """
        # Extract predictions if they are part of a tuple
        if isinstance(preds, tuple):
            predictions = preds[0]
        else:
            predictions = preds
            
        # Unscale future glucose values
        future_glucose_unscaled = (
            future_glucose * target_scales[:, 1].unsqueeze(1) + target_scales[:, 0].unsqueeze(1)
        )
        
        # Compute quantile loss
        q_loss = quantile_loss(predictions, future_glucose_unscaled, self.quantiles)
        
        # Compute RMSE with median predictions
        median_idx = self.num_quantiles // 2
        median_pred = predictions[:, :, median_idx]
        median_pred_eval = median_pred[:, :self.eval_window]
        future_glucose_unscaled_eval = future_glucose_unscaled[:, :self.eval_window]
        rmse = torch.sqrt(F.mse_loss(median_pred_eval, future_glucose_unscaled_eval))
        
        # Compute iAUC metrics
        pred_iAUC, true_iAUC = compute_iAUC(
            median_pred, future_glucose, past_glucose, target_scales, eval_window=self.eval_window
        )
        
        # Compute losses
        iAUC_loss = F.mse_loss(pred_iAUC, true_iAUC)
        weighted_iAUC_loss = self.loss_iauc_weight * iAUC_loss
        total_loss = q_loss + weighted_iAUC_loss
        
        return {
            "metrics": {
                "q_loss": q_loss,
                "rmse": rmse,
                f"iAUC_eh{self.eval_window}_loss": iAUC_loss,
                f"iAUC_eh{self.eval_window}_weighted_loss": weighted_iAUC_loss,
                "total_loss": total_loss,
            },
            f"pred_iAUC_{self.eval_window}": pred_iAUC,
            f"true_iAUC_{self.eval_window}": true_iAUC,        
        }

    def log_food_embeddings(self, label: str) -> None:
        """
        Log food embeddings to wandb for visualization.
        
        Args:
            label: Label for the embeddings (e.g., 'on_fit_start', 'on_fit_end')
        """
        logger.info(f"Logging food embeddings for {label}")
        
        # Retrieve the food embeddings from the embedding layer
        food_emb = self.meal_encoder.food_emb.weight  # shape: (num_foods, food_embed_dim)
        
        # Process the embeddings in batches using the configured batch size
        batch_size = self.food_embedding_projection_batch_size
        projected_embeddings = []
        
        with torch.no_grad():
            # Loop over the embeddings in batches
            for i in tqdm(range(0, food_emb.size(0), batch_size), desc="Projecting food embeddings"):
                batch = food_emb[i : i + batch_size]
                # Apply projection
                proj_batch = self.meal_encoder.food_emb_proj(batch)
                projected_embeddings.append(proj_batch.detach().cpu())
        
        # Concatenate all batches and convert to numpy
        projected_embeddings = torch.cat(projected_embeddings, dim=0).numpy()
        embedding_cols = [f"proj_embedding_{i}" for i in range(projected_embeddings.shape[1])]
        
        # Create DataFrame with embeddings and metadata
        food_embeddings_df = pd.DataFrame(projected_embeddings, columns=embedding_cols)
        food_embeddings_df.insert(0, 'food_group_name', self.food_group_names)
        food_embeddings_df.insert(0, 'food_name', self.food_names)
        
        # Log the DataFrame as a wandb Table
        self.logger.experiment.log({f"food_embeddings_{label}": wandb.Table(dataframe=food_embeddings_df)})

    def configure_optimizers(self):
        """Configure optimizers for PyTorch Lightning."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.optimizer_lr,
            weight_decay=self.weight_decay
        )
        
        return optimizer
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, config: ExperimentConfig, num_foods: int, 
                             food_macro_dim: int, food_names: List[str], food_group_names: List[str], **kwargs):
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            config: Model configuration
            num_foods: Number of food items
            food_macro_dim: Dimension of food macro features
            food_names: List of food names
            food_group_names: List of food group names
            **kwargs: Additional arguments to pass to the parent load_from_checkpoint
            
        Returns:
            Loaded model instance
        """
        return super().load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            config=config,
            num_foods=num_foods,
            food_macro_dim=food_macro_dim,
            food_names=food_names,
            food_group_names=food_group_names,
            **kwargs
        )

    @classmethod
    def from_dataset(cls, dataset: PPGRToMealGlucoseWrapper, config: ExperimentConfig):
        """
        Create a model instance from a dataset and config.
        
        Args:
            dataset: Dataset instance containing food metadata
            config: Model configuration
            
        Returns:
            Model instance
        """
        model = cls(
            config=config,
            num_foods=dataset.num_foods,
            food_macro_dim=dataset.num_nutrients,
            food_names=dataset.food_names,
            food_group_names=dataset.food_group_names,
        )
        
        # Bootstrap with pretrained embeddings if configured
        if config.use_bootstraped_food_embeddings:
            pretrained_weights = dataset.get_food_id_embeddings()
            model.meal_encoder._bootstrap_food_id_embeddings(
                pretrained_weights, 
                freeze_embeddings=config.freeze_food_id_embeddings
            )
            
        return model