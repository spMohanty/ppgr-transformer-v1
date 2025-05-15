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
from torchmetrics.functional.regression import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    symmetric_mean_absolute_percentage_error,
)

from .encoders import (
    UserEncoder, MealEncoder, PatchedGlucoseEncoder, 
    TemporalEncoder, GlucosePatcher, PatchEncoder, MicrobiomeEncoder
)
from .transformer_blocks import TransformerDecoderLayer, TransformerDecoder
from config import ExperimentConfig
from plot_helpers import plot_meal_self_attention, plot_forecast_examples, plot_iAUC_scatter
from dataset import PPGRToMealGlucoseWrapper
from .losses import quantile_loss, compute_iAUC, unscale_tensor


class RotaryPositionalEmbeddings(nn.Module):
    """
    Rotary Positional Embeddings (RoPE) as proposed in https://arxiv.org/abs/2104.09864.
    
    This implementation supports offset indices to handle negative positions,
    similar to TimeEmbedding.
    """
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10_000,
        offset: int = 0
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.offset = offset
        self.rope_init()

    def rope_init(self):
        # Precompute frequency bands
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Create position indexes from 0 to max_seq_len-1
        seq_idx = torch.arange(
            max_seq_len, dtype=torch.float, device=self.theta.device
        )

        # Outer product of theta and position index
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta)
        
        # Cache includes both the cos and sin components
        # Shape: [max_seq_len, dim//2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary positional embeddings to input tensor.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            positions: Position indices of shape [batch_size, seq_len]
            
        Returns:
            Tensor with rotary embeddings applied
        """
        # Ensure positions include offset (for negative positions handling)
        positions = positions + self.offset
        
        # Ensure position indices are within bounds
        positions = torch.clamp(positions, 0, self.max_seq_len - 1)
        
        # Get batch size and sequence length
        batch_size, seq_len = positions.shape
        
        # Get cached embeddings for the given positions
        # Shape after indexing: [batch_size, seq_len, dim//2, 2]
        rope_cache = self.cache[positions]
        
        # Reshape input for rotation
        # From [batch_size, seq_len, dim] to [batch_size, seq_len, dim//2, 2]
        x_reshaped = x.float().reshape(batch_size, seq_len, -1, 2)
        
        # Apply rotary transformation using the cached values
        # cos(θ)x - sin(θ)y and sin(θ)x + cos(θ)y
        x_out = torch.stack(
            [
                x_reshaped[..., 0] * rope_cache[..., 0] - 
                x_reshaped[..., 1] * rope_cache[..., 1],
                
                x_reshaped[..., 1] * rope_cache[..., 0] + 
                x_reshaped[..., 0] * rope_cache[..., 1],
            ],
            dim=-1,
        )
        
        # Reshape back to original shape - ensure this matches the input shape
        x_out = x_out.reshape(batch_size, seq_len, self.dim)
        
        # Return with the same dtype as the input
        return x_out.type_as(x)

class MealGlucoseForecastModel(pl.LightningModule):
    """
    PyTorch Lightning model for forecasting glucose levels based on meal data.
    """
    def __init__(
        self, 
        config: ExperimentConfig, 
        dataset_metadata: Dict[str, Any],
        num_foods: int, 
        food_macro_dim: int, 
        food_names: List[str], 
        food_group_names: List[str]
    ):
        super().__init__()
        # Save hyperparameters correctly
        self.save_hyperparameters({
            'config': asdict(config),
            'dataset_metadata': dataset_metadata,
            'num_foods': num_foods,
            'food_macro_dim': food_macro_dim
        })
        self.config = config
        self.dataset_metadata = dataset_metadata
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
        self._init_positional_embeddings(config)
        self._init_user_encoder(config, dataset_metadata)
        self._init_temporal_encoder(config, dataset_metadata)
        self._init_meal_encoder(config)
        self._init_glucose_encoder(config)
        self._init_decoder(config)
        
        
        # Future time query embeddings for each forecast horizon step
        self.future_time_queries = nn.Embedding(config.prediction_length + 1, config.hidden_dim)
        # NOTE: we are adding 1 to the prediction length, as we want to stay consistent and use 1-indexed values for future time points
        # So basically, the embedding for 0 is not really used here. todo: see if refactoring this makes sense.

        # Final projection: hidden_dim -> num_quantiles
        self.forecast_linear = nn.Linear(config.hidden_dim, config.num_quantiles)

        # Register the quantiles as a buffer
        self.register_buffer("quantiles", torch.linspace(0.05, 0.95, steps=config.num_quantiles))
        
        # Initialize tracking variables for plotting
        self.example_forecasts = None
        self.example_attn_weights_past = None
        self.example_attn_weights_future = None
        self.example_meal_self_attn_past = None
        self.example_meal_self_attn_future = None
                
        # Create learnable type embeddings for each sequence type
        self.type_embeddings = nn.Embedding(5, config.hidden_dim)  # 4 types: past_meals, future_meals, glucose, user, microbiome
        
    def _init_positional_embeddings(self, config: ExperimentConfig) -> None:
        """Initialize the positional embeddings."""
        max_encoder_length = config.max_encoder_length
        max_prediction_length = config.prediction_length
        total_range = max_encoder_length + max_prediction_length + 100  # a safe upper bound
                
        # Use rotary position embeddings
        # The offset is max_encoder_length to center positions around the last valid position
        self.rope_emb = RotaryPositionalEmbeddings(
            dim=config.hidden_dim,
            max_seq_len=total_range,
            base=10000,
            offset=max_encoder_length 
        )
        
    def _init_user_encoder(self, config: ExperimentConfig, dataset_metadata: Dict[str, Any]) -> None:
        """Initialize the user encoder component."""        
        # Get sizes of categorical variables        
        self.user_static_categoricals = dataset_metadata["user_static_categoricals"]
        self.user_static_reals = dataset_metadata["user_static_reals"]
        microbiome_embeddings_dim = dataset_metadata["microbiome_embeddings_dim"]

        categorical_variable_sizes = {}
        for cat in self.user_static_categoricals:
            categorical_variable_sizes[cat] = len(dataset_metadata["categorical_encoders"][cat].classes_)
        
        self.user_encoder = UserEncoder(
            categorical_variable_sizes=categorical_variable_sizes,
            real_variables=self.user_static_reals,
            hidden_dim=config.hidden_dim,
            dropout_rate=config.dropout_rate,
            project_to_single_vector=config.project_user_features_to_single_vector,
        )
        
        # Initialize microbiome encoder
        self.microbiome_encoder = MicrobiomeEncoder(
            input_dim=microbiome_embeddings_dim,
            hidden_dim=config.hidden_dim * 2,
            output_dim=config.hidden_dim,
            dropout_rate=config.dropout_rate,
            use_batch_norm=True,
        )
        

    def _init_temporal_encoder(self, config: ExperimentConfig, dataset_metadata: Dict[str, Any]) -> None:
        """Initialize the temporal encoder component."""
        # Get sizes of temporal categorical variables
        temporal_categoricals = dataset_metadata.get("temporal_categoricals", [])
        temporal_reals = dataset_metadata.get("temporal_reals", [])
        
        categorical_variable_sizes = {}
        for cat in temporal_categoricals:
            categorical_variable_sizes[cat] = len(dataset_metadata["categorical_encoders"][cat].classes_)
        
        self.temporal_encoder = TemporalEncoder(
            categorical_variable_sizes=categorical_variable_sizes,
            real_variables=temporal_reals,
            hidden_dim=config.hidden_dim,
            dropout_rate=config.dropout_rate,
            use_batch_norm=True,
            positional_embedding=self.rope_emb  # Pass the positional embedding
        )
        

    def _init_meal_encoder(self, config: ExperimentConfig) -> None:
        """Initialize the meal encoder component."""
        self.meal_encoder = MealEncoder(
            food_embed_dim=config.food_embed_dim,
            hidden_dim=config.hidden_dim,
            num_foods=self.num_foods,
            food_macro_dim=self.food_macro_dim,
            food_names=self.food_names,
            food_group_names=self.food_group_names,
            positional_embedding=self.rope_emb,
            max_meals=config.max_meals,
            num_heads=config.num_heads,
            num_layers=config.transformer_encoder_layers,
            layers_share_weights=config.transformer_encoder_layers_share_weights,
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
            positional_embedding=self.rope_emb,
            num_heads=config.num_heads,
            num_layers=config.transformer_encoder_layers,
            layers_share_weights=config.transformer_encoder_layers_share_weights,
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
            layers_share_weights=config.transformer_decoder_layers_share_weights,
            norm=nn.LayerNorm(config.hidden_dim)
        )
    
    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        return_attn: bool = False,
        return_meal_self_attn: bool = False
    ) -> Union[torch.Tensor, Tuple]:
        """
        Forward pass of the model.
        
        Returns either a single tensor [B, T_future, Q] with predictions
        or a tuple if return_attn=True with the following elements:
          (pred_future, past_meal_enc, attn_past, future_meal_enc, attn_future, meal_self_attn_past, meal_self_attn_future)
        """        
        # Extract parts of batch
        # Unpack batch
        user_categoricals = batch["user_categoricals"] # [B, num_user_cat_features]
        user_reals = batch["user_reals"]               # [B, num_user_real_features]
        user_microbiome_embeddings = batch["user_microbiome_embeddings"] # [B, num_microbiome_features]
        past_glucose = batch["past_glucose"]         # [B, T_past]
        past_meal_ids = batch["past_meal_ids"]       # [B, T_past, M]
        past_meal_macros = batch["past_meal_macros"] # [B, T_past, M, food_macro_dim]
        future_meal_ids = batch["future_meal_ids"]   # [B, T_future, M]
        future_meal_macros = batch["future_meal_macros"] # [B, T_future, M, food_macro_dim]
        future_glucose = batch["future_glucose"]     # [B, T_future]
        target_scales = batch["target_scales"]       # [B, 2]
        encoder_lengths = batch["encoder_lengths"]   # [B]
        encoder_padding_mask = batch["encoder_padding_mask"] # [B, T_past]
        metadata = batch["metadata"] # Dict with metadata
        
        device = past_glucose.device
        B = past_glucose.size(0)
        
        # Get user embeddings - now passing the entire batch to the encoder
        user_embeddings = self.user_encoder(user_categoricals, user_reals) # [B, num_user_{cat+real}_features, hidden_dim]
                
        # Handle microbiome data
        microbiome_embeddings = self.microbiome_encoder(batch.get("user_microbiome_embeddings"))
        microbiome_embeddings = microbiome_embeddings.unsqueeze(1)  # [B, 1, hidden_dim]
        
        # Get sequence lengths for positional encoding
        T_past = past_meal_ids.size(1) if past_meal_ids is not None else 0
        T_future = future_meal_ids.size(1) if future_meal_ids is not None else 0
        max_encoder_length = self.config.max_encoder_length
        
        # 1) Generate time indices for positional embeddings FIRST
        # Regular time indices
        past_indices = torch.arange(T_past, device=device).unsqueeze(0).expand(B, -1)
        future_indices = torch.arange(T_future, device=device).unsqueeze(0).expand(B, -1) + T_past
        
        # Center indices by subtracting (max_encoder_length - 1) for each batch item
        # This makes the last valid position have index 0
        batch_max_encoder_length = past_glucose.shape[-1] 
        offset_value = batch_max_encoder_length - 1
        centered_offset = offset_value * torch.ones((B, 1), device=device, dtype=torch.long)
        
        # Apply centering to all indices
        past_indices = past_indices - centered_offset
        future_indices = future_indices - centered_offset  # Future will start at index 1
        
        # 2) Generate glucose patch position indices and center them
        # This time don't create estimated patch indices, we'll use the actual ones
                
        # 3) Encode glucose sequence with positional information
        glucose_enc, glucose_attn_weights, patch_indices = self.glucose_encoder(
            past_glucose, 
            positions=past_indices,  # Pass past_indices for consistency
            mask=encoder_padding_mask, 
            return_self_attn=False
        )
        
        # Center the patch indices using the same offset as other sequences
        # patch_indices now already contains the END position of each patch
        offset_tensor = torch.tensor([offset_value], device=patch_indices.device)
        centered_patch_indices = patch_indices - offset_tensor
                
        # Encode temporal information
        past_temporal_emb = self.temporal_encoder(
            temporal_categoricals=batch["past_temporal_categoricals"],
            temporal_reals=batch["past_temporal_reals"],
            positions=past_indices,  # Use the same positions as for meals
            mask=encoder_padding_mask
        )
        future_temporal_emb = self.temporal_encoder(
            temporal_categoricals=batch["future_temporal_categoricals"],
            temporal_reals=batch["future_temporal_reals"],
            positions=future_indices,  # Use the same positions as for future meals
            mask=None  # Future usually doesn't have a mask
        )
                
        
        # 4) Encode past & future meals with positional information
        past_meal_enc, meal_self_attn_past = self.meal_encoder(
            past_meal_ids, past_meal_macros, 
            positions=past_indices,  # Pass past position indices
            mask=encoder_padding_mask, 
            return_self_attn=return_meal_self_attn
        )
        future_meal_enc, meal_self_attn_future = self.meal_encoder(
            future_meal_ids, future_meal_macros, 
            positions=future_indices,  # Pass future position indices
            return_self_attn=return_meal_self_attn
        )
        
        # Get number of glucose patches for further processing
        N_patches = glucose_enc.size(1)
        
        
        # Add temporal embeddings to the meal + glucose encodings
        past_meal_enc = past_meal_enc + past_temporal_emb  # [B, T_past, hidden_dim]
        future_meal_enc = future_meal_enc + future_temporal_emb  # [B, T_future, hidden_dim]
        glucose_enc = glucose_enc + past_temporal_emb[:, centered_patch_indices, :]  # [B, N_patches, hidden_dim]
        
        # Create type IDs for each sequence
        past_meal_type_id = torch.zeros((B, T_past), device=device, dtype=torch.long)
        future_meal_type_id = torch.ones((B, T_future), device=device, dtype=torch.long)
        glucose_type_id = 2 * torch.ones((B, N_patches), device=device, dtype=torch.long)
        user_type_id = 3 * torch.ones((B, user_embeddings.shape[1]), device=device, dtype=torch.long)
        microbiome_type_id = 4 * torch.ones((B, 1), device=device, dtype=torch.long)
        
        # Unqueeze user_embeddings to match the shape of past_meal_enc
        # user_embeddings = user_embeddings.unsqueeze(1).expand(B, 1, -1)
                
        # Add type embeddings to each sequence
        past_meal_enc = past_meal_enc + self.type_embeddings(past_meal_type_id)
        future_meal_enc = future_meal_enc + self.type_embeddings(future_meal_type_id)
        glucose_enc = glucose_enc + self.type_embeddings(glucose_type_id)
        user_embeddings = user_embeddings + self.type_embeddings(user_type_id)
        microbiome_embeddings = microbiome_embeddings + self.type_embeddings(microbiome_type_id)
        
        
        # 5) Combine all encodings in a single "memory" sequence
        memory = [
            past_meal_enc,
            future_meal_enc,
            glucose_enc,
            user_embeddings,
            microbiome_embeddings
        ]
        memory = torch.cat(memory, dim=1)

        # 6) Prepare query embeddings for each forecast horizon step
        # Make query indices start at 1 (first future position)
        query_indices = torch.arange(self.forecast_horizon, device=device).unsqueeze(0).expand(B, -1) + 1
        # NOTE: query_indices also start at 1 (and is same as future_indices)
        
        # Get query embeddings and add positional encoding
        query_emb = self.future_time_queries(query_indices)  # [B, T_future, hidden_dim]
        query_emb = self.rope_emb(query_emb, query_indices)

        # 7) Decoder: Process queries with the memory
        decoder_output, cross_attn = self.decoder(
            tgt=query_emb,
            memory=memory,
            return_attn=return_attn
        )

        # 8) Extract attention weights for past and future if needed
        attn_past = None
        attn_future = None
        if cross_attn is not None:
            past_len = past_meal_enc.shape[1]
            future_len = future_meal_enc.shape[1]
            # Split attention weights for past and future meals
            attn_past = cross_attn[:, :, :past_len]  # => [B, T_future, past_len]
            attn_future = cross_attn[:, :, past_len : past_len + future_len]

        # Force full precision for the residual connection
        with torch.amp.autocast("cuda", enabled=False):
            # 9) Final projection to get quantile predictions
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
        user_categoricals = batch["user_categoricals"]
        user_reals = batch["user_reals"]
        user_microbiome_embeddings = batch["user_microbiome_embeddings"]        
        past_temporal_categoricals = batch["past_temporal_categoricals"]
        past_temporal_reals = batch["past_temporal_reals"]
        future_temporal_categoricals = batch["future_temporal_categoricals"]
        future_temporal_reals = batch["future_temporal_reals"]
        past_glucose = batch["past_glucose"]
        past_meal_ids = batch["past_meal_ids"]
        past_meal_macros = batch["past_meal_macros"]
        future_meal_ids = batch["future_meal_ids"]
        future_meal_macros = batch["future_meal_macros"]
        future_glucose = batch["future_glucose"]
        target_scales = batch["target_scales"]
        encoder_lengths = batch["encoder_lengths"]
        encoder_padding_mask = batch["encoder_padding_mask"]
        metadata = batch["metadata"]
        # Ensure target_scales has the right shape
        if target_scales.dim() > 2:
            target_scales = target_scales.view(target_scales.size(0), -1)
            
        # Forward pass with attention weights
        preds = self(batch,
                    return_attn=True,
                    return_meal_self_attn=True)
        
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
                plotted_indices = plot_meal_self_attention(
                    self.example_meal_self_attn_past,
                    self.example_forecasts["past_meal_ids"],
                    self.example_meal_self_attn_future,
                    self.example_forecasts["future_meal_ids"],
                    self.logger,
                    self.global_step,
                    fixed_indices=fixed_indices
                )
            
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
        fig_scatter, corr = plot_iAUC_scatter(all_pred_iAUC, all_true_iAUC, self.config.disable_plots)
        if fig_scatter is not None:
            self.logger.experiment.log({
                f"{phase}_iAUC_eh{self.eval_window}_scatter": wandb.Image(fig_scatter),
            })
            plt.close(fig_scatter)
        
        # Log metrics
        self.logger.experiment.log({f"{phase}_iAUC_eh{self.eval_window}_correlation": corr.item()})
        # Also log correlation to Lightning metrics
        self.log(f"{phase}_iAUC_eh{self.eval_window}_correlation", corr.item(), on_step=False, on_epoch=True, prog_bar=True)
        
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

    def on_test_epoch_end(self):
        """End of test epoch processing."""
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
        # Unpack batch
        user_categoricals = batch["user_categoricals"]
        user_reals = batch["user_reals"]
        user_microbiome_embeddings = batch["user_microbiome_embeddings"]
        past_temporal_categoricals = batch["past_temporal_categoricals"]
        past_temporal_reals = batch["past_temporal_reals"]
        future_temporal_categoricals = batch["future_temporal_categoricals"]
        future_temporal_reals = batch["future_temporal_reals"]
        past_glucose = batch["past_glucose"]         
        past_meal_ids = batch["past_meal_ids"]
        past_meal_macros = batch["past_meal_macros"]
        future_meal_ids = batch["future_meal_ids"]
        future_meal_macros = batch["future_meal_macros"]
        future_glucose = batch["future_glucose"]
        target_scales = batch["target_scales"]
        encoder_lengths = batch["encoder_lengths"]
        encoder_padding_mask = batch["encoder_padding_mask"]        
        metadata = batch["metadata"]
        # Forward pass
        preds = self(batch)
        
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
        
        # Flatten predictions and targets to 1D contiguous tensors to avoid view errors in torchmetrics
        pred_flat = median_pred_eval.contiguous().reshape(-1)
        target_flat = future_glucose_unscaled_eval.contiguous().reshape(-1)
        rmse = mean_squared_error(pred_flat, target_flat, squared=False)
        mae = mean_absolute_error(pred_flat, target_flat)
        mape = mean_absolute_percentage_error(pred_flat, target_flat)
        smape = symmetric_mean_absolute_percentage_error(pred_flat, target_flat)
        
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
                "mae": mae,
                "mape": mape,
                "smape": smape,
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
        dataset_metadata = dataset[0]["metadata"] # metadata is the last element in the tuple - todo: change this to dict 
                
        model = cls(
            config=config,
            dataset_metadata=dataset_metadata,
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