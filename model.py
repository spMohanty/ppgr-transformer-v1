#!/usr/bin/env python
import os
import random
import pickle
import hashlib
import logging
import warnings
import math

from dataclasses import dataclass

# Suppress the nested tensors prototype warning from PyTorch.
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="The PyTorch API of nested tensors is in prototype stage"
)

import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import RichModelSummary, RichProgressBar
from pytorch_lightning.loggers import WandbLogger
import wandb

from dataset import create_cached_dataset
from utils import unscale_tensor

# -----------------------------------------------------------------------------
# Experiment Configuration
# -----------------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    # Dataset / caching settings
    dataset_version: str = "v0.4"
    cache_dir: str = "/scratch/mohanty/food/ppgr-v1/datasets-cache"
    use_cache: bool = True
    debug_mode: bool = False
    dataloader_num_workers: int = 7  # Added configurable dataloader_num_workers parameter

    # Data splitting & sequence parameters
    min_encoder_length: int = 8 * 4    # e.g., 8 * 4
    prediction_length: int = 8 * 4     # e.g., 8 * 4
    eval_window: int = 2 * 4            # e.g., 2 * 4
    validation_percentage: float = 0.1
    test_percentage: float = 0.1

    # Data options
    is_food_anchored: bool = True
    sliding_window_stride: int = None
    use_meal_level_food_covariates: bool = True
    use_bootstraped_food_embeddings: bool = True
    use_microbiome_embeddings: bool = True
    group_by_columns: list = None

    # Feature lists (users, food, temporal)
    user_static_categoricals: list = None
    user_static_reals: list = None
    food_categoricals: list = None
    food_reals: list = None
    temporal_categoricals: list = None
    temporal_reals: list = None
    targets: list = None

    # Model hyperparameters
    food_embed_dim: int = 64
    hidden_dim: int = 256
    num_heads: int = 4
    enc_layers: int = 2
    residual_pred: bool = True
    num_quantiles: int = 7
    loss_iAUC_weight: float = 0.00

    # New dropout hyperparameters
    dropout_rate: float = 0.1          # Used for projections, cross-attention, forecast MLP, etc.
    transformer_dropout: float = 0.1   # Used within Transformer layers

    # Training hyperparameters
    batch_size: int = 1024 * 2
    max_epochs: int = 5
    optimizer_lr: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip_val: float = 0.1  # Added gradient clipping parameter

    # WandB logging
    wandb_project: str = "meal-representations-learning-v0"
    wandb_run_name: str = "MealGlucoseForecastModel_Run"

    # Precision
    precision: str = "bf16"

    def __post_init__(self):
        # Set default lists if not provided.
        if self.group_by_columns is None:
            self.group_by_columns = ["timeseries_block_id"]
        if self.user_static_categoricals is None:
            self.user_static_categoricals = [
                "user_id", "user__edu_degree", "user__income",
                "user__household_desc", "user__job_status", "user__smoking",
                "user__health_state", "user__physical_activities_frequency",
            ]
        if self.user_static_reals is None:
            self.user_static_reals = [
                "user__age", "user__weight", "user__height",
                "user__bmi", "user__general_hunger_level",
                "user__morning_hunger_level", "user__mid_hunger_level",
                "user__evening_hunger_level",
            ]
        if self.use_meal_level_food_covariates:
            self.food_categoricals = ["food__food_group_cname", "food_id"]
        else:
            self.food_categoricals = [
                "food__vegetables_fruits",
                "food__grains_potatoes_pulses",
                "food__unclassified",
                "food__non_alcoholic_beverages",
                "food__dairy_products_meat_fish_eggs_tofu",
                "food__sweets_salty_snacks_alcohol",
                "food__oils_fats_nuts",
            ]
        if self.food_reals is None:
            self.food_reals = [
                "food__eaten_quantity_in_gram", "food__energy_kcal_eaten",
                "food__carb_eaten", "food__fat_eaten",
                "food__protein_eaten", "food__fiber_eaten",
                "food__alcohol_eaten",
            ]
        if self.temporal_categoricals is None:
            self.temporal_categoricals = ["loc_eaten_dow", "loc_eaten_dow_type", "loc_eaten_season"]
        if self.temporal_reals is None:
            self.temporal_reals = ["loc_eaten_hour"]
        if self.targets is None:
            self.targets = ["val"]

# -----------------------------------------------------------------------------
# Custom Transformer Encoder Classes (to return self-attention)
# -----------------------------------------------------------------------------
class TransformerEncoderLayerWithAttn(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayerWithAttn, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)  # dropout after activation
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, src, src_mask=None, src_key_padding_mask=None, return_attn=False):
        attn_output, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True
        )
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        if return_attn:
            return src, attn_weights
        else:
            return src

class TransformerEncoderWithAttn(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoderWithAttn, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None, return_attn=False):
        output = src
        attn_weights = None
        for mod in self.layers:
            if return_attn:
                output, attn_weights = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, return_attn=True)
            else:
                output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        if return_attn:
            return output, attn_weights
        return output

# -----------------------------------------------------------------------------
# Modified Meal Encoder (dropout configurable via parameters)
# -----------------------------------------------------------------------------
class MealEncoder(nn.Module):
    def __init__(
        self,
        food_embed_dim: int,
        hidden_dim: int,
        num_foods: int,
        macro_dim: int,
        max_meals: int = 11,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout_rate: float = 0.2,
        transformer_dropout: float = 0.1,
    ):
        super(MealEncoder, self).__init__()
        self.food_embed_dim = food_embed_dim
        self.hidden_dim = hidden_dim
        self.max_meals = max_meals
        self.num_foods = num_foods
        self.macro_dim = macro_dim

        self.food_emb = nn.Embedding(num_foods, food_embed_dim, padding_idx=0)
        self.food_emb_proj = nn.Linear(food_embed_dim, hidden_dim)
        self.macro_proj = nn.Linear(macro_dim, hidden_dim, bias=False)
        self.pos_emb = nn.Embedding(max_meals, hidden_dim)
        self.start_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # Use the configurable dropout rate
        self.dropout = nn.Dropout(dropout_rate)

        encoder_layer = TransformerEncoderLayerWithAttn(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=transformer_dropout,  # configurable dropout in transformer layer
            activation="relu"
        )
        norm = nn.LayerNorm(hidden_dim)
        self.encoder = TransformerEncoderWithAttn(encoder_layer, num_layers=num_layers, norm=norm)

    def forward(self, meal_ids: torch.LongTensor, meal_macros: torch.Tensor, return_self_attn: bool = False):
        B, T, M = meal_ids.size()
        meal_ids_flat = meal_ids.view(B * T, M)
        meal_macros_flat = meal_macros.view(B * T, M, -1)

        food_emb = self.food_emb(meal_ids_flat)
        food_emb = self.food_emb_proj(food_emb)
        food_emb = self.dropout(food_emb)  # dropout on food embeddings

        macro_emb = self.macro_proj(meal_macros_flat)
        macro_emb = self.dropout(macro_emb)  # dropout on macro embeddings

        meal_token_emb = food_emb + macro_emb

        pos_indices = torch.arange(self.max_meals, device=meal_ids.device)
        pos_enc = self.pos_emb(pos_indices).unsqueeze(0)
        meal_token_emb = meal_token_emb + pos_enc
        meal_token_emb = self.dropout(meal_token_emb)  # dropout after adding positional encoding

        start_token_expanded = self.start_token.expand(B * T, -1, -1)
        meal_token_emb = torch.cat([start_token_expanded, meal_token_emb], dim=1)

        pad_mask = meal_ids_flat == 0
        pad_mask = torch.cat(
            [torch.zeros(B * T, 1, device=pad_mask.device, dtype=torch.bool), pad_mask],
            dim=1,
        )
        if return_self_attn:
            meal_attn_out, self_attn = self.encoder(meal_token_emb, src_key_padding_mask=pad_mask, return_attn=True)
        else:
            meal_attn_out = self.encoder(meal_token_emb, src_key_padding_mask=pad_mask)
        # Use the start token output as the meal embedding.
        meal_timestep_emb = meal_attn_out[:, 0, :]
        meal_timestep_emb = meal_timestep_emb.view(B, T, self.hidden_dim)
        if return_self_attn:
            num_tokens = meal_attn_out.size(1)
            self_attn = self_attn.view(B, T, num_tokens, num_tokens)
            return meal_timestep_emb, self_attn
        return meal_timestep_emb

# -----------------------------------------------------------------------------
# Glucose Encoder (dropout configurable)
# -----------------------------------------------------------------------------
class GlucoseEncoder(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int = 4, num_layers: int = 1, max_seq_len: int = 100, dropout_rate: float = 0.2
    ):
        super(GlucoseEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.glucose_proj = nn.Linear(1, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.1,  # you can also use transformer_dropout if desired
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, glucose_seq: torch.Tensor) -> torch.Tensor:
        glucose_seq = glucose_seq.to(next(self.parameters()).dtype)
        B, T = glucose_seq.size()
        x = glucose_seq.unsqueeze(-1)
        x = self.glucose_proj(x)
        x = self.dropout(x)  # dropout after projection
        pos_indices = torch.arange(T, device=glucose_seq.device)
        pos_enc = self.pos_emb(pos_indices).unsqueeze(0)
        x = x + pos_enc
        x = self.encoder(x)
        return x

# -----------------------------------------------------------------------------
# MealGlucoseForecastModel with configurable dropout parameters
# -----------------------------------------------------------------------------
class MealGlucoseForecastModel(pl.LightningModule):
    def __init__(
        self,
        food_embed_dim: int,
        hidden_dim: int,
        num_foods: int,
        macro_dim: int,
        max_meals: int = 11,
        glucose_seq_len: int = 20,
        forecast_horizon: int = 4,
        eval_window: int = None,
        num_heads: int = 4,
        enc_layers: int = 1,
        residual_pred: bool = True,
        num_quantiles: int = 7,
        loss_iAUC_weight: float = 1,
        dropout_rate: float = 0.2,
        transformer_dropout: float = 0.1,
    ):
        super(MealGlucoseForecastModel, self).__init__()
        self.forecast_horizon = forecast_horizon
        self.eval_window = eval_window if eval_window is not None else forecast_horizon

        self.food_embed_dim = food_embed_dim
        self.hidden_dim = hidden_dim
        self.max_meals = max_meals
        self.num_foods = num_foods
        self.macro_dim = macro_dim
        self.residual_pred = residual_pred
        self.num_quantiles = num_quantiles
        self.loss_iAUC_weight = loss_iAUC_weight

        # Encoders with configurable dropout
        self.meal_encoder = MealEncoder(
            food_embed_dim, hidden_dim, num_foods, macro_dim, max_meals,
            num_heads, enc_layers, dropout_rate=dropout_rate, transformer_dropout=transformer_dropout
        )
        self.glucose_encoder = GlucoseEncoder(
            hidden_dim, num_heads, enc_layers, glucose_seq_len, dropout_rate=dropout_rate
        )

        # Cross-Attention Modules with dropout set from transformer_dropout
        self.cross_attn_past = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=transformer_dropout, batch_first=True
        )
        self.cross_attn_future = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=transformer_dropout, batch_first=True
        )

        # Dropout layer applied after cross-attention outputs and in the forecast MLP
        self.dropout = nn.Dropout(dropout_rate)

        # Forecast MLP with dropout between layers
        self.forecast_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, forecast_horizon * num_quantiles)
        )

        self.register_buffer("quantiles", torch.linspace(0.05, 0.95, steps=self.num_quantiles))
        self.last_attn_weights = None
        self.example_forecasts = None

    def forward(
        self,
        past_glucose,
        past_meal_ids,
        past_meal_macros,
        future_meal_ids,
        future_meal_macros,
        target_scales,
        return_attn: bool = False,
        return_meal_self_attn: bool = False,
    ):
        logging.debug("MealGlucoseForecastModel: --- Forward Pass Start ---")
        glucose_enc = self.glucose_encoder(past_glucose)  # [B, T_glucose, hidden_dim]

        if return_meal_self_attn:
            past_meal_enc, meal_self_attn_past = self.meal_encoder(past_meal_ids, past_meal_macros, return_self_attn=True)
            future_meal_enc, meal_self_attn_future = self.meal_encoder(future_meal_ids, future_meal_macros, return_self_attn=True)
        else:
            past_meal_enc = self.meal_encoder(past_meal_ids, past_meal_macros)
            future_meal_enc = self.meal_encoder(future_meal_ids, future_meal_macros)

        attn_output_past, attn_weights_past = self.cross_attn_past(
            query=glucose_enc, key=past_meal_enc, value=past_meal_enc, need_weights=True
        )
        attn_output_past = self.dropout(attn_output_past)  # configurable dropout
        attn_output_future, attn_weights_future = self.cross_attn_future(
            query=glucose_enc, key=future_meal_enc, value=future_meal_enc, need_weights=True
        )
        attn_output_future = self.dropout(attn_output_future)
        combined_glucose = glucose_enc + attn_output_past + attn_output_future
        self.last_attn_weights = attn_weights_future

        final_rep = torch.cat(
            [combined_glucose[:, -1, :], future_meal_enc[:, -1, :]],
            dim=-1
        )
        with torch.amp.autocast("cuda", enabled=False):
            final_rep_fp32 = final_rep.float()
            pred_future = self.forecast_mlp(final_rep_fp32)
            pred_future = pred_future.view(final_rep.size(0), self.forecast_horizon, self.num_quantiles)
            if self.residual_pred:
                last_val = past_glucose[:, -1].unsqueeze(1).unsqueeze(-1).float()
                pred_future = pred_future + last_val
        pred_future = unscale_tensor(pred_future, target_scales)
        logging.debug("MealGlucoseForecastModel: --- Forward Pass End ---")
        if return_attn:
            if return_meal_self_attn:
                return (pred_future, past_meal_enc, attn_weights_past, future_meal_enc, attn_weights_future,
                        meal_self_attn_past, meal_self_attn_future)
            return (pred_future, past_meal_enc, attn_weights_past, future_meal_enc, attn_weights_future)
        return pred_future

    def _compute_forecast_metrics(self, past_glucose, future_glucose, target_scales, preds):
        if isinstance(preds, tuple):
            predictions = preds[0]
        else:
            predictions = preds
        future_glucose_unscaled = (
            future_glucose * target_scales[:, 1].unsqueeze(1) + target_scales[:, 0].unsqueeze(1)
        )
        q_loss = quantile_loss(predictions, future_glucose_unscaled, self.quantiles)
        median_idx = self.num_quantiles // 2
        median_pred = predictions[:, :, median_idx]
        median_pred_eval = median_pred[:, :self.eval_window]
        future_glucose_unscaled_eval = future_glucose_unscaled[:, :self.eval_window]
        rmse = torch.sqrt(F.mse_loss(median_pred_eval, future_glucose_unscaled_eval))
        pred_iAUC, true_iAUC = compute_iAUC(
            median_pred, future_glucose, past_glucose, target_scales, eval_window=self.eval_window
        )
        iAUC_loss = F.mse_loss(pred_iAUC, true_iAUC)
        weighted_iAUC_loss = self.loss_iAUC_weight * iAUC_loss
        total_loss = q_loss + weighted_iAUC_loss
        return {
            "q_loss": q_loss,
            "rmse": rmse,
            "pred_iAUC": pred_iAUC,
            "true_iAUC": true_iAUC,
            "iAUC_loss": weighted_iAUC_loss,
            "total_loss": total_loss,
        }

    def training_step(self, batch, batch_idx):
        (past_glucose, past_meal_ids, past_meal_macros,
         future_meal_ids, future_meal_macros, future_glucose, target_scales) = batch
        if target_scales.dim() > 2:
            target_scales = target_scales.view(target_scales.size(0), -1)
        preds = self(
            past_glucose,
            past_meal_ids,
            past_meal_macros,
            future_meal_ids,
            future_meal_macros,
            target_scales,
            return_attn=False,
            return_meal_self_attn=False,
        )
        metrics = self._compute_forecast_metrics(past_glucose, future_glucose, target_scales, preds)
        self.log("train_quantile_loss", metrics["q_loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_rmse", metrics["rmse"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_iAUC_loss", metrics["iAUC_loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_total_loss", metrics["total_loss"], on_step=True, on_epoch=True, prog_bar=True)
        return metrics["total_loss"]

    def validation_step(self, batch, batch_idx):
        (past_glucose, past_meal_ids, past_meal_macros,
         future_meal_ids, future_meal_macros, future_glucose, target_scales) = batch
        if target_scales.dim() > 2:
            target_scales = target_scales.view(target_scales.size(0), -1)
        preds = self(
            past_glucose,
            past_meal_ids,
            past_meal_macros,
            future_meal_ids,
            future_meal_macros,
            target_scales,
            return_attn=True,
            return_meal_self_attn=True,
        )
        metrics = self._compute_forecast_metrics(past_glucose, future_glucose, target_scales, preds)
        self.log("val_quantile_loss", metrics["q_loss"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_rmse", metrics["rmse"], on_step=False, on_epoch=True, prog_bar=True)
        if not hasattr(self, "val_outputs"):
            self.val_outputs = []
        self.val_outputs.append({
            "val_quantile_loss": metrics["q_loss"],
            "pred_iAUC": metrics["pred_iAUC"],
            "true_iAUC": metrics["true_iAUC"],
        })
        if batch_idx == 0:
            (pred_future, _, attn_past, _, attn_future, meal_self_attn_past, meal_self_attn_future) = preds
            self.example_forecasts = {
                "past": unscale_tensor(past_glucose, target_scales).detach().cpu(),
                "pred": pred_future.detach().cpu(),
                "truth": (future_glucose * target_scales[:, 1].unsqueeze(1) + target_scales[:, 0].unsqueeze(1)).detach().cpu(),
                "future_meal_ids": future_meal_ids.detach().cpu(),
                "past_meal_ids": past_meal_ids.detach().cpu(),
            }
            self.example_attn_weights_past = attn_past.detach().cpu()
            self.example_attn_weights_future = attn_future.detach().cpu()
            self.example_meal_self_attn_past = meal_self_attn_past # do not detach, as plotting functions needs to do some more processing
            self.example_meal_self_attn_future = meal_self_attn_future # do not detach, as plotting functions needs to do some more processing
        return {
            "val_quantile_loss": metrics["q_loss"],
            "pred_iAUC": metrics["pred_iAUC"],
            "true_iAUC": metrics["true_iAUC"],
        }

    def on_validation_epoch_end(self):
        if not hasattr(self, "val_outputs") or len(self.val_outputs) == 0:
            return
        outputs = self.val_outputs
        
        # Convert food embeddings to pandas DataFrame
        food_embeddings = self.meal_encoder.food_emb.weight.detach().cpu().numpy()
        
        # Create column names for the DataFrame
        embedding_cols = [f"embedding_{i}" for i in range(self.food_embed_dim)]
        food_vocab = [f"food_id_{i}" for i in range(self.num_foods)]
        
        
        
        dataset = self.trainer.val_dataloaders.dataset
        dataset.num_foods
        len(dataset.ppgr_dataset.categorical_encoders["food_id"].classes_)
        
        
        breakpoint()
                
        
        # Log to wandb as before
        self.logger.experiment.log({"food_embeddings": wandb.Table(
            data=food_embeddings,
            columns=embedding_cols,
        )}, step=self.global_step)
        
                
        # Rest of the existing validation epoch end code
        if self.example_forecasts is not None:
            fixed_indices = getattr(self, "fixed_forecast_indices", None)
            
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
        
        all_pred_iAUC = torch.cat([output["pred_iAUC"] for output in outputs], dim=0)
        all_true_iAUC = torch.cat([output["true_iAUC"] for output in outputs], dim=0)
        fig_scatter, corr = plot_iAUC_scatter(all_pred_iAUC, all_true_iAUC)
        self.logger.experiment.log({
            "iAUC_scatter": wandb.Image(fig_scatter),
            "iAUC_correlation": corr.item(),
            "global_step": self.global_step
        })
        plt.close(fig_scatter)
        self.log("val_iAUC_corr", corr.item(), prog_bar=True)
        self.val_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-5)

    def on_train_epoch_end(self):
        pass

# -----------------------------------------------------------------------------
# Loss and Metric Helper Functions
# -----------------------------------------------------------------------------
def quantile_loss(predictions, targets, quantiles):
    targets_expanded = targets.unsqueeze(-1)
    errors = targets_expanded - predictions
    losses = torch.max((quantiles - 1) * errors, quantiles * errors)
    return losses.mean()

def compute_iAUC(median_pred, future_glucose, past_glucose, target_scales, eval_window=None):
    past_glucose_unscaled = unscale_tensor(past_glucose, target_scales)
    future_glucose_unscaled = future_glucose * target_scales[:, 1].unsqueeze(1) + target_scales[:, 0].unsqueeze(1)
    if eval_window is not None:
        median_pred_eval = median_pred[:, :eval_window]
        future_glucose_unscaled_eval = future_glucose_unscaled[:, :eval_window]
    else:
        median_pred_eval = median_pred
        future_glucose_unscaled_eval = future_glucose_unscaled
    baseline = past_glucose_unscaled[:, -2:].mean(dim=1)
    pred_diff = median_pred_eval - baseline.unsqueeze(1)
    true_diff = future_glucose_unscaled_eval - baseline.unsqueeze(1)
    pred_iAUC = torch.trapz(torch.clamp(pred_diff, min=0), dx=1, dim=1)
    true_iAUC = torch.trapz(torch.clamp(true_diff, min=0), dx=1, dim=1)
    return pred_iAUC, true_iAUC

# -----------------------------------------------------------------------------
# Plotting Functions
# -----------------------------------------------------------------------------
def plot_forecast_examples(forecasts, attn_weights_past, attn_weights_future, quantiles, logger, global_step, fixed_indices=None):
    past = forecasts["past"]
    pred = forecasts["pred"]
    truth = forecasts["truth"]
    meal_ids_future = forecasts["future_meal_ids"]
    meal_ids_past = forecasts["past_meal_ids"]
    num_examples = min(4, past.size(0))
    if fixed_indices is None:
        fixed_indices = random.sample(list(range(past.size(0))), num_examples)
    sampled_indices = fixed_indices
    fig, axs = plt.subplots(num_examples, 3, figsize=(18, 4 * num_examples))
    if num_examples == 1:
        axs = [axs]
    for i, idx in enumerate(sampled_indices):
        ax_ts = axs[i][0]
        ax_attn_past = axs[i][1]
        ax_attn_future = axs[i][2]
        past_i = past[idx].cpu().numpy()
        pred_i = pred[idx].cpu().numpy()
        truth_i = truth[idx].cpu().numpy()
        attn_past_i = attn_weights_past[idx].cpu().numpy()
        attn_future_i = attn_weights_future[idx].cpu().numpy()
        T_context = past_i.shape[0]
        T_forecast = pred_i.shape[0]
        x_hist = list(range(-T_context + 1, 1))
        x_forecast = list(range(1, T_forecast + 1))
        ax_ts.plot(x_hist, past_i, marker="o", markersize=2, label="Historical Glucose")
        ax_ts.plot(x_forecast, truth_i, marker="o", markersize=2, label="Ground Truth Forecast")
        num_q = pred_i.shape[1]
        base_color = "blue"
        median_index = num_q // 2
        for qi in range(num_q - 1):
            alpha_val = 0.1 + (abs(qi - median_index)) * 0.05
            ax_ts.fill_between(x_forecast, pred_i[:, qi], pred_i[:, qi + 1], color=base_color, alpha=alpha_val / 2)
        ax_ts.plot(x_forecast, pred_i[:, median_index], marker="o", markersize=2, color="darkblue", label="Median Forecast")
        meal_label_added = False
        meals_past = meal_ids_past[idx].cpu().numpy()
        T_past = meals_past.shape[0]
        for t, meal in enumerate(meals_past):
            if (meal != 0).any():
                relative_time = t - T_past + 1
                if not meal_label_added:
                    ax_ts.axvline(x=relative_time, color="purple", linestyle="--", alpha=0.7, label="Meal Consumption")
                    meal_label_added = True
                else:
                    ax_ts.axvline(x=relative_time, color="purple", linestyle="--", alpha=0.7)
        meals_future = meal_ids_future[idx].cpu().numpy()
        for t, meal in enumerate(meals_future):
            if (meal != 0).any():
                relative_time = t + 1
                ax_ts.axvline(x=relative_time, color="purple", linestyle="--", alpha=0.7)
        ax_ts.set_xlabel("Relative Timestep")
        ax_ts.set_ylabel("Glucose Level")
        ax_ts.set_title(f"Forecast Example {i} (Idx: {idx})")
        ax_ts.legend(fontsize="small")
        im_past = ax_attn_past.imshow(attn_past_i, aspect="auto", cmap="viridis")
        ax_attn_past.set_title("Past Meals Attention")
        ax_attn_past.set_xlabel("Past Meal Timestep")
        ax_attn_past.set_ylabel("Glucose Timestep")
        fig.colorbar(im_past, ax=ax_attn_past, fraction=0.046, pad=0.04)
        im_future = ax_attn_future.imshow(attn_future_i, aspect="auto", cmap="viridis")
        ax_attn_future.set_title("Future Meals Attention")
        ax_attn_future.set_xlabel("Future Meal Timestep")
        ax_attn_future.set_ylabel("Glucose Timestep")
        fig.colorbar(im_future, ax=ax_attn_future, fraction=0.046, pad=0.04)
    fig.tight_layout()
    logger.experiment.log({"forecast_samples": wandb.Image(fig), "global_step": global_step})
    return fixed_indices, fig

def plot_iAUC_scatter(all_pred_iAUC, all_true_iAUC):
    mean_pred = torch.mean(all_pred_iAUC)
    mean_true = torch.mean(all_true_iAUC)
    cov = torch.mean((all_true_iAUC - mean_true) * (all_pred_iAUC - mean_pred))
    std_true = torch.std(all_true_iAUC, unbiased=False)
    std_pred = torch.std(all_pred_iAUC, unbiased=False)
    corr = cov / (std_true * std_pred)
    fig_scatter, ax_scatter = plt.subplots(figsize=(6, 6))
    ax_scatter.scatter(all_true_iAUC.cpu().numpy(), all_pred_iAUC.cpu().numpy(), alpha=0.5, s=0.5)
    ax_scatter.set_xlabel("True iAUC")
    ax_scatter.set_ylabel("Predicted iAUC")
    ax_scatter.set_title("iAUC Scatter Plot")
    ax_scatter.grid(True)
    ax_scatter.text(0.05, 0.95, f'Corr: {corr.item():.2f}', transform=ax_scatter.transAxes,
                    fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    return fig_scatter, corr

def pack_valid_attn_subblocks(attn_4d: torch.Tensor, meal_ids_3d: torch.Tensor):
    """
    Extract valid sub-blocks from attention [B,T,M,M] by removing padding around
    meal tokens.  Returns one "big" 2D numpy array, plus the total height (sum
    of valid M_i) and the max width (max(M_i)).
    
    attn_4d: [B, T, M, M] attention.  We'll ignore the start token dimension, so
             effectively we treat it as attn_4d[:, :, 1:, 1:].
    meal_ids_3d: [B, T, M] corresponding meal IDs.  A value of 0 means "padded".
    """
    # 1) Strip off the first row/col if you're ignoring the "start token"
    attn_4d = attn_4d[:, :, 1:, 1:]  # shape [B, T, M-1, M-1]
    meal_ids_3d = meal_ids_3d[:, :, 1:]  # shape [B, T, M-1]

    B, T, M, _ = attn_4d.shape
    # For each (b, t), figure out how many valid tokens there actually are:
    # (i.e. the number of non‐zero meal_ids).
    subblocks = []
    sizes = []
    for b in range(B):
        for t in range(T):
            meal_ids_slice = meal_ids_3d[b, t]  # shape [M]
            valid_count = (meal_ids_slice != 0).sum().item()
            if valid_count > 0:
                # slice out the valid sub-block [valid_count x valid_count]
                block = attn_4d[b, t, :valid_count, :valid_count]
                subblocks.append(block.cpu().numpy())
                sizes.append(valid_count)

    if not subblocks:
        # No valid sub‐blocks at all
        return np.zeros((1,1)), 1, 1

    # 2) Figure out how large an array we need:
    # Height is sum of all valid_counts, width is max of valid_counts
    total_height = sum(sizes)
    max_width = max(sizes)

    # 3) Create the big 2D array (we will fill it with NaN so that any "unfilled"
    # region is just blank).
    big_array = np.full((total_height, max_width), np.nan, dtype=np.float32)

    # 4) Copy each sub‐block in, one under the other
    row_offset = 0
    for block, size in zip(subblocks, sizes):
        big_array[row_offset:row_offset+size, 0:size] = block
        row_offset += size

    return big_array, total_height, max_width


def plot_meal_self_attention(
    attn_weights_past: torch.Tensor,
    meal_ids_past: torch.Tensor,
    attn_weights_future: torch.Tensor,
    meal_ids_future: torch.Tensor,
    logger,
    global_step: int,
    fixed_indices: list = None,
    max_examples: int = 3,
    random_samples: bool = True
):
    """
    Show up to `max_examples` samples from the batch, each in its own row of a
    2-column figure: left = self-attn for the 'past' meals, right = self-attn
    for the 'future' meals.
    """
    # Decide which batch indices we will plot:
    batch_size = attn_weights_past.size(0)
    if fixed_indices is not None:
        # Use the fixed indices provided (limit to at most max_examples)
        indices = fixed_indices[:min(max_examples, len(fixed_indices))]
    elif random_samples:
        indices = random.sample(range(batch_size), k=min(max_examples, batch_size))
    else:
        indices = list(range(min(max_examples, batch_size)))

    # Increase figure size and adjust spacing
    fig, axes = plt.subplots(
        nrows=len(indices), 
        ncols=2, 
        figsize=(16, 5 * len(indices)),  # Wider figure, more height per row
        constrained_layout=True  # Better automatic layout handling
    )
    if len(indices) == 1:
        axes = [axes]

    # A helper that packs sub-blocks and returns boundaries to draw horizontal lines.
    def pack_with_boundaries(attn_4d, meal_ids_3d):
        """
        Returns (big_array, subblock_row_boundaries).

        big_array is the result of pack_valid_attn_subblocks.
        subblock_row_boundaries is the cumulative row index after each sub-block.
        """
        # Strip off "start token" dimension:
        attn_4d = attn_4d[:, :, 1:, 1:]    # shape [B, T, M-1, M-1]
        meal_ids_3d = meal_ids_3d[:, :, 1:]  # shape [B, T, M-1]

        # We only have a single sample in [B], so drop that dimension:
        attn_4d = attn_4d[0]      # [T, M-1, M-1]
        meal_ids_3d = meal_ids_3d[0]  # [T, M-1]

        subblocks = []
        sizes = []
        for t in range(attn_4d.size(0)):
            meal_ids_slice = meal_ids_3d[t]
            valid_count = (meal_ids_slice != 0).sum().item()
            if valid_count > 0:
                block = attn_4d[t, :valid_count, :valid_count]
                subblocks.append(block.cpu().numpy())
                sizes.append(valid_count)
        
        if not subblocks:
            return np.zeros((1, 1), dtype=np.float32), []

        # Create big packed array:
        total_height = sum(sizes)
        max_width = max(sizes)
        big_array = np.full((total_height, max_width), np.nan, dtype=np.float32)

        boundaries = []
        row_offset = 0
        for sz, sb in zip(sizes, subblocks):
            big_array[row_offset:row_offset+sz, 0:sz] = sb
            row_offset += sz
            boundaries.append(row_offset)  # The boundary after this sub-block

        return big_array, boundaries

    # Define a consistent colormap and normalization
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=0, vmax=1)

    # Loop over the chosen examples:
    for row_i, idx in enumerate(indices):
        ax_past = axes[row_i][0]
        ax_fut = axes[row_i][1]

        # Extract the single sample's T×M×M from the batch:
        attn_past_1 = attn_weights_past[idx:idx+1]   # shape [1,T,M,M]
        meals_past_1 = meal_ids_past[idx:idx+1]        # shape [1,T,M]
        attn_fut_1  = attn_weights_future[idx:idx+1]   # shape [1,T,M,M]
        meals_fut_1 = meal_ids_future[idx:idx+1]        # shape [1,T,M]

        # Pack and plot with improved formatting
        packed_past, boundaries_past = pack_with_boundaries(attn_past_1, meals_past_1)
        packed_fut, boundaries_future = pack_with_boundaries(attn_fut_1, meals_fut_1)

        # Past attention plot
        im_past = ax_past.imshow(packed_past, aspect="auto", cmap=cmap, norm=norm)
        ax_past.set_title(f"Sample {idx} - Past Self-Attention", pad=10, fontsize=12)
        ax_past.set_xlabel("Token Position", fontsize=10)
        ax_past.set_ylabel("Stacked Timesteps", fontsize=10)
        
        # Add boundaries with improved visibility
        for b in boundaries_past:
            ax_past.axhline(y=b-0.5, color="white", linestyle="--", linewidth=0.8, alpha=0.6)
        
        # Customize ticks
        ax_past.tick_params(axis='both', which='major', labelsize=9)

        # Future attention plot
        im_fut = ax_fut.imshow(packed_fut, aspect="auto", cmap=cmap, norm=norm)
        ax_fut.set_title(f"Sample {idx} - Future Self-Attention", pad=10, fontsize=12)
        ax_fut.set_xlabel("Token Position", fontsize=10)
        ax_fut.set_ylabel("Stacked Timesteps", fontsize=10)
        
        # Add boundaries with improved visibility
        for b in boundaries_future:
            ax_fut.axhline(y=b-0.5, color="white", linestyle="--", linewidth=0.8, alpha=0.6)
        
        # Customize ticks
        ax_fut.tick_params(axis='both', which='major', labelsize=9)

        # Add gridlines for better readability
        ax_past.grid(False)
        ax_fut.grid(False)

    # Main title with improved positioning
    fig.suptitle("Meal Self-Attention for Selected Samples", 
                 fontsize=14, 
                 y=1.01,  # Slightly adjusted for constrained_layout
                 fontweight='bold')

    # Add a single colorbar with better positioning and formatting
    cbar = fig.colorbar(im_fut, ax=axes, 
                       aspect=30)
    cbar.ax.set_ylabel("Attention Weight", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # Log to WandB
    logger.experiment.log({"meal_self_attention_samples": wandb.Image(fig), 
                          "global_step": global_step})
    plt.close(fig)
    return fig

# -----------------------------------------------------------------------------
# DataLoader & Trainer Setup Functions
# -----------------------------------------------------------------------------
def get_dataloaders(config: ExperimentConfig):
    (training_dataset, validation_dataset, test_dataset, categorical_encoders, continuous_scalers) = create_cached_dataset(
        dataset_version=config.dataset_version,
        debug_mode=config.debug_mode,
        validation_percentage=config.validation_percentage,
        test_percentage=config.test_percentage,
        min_encoder_length=config.min_encoder_length,
        prediction_length=config.prediction_length,
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
    train_loader = DataLoader(
        training_dataset, 
        batch_size=config.batch_size, 
        num_workers=config.dataloader_num_workers, 
        pin_memory=True, 
        persistent_workers=True, 
        shuffle=True
    )
    val_loader = DataLoader(
        validation_dataset, 
        batch_size=config.batch_size, 
        num_workers=config.dataloader_num_workers, 
        pin_memory=True, 
        persistent_workers=True
    )
    return train_loader, val_loader, training_dataset

def get_trainer(config: ExperimentConfig, callbacks):
    wandb_logger = WandbLogger(
        project=config.wandb_project,
        name=config.wandb_run_name,
        config={
            "food_embed_dim": config.food_embed_dim,
            "hidden_dim": config.hidden_dim,
            "batch_size": config.batch_size,
            "optimizer_lr": config.optimizer_lr,
            "precision": config.precision,
            "debug": config.debug_mode,
        },
        log_model=True,
    )
    precision_value = int(config.precision) if config.precision == "32" else "bf16"
    trainer = pl.Trainer(
        profiler="simple",
        max_epochs=config.max_epochs,
        enable_checkpointing=False,
        logger=wandb_logger,
        callbacks=callbacks,
        precision=precision_value,
        gradient_clip_val=config.gradient_clip_val  # Use the value from config
    )
    return trainer

# -----------------------------------------------------------------------------
# Main Training & Evaluation Entry Point
# -----------------------------------------------------------------------------
@click.command()
@click.option("--debug/--no-debug", default=False, help="Enable debug logging.")
@click.option("--no-cache", is_flag=True, default=False, help="Ignore cached datasets and rebuild dataset from scratch.")
@click.option("--precision", type=click.Choice(["32", "bf16"]), default="bf16", help="Training precision: bf16 or 32")
def main(debug, no_cache, precision):
    logging.getLogger().setLevel(logging.DEBUG if debug else logging.INFO)
    logging.info(f"Starting main with debug={debug}, no_cache={no_cache}, precision={precision}.")
    config = ExperimentConfig(debug_mode=debug, use_cache=not no_cache, precision=precision)
    config.min_encoder_length = 32
    config.prediction_length = 32
    config.eval_window = 8
    train_loader, val_loader, training_dataset = get_dataloaders(config)
    model = MealGlucoseForecastModel(
        food_embed_dim=config.food_embed_dim,
        hidden_dim=config.hidden_dim,
        num_foods=training_dataset.num_foods,
        macro_dim=training_dataset.num_nutrients,
        max_meals=training_dataset.max_meals,
        glucose_seq_len=config.min_encoder_length,
        forecast_horizon=config.prediction_length,
        eval_window=config.eval_window,
        num_heads=config.num_heads,
        enc_layers=config.enc_layers,
        residual_pred=config.residual_pred,
        num_quantiles=config.num_quantiles,
        loss_iAUC_weight=config.loss_iAUC_weight,
        dropout_rate=config.dropout_rate,
        transformer_dropout=config.transformer_dropout,
    )
    
    # Compile the model
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = model.to(device)
    # model = torch.compile(model)
    
    rich_model_summary = RichModelSummary(max_depth=2)
    rich_progress_bar = RichProgressBar()
    callbacks = [rich_model_summary, rich_progress_bar]
    trainer = get_trainer(config, callbacks)
    logging.info("Starting training.")
    trainer.fit(model, train_loader, val_loader)
    logging.info("Training complete.")
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))
        batch = [x.to(device) for x in batch]
        (past_glucose, past_meal_ids, past_meal_macros,
         future_meal_ids, future_meal_macros, future_glucose, target_scales) = batch
        preds = model(
            past_glucose,
            past_meal_ids,
            past_meal_macros,
            future_meal_ids,
            future_meal_macros,
            target_scales,
            return_attn=True,
            return_meal_self_attn=True,
        )
        (pred_future, _, attn_past, _, attn_future, meal_self_attn_past, meal_self_attn_future) = preds
    logging.info("Predicted future glucose (first sample): %s", pred_future[0].cpu().numpy())
    logging.info("Actual future glucose (first sample):   %s", future_glucose[0].cpu().numpy())
    logging.info("Past cross-attention shape: %s", attn_past.shape)
    logging.info("Future cross-attention shape: %s", attn_future.shape)
    logging.info("Meal self-attention (past) shape: %s", meal_self_attn_past.shape)
    logging.info("Meal self-attention (future) shape: %s", meal_self_attn_future.shape)
    
if __name__ == "__main__":
    main()
