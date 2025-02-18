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

    # Data splitting & sequence parameters
    min_encoder_length: int = 32    # e.g., 8 * 4
    prediction_length: int = 32     # e.g., 8 * 4
    eval_window: int = 8            # e.g., 2 * 4
    validation_percentage: float = 0.1
    test_percentage: float = 0.1

    # Data options
    is_food_anchored: bool = False
    sliding_window_stride: int = 1
    use_meal_level_food_covariates: bool = True
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
    loss_iAUC_weight: float = 0.01

    # Training hyperparameters
    batch_size: int = 1024
    max_epochs: int = 100
    optimizer_lr: float = 1e-3
    weight_decay: float = 1e-5

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
        self.dropout = nn.Dropout(dropout)
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
# Modified Meal Encoder (with self-attention return option)
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

        encoder_layer = TransformerEncoderLayerWithAttn(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
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
        macro_emb = self.macro_proj(meal_macros_flat)
        meal_token_emb = food_emb + macro_emb

        pos_indices = torch.arange(self.max_meals, device=meal_ids.device)
        pos_enc = self.pos_emb(pos_indices).unsqueeze(0)
        meal_token_emb = meal_token_emb + pos_enc

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
        # Use the output of the start token as the meal embedding.
        meal_timestep_emb = meal_attn_out[:, 0, :]
        meal_timestep_emb = meal_timestep_emb.view(B, T, self.hidden_dim)
        if return_self_attn:
            # Reshape self_attn from [B*T, num_tokens, num_tokens] to [B, T, num_tokens, num_tokens]
            num_tokens = meal_attn_out.size(1)
            self_attn = self_attn.view(B, T, num_tokens, num_tokens)
            return meal_timestep_emb, self_attn
        return meal_timestep_emb

# -----------------------------------------------------------------------------
# Glucose Encoder (unchanged)
# -----------------------------------------------------------------------------
class GlucoseEncoder(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int = 4, num_layers: int = 1, max_seq_len: int = 100
    ):
        super(GlucoseEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.glucose_proj = nn.Linear(1, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, glucose_seq: torch.Tensor) -> torch.Tensor:
        glucose_seq = glucose_seq.to(next(self.parameters()).dtype)
        B, T = glucose_seq.size()
        x = glucose_seq.unsqueeze(-1)
        x = self.glucose_proj(x)
        pos_indices = torch.arange(T, device=glucose_seq.device)
        pos_enc = self.pos_emb(pos_indices).unsqueeze(0)
        x = x + pos_enc
        x = self.encoder(x)
        return x

# -----------------------------------------------------------------------------
# MealGlucoseForecastModel with Optional Meal Self-Attention Return
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

        # Encoders
        self.meal_encoder = MealEncoder(
            food_embed_dim, hidden_dim, num_foods, macro_dim, max_meals, num_heads, num_layers=enc_layers
        )
        self.glucose_encoder = GlucoseEncoder(
            hidden_dim, num_heads, enc_layers, glucose_seq_len
        )

        # Cross-Attention Modules
        self.cross_attn_past = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.cross_attn_future = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )

        # Forecast MLP
        self.forecast_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
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

        # Encode meals—with optional self-attention weights
        if return_meal_self_attn:
            past_meal_enc, meal_self_attn_past = self.meal_encoder(past_meal_ids, past_meal_macros, return_self_attn=True)
            future_meal_enc, meal_self_attn_future = self.meal_encoder(future_meal_ids, future_meal_macros, return_self_attn=True)
        else:
            past_meal_enc = self.meal_encoder(past_meal_ids, past_meal_macros)
            future_meal_enc = self.meal_encoder(future_meal_ids, future_meal_macros)

        # Cross-attention for past and future meals
        attn_output_past, attn_weights_past = self.cross_attn_past(
            query=glucose_enc, key=past_meal_enc, value=past_meal_enc, need_weights=True
        )
        attn_output_future, attn_weights_future = self.cross_attn_future(
            query=glucose_enc, key=future_meal_enc, value=future_meal_enc, need_weights=True
        )
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
        # Request both cross-attention and meal self-attention weights
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
        # Save examples and attention weights for logging on the first validation batch.
        if batch_idx == 0:
            # preds returns:
            # (pred_future, past_meal_enc, attn_weights_past, future_meal_enc, attn_weights_future, meal_self_attn_past, meal_self_attn_future)
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
            self.example_meal_self_attn_past = meal_self_attn_past.detach().cpu()
            self.example_meal_self_attn_future = meal_self_attn_future.detach().cpu()
        return {
            "val_quantile_loss": metrics["q_loss"],
            "pred_iAUC": metrics["pred_iAUC"],
            "true_iAUC": metrics["true_iAUC"],
        }

    def on_validation_epoch_end(self):
        if not hasattr(self, "val_outputs") or len(self.val_outputs) == 0:
            return
        outputs = self.val_outputs

        # Plot aggregated meal self-attention weights and forecast examples if available
        if self.example_forecasts is not None:
            fixed_indices = getattr(self, "fixed_forecast_indices", None)
            if False and hasattr(self, "example_meal_self_attn_past") and hasattr(self, "example_meal_self_attn_future"):
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
            # Plot forecast examples (which uses self.example_forecasts)
            fixed_indices, fig = plot_forecast_examples(
                self.example_forecasts,
                self.example_attn_weights_past,
                self.example_attn_weights_future,
                self.quantiles,
                self.logger,
                self.global_step,
                fixed_indices=getattr(self, "fixed_forecast_indices", None)
            )
            self.fixed_forecast_indices = fixed_indices
            plt.close(fig)
            # Clear the example_forecasts once they've been used
            self.example_forecasts = None

        # Plot iAUC scatter plot and correlation
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
    """
    Plot forecast examples along with two attention heatmaps:
      1) Forecast time-series plot (with vertical meal consumption bars)
      2) Past Meals Attention Heatmap
      3) Future Meals Attention Heatmap
    """
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


import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import wandb

def plot_meal_self_attention(attn_weights_past, meal_ids_past, attn_weights_future, meal_ids_future, logger, global_step, fixed_indices=None):
    """
    For each past and future meal timestep (restricted to fixed_indices if provided),
    extract the 11x11 self-attention map for the food tokens (ignoring the start token)
    in a tensor-native manner. Then mask out rows/columns corresponding to invalid food items 
    (i.e. where meal_id <= 0). Only timesteps with at least one valid food item are included.
    
    Parameters:
      - attn_weights_past: Tensor of shape [B, T_past, 1+max_meals, 1+max_meals]
      - meal_ids_past: Tensor of shape [B, T_past, max_meals]
      - attn_weights_future: Tensor of shape [B, T_future, 1+max_meals, 1+max_meals]
      - meal_ids_future: Tensor of shape [B, T_future, max_meals]
      - logger: A WandB logger for logging the figure.
      - global_step: Current training step (for logging).
      - fixed_indices: Optional list (or tensor) of batch indices to restrict processing.
      
    Returns:
      - The aggregated matplotlib figure.
    """
    # Restrict to fixed indices if provided.
    if fixed_indices is not None:
        if not torch.is_tensor(fixed_indices):
            fixed_indices = torch.tensor(fixed_indices, dtype=torch.long, device=attn_weights_past.device)
        attn_weights_past = attn_weights_past.index_select(0, fixed_indices)
        meal_ids_past = meal_ids_past.index_select(0, fixed_indices)
        attn_weights_future = attn_weights_future.index_select(0, fixed_indices)
        meal_ids_future = meal_ids_future.index_select(0, fixed_indices)
    
    mini_plots = []  # List to hold (attn_matrix, label) tuples.
    
    # --- Process Past Meal Timesteps ---
    # Extract food tokens (ignore the start token): shape [B, T_past, max_meals, max_meals]
    attn_past_sub = attn_weights_past[:, :, 1:, 1:]
    # Validity mask for food tokens: shape [B, T_past, max_meals]
    valid_mask = (meal_ids_past > 0)
    # Combined mask: valid only if both row and column are valid.
    combined_mask = valid_mask.unsqueeze(-1) & valid_mask.unsqueeze(-2)  # shape [B, T_past, max_meals, max_meals]
    # Clone and set invalid positions to NaN.
    attn_past_mod = attn_past_sub.clone()
    attn_past_mod[~combined_mask] = float('nan')
    # Find timesteps with at least one valid food item.
    valid_counts = valid_mask.sum(dim=-1)  # shape [B, T_past]
    valid_indices = torch.nonzero(valid_counts > 0, as_tuple=False)  # Each row: [b, t]
    
    for idx in valid_indices:
        b, t = idx.tolist()
        label = f"Past (b={b}, t={t})"
        mini_plots.append((attn_past_mod[b, t].cpu().numpy(), label))
    
    # --- Process Future Meal Timesteps ---
    attn_future_sub = attn_weights_future[:, :, 1:, 1:]  # shape [B, T_future, max_meals, max_meals]
    valid_mask_future = (meal_ids_future > 0)
    combined_mask_future = valid_mask_future.unsqueeze(-1) & valid_mask_future.unsqueeze(-2)
    attn_future_mod = attn_future_sub.clone()
    attn_future_mod[~combined_mask_future] = float('nan')
    valid_counts_future = valid_mask_future.sum(dim=-1)  # shape [B, T_future]
    valid_indices_future = torch.nonzero(valid_counts_future > 0, as_tuple=False)
    
    for idx in valid_indices_future:
        b, t = idx.tolist()
        label = f"Future (b={b}, t={t})"
        mini_plots.append((attn_future_mod[b, t].cpu().numpy(), label))
    
    n_plots = len(mini_plots)
    if n_plots == 0:
        return None  # Nothing to plot.
    
    # Arrange mini-plots in a grid.
    ncols = 5
    nrows = math.ceil(n_plots / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    axes = np.array(axes).reshape(-1)
    
    max_meals = meal_ids_past.shape[-1]  # e.g., 11
    for i, (attn_matrix, label) in enumerate(mini_plots):
        ax = axes[i]
        im = ax.imshow(attn_matrix, interpolation="nearest", cmap="viridis")
        ax.set_title(label, fontsize=8)
        ax.set_xticks(range(max_meals))
        ax.set_yticks(range(max_meals))
        # Label tokens from 1 to max_meals.
        xticklabels = [str(j+1) for j in range(max_meals)]
        yticklabels = [str(j+1) for j in range(max_meals)]
        ax.set_xticklabels(xticklabels, fontsize=6)
        ax.set_yticklabels(yticklabels, fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Turn off unused axes.
    for j in range(n_plots, len(axes)):
        axes[j].axis("off")
    
    fig.suptitle("Aggregate 11x11 Meal Self-Attention (Food Items Only)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    logger.experiment.log({"meal_self_attention": wandb.Image(fig), "global_step": global_step})
    
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
    train_loader = DataLoader(training_dataset, batch_size=config.batch_size, num_workers=7, pin_memory=True, persistent_workers=True, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=config.batch_size, num_workers=7, pin_memory=True, persistent_workers=True)
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
    )
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
