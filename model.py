#!/usr/bin/env python
import os
import random
import pickle
import hashlib
import logging
import warnings

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
    is_food_anchored: bool = True
    sliding_window_stride: int = None
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
    max_epochs: int = 30
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
# Model Components
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

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, meal_ids: torch.LongTensor, meal_macros: torch.Tensor) -> torch.Tensor:
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
        meal_attn_out = self.encoder(meal_token_emb, src_key_padding_mask=pad_mask)
        meal_timestep_emb = meal_attn_out[:, 0, :]
        meal_timestep_emb = meal_timestep_emb.view(B, T, self.hidden_dim)
        return meal_timestep_emb


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
        # Ensure proper precision.
        glucose_seq = glucose_seq.to(next(self.parameters()).dtype)
        B, T = glucose_seq.size()
        x = glucose_seq.unsqueeze(-1)
        x = self.glucose_proj(x)
        pos_indices = torch.arange(T, device=glucose_seq.device)
        pos_enc = self.pos_emb(pos_indices).unsqueeze(0)
        x = x + pos_enc
        x = self.encoder(x)
        return x


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

        self.meal_encoder = MealEncoder(
            food_embed_dim, hidden_dim, num_foods, macro_dim, max_meals, num_heads, num_layers=enc_layers
        )
        self.glucose_encoder = GlucoseEncoder(
            hidden_dim, num_heads, enc_layers, glucose_seq_len
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
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
        return_attn=False,
    ):
        glucose_enc = self.glucose_encoder(past_glucose)
        past_meal_enc = self.meal_encoder(past_meal_ids, past_meal_macros)
        future_meal_enc = self.meal_encoder(future_meal_ids, future_meal_macros)

        self.past_meal_len = past_meal_enc.size(1)
        self.future_meal_len = future_meal_enc.size(1)
        meal_enc_combined = torch.cat([past_meal_enc, future_meal_enc], dim=1)

        attn_output, attn_weights = self.cross_attn(
            query=glucose_enc, key=meal_enc_combined, value=meal_enc_combined, need_weights=True
        )
        combined_glucose = attn_output + glucose_enc
        self.last_attn_weights = attn_weights

        final_rep = torch.cat(
            [combined_glucose[:, -1, :], future_meal_enc[:, -1, :]], dim=-1
        )
        # For numerical stability, cast to float32.
        final_rep_fp32 = final_rep.float()
        pred_future = self.forecast_mlp(final_rep_fp32)
        pred_future = pred_future.view(final_rep.size(0), self.forecast_horizon, self.num_quantiles)
        if self.residual_pred:
            last_val = past_glucose[:, -1].unsqueeze(1).unsqueeze(-1).float()
            pred_future = pred_future + last_val

        pred_future = unscale_tensor(pred_future, target_scales)
        if return_attn:
            return pred_future, past_meal_enc, attn_weights
        return pred_future

    def _compute_forecast_metrics(self, past_glucose, future_glucose, target_scales, preds):
        """
        Unified helper that computes quantile loss, RMSE, and iAUC-based loss.
        It also extracts attention weights if they are returned.
        """
        # If predictions is a tuple, then the actual forecast tensor is the first element.
        if isinstance(preds, tuple):
            predictions = preds[0]
            attn_weights = preds[2]
        else:
            predictions = preds
            attn_weights = None

        # Unscale future glucose
        future_glucose_unscaled = (
            future_glucose * target_scales[:, 1].unsqueeze(1) + target_scales[:, 0].unsqueeze(1)
        )
        # Quantile loss computed on the raw predictions.
        q_loss = quantile_loss(predictions, future_glucose_unscaled, self.quantiles)

        # For iAUC and RMSE, extract the median forecast (assumed to be at index 3).
        median_pred = predictions[:, :, 3]

        # Use eval_window to compute RMSE for only a subset of the forecast horizon.
        median_pred_eval = median_pred[:, :self.eval_window]
        future_glucose_unscaled_eval = future_glucose_unscaled[:, :self.eval_window]
        rmse = torch.sqrt(F.mse_loss(median_pred_eval, future_glucose_unscaled_eval))

        # Compute iAUC losses based on the median prediction.
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
            "attn_weights": attn_weights,
        }

    def training_step(self, batch, batch_idx):
        (
            past_glucose,
            past_meal_ids,
            past_meal_macros,
            future_meal_ids,
            future_meal_macros,
            future_glucose,
            target_scales,
        ) = batch

        if target_scales.dim() > 2:
            target_scales = target_scales.view(target_scales.size(0), -1)

        # Forward pass without attention return.
        preds = self(
            past_glucose,
            past_meal_ids,
            past_meal_macros,
            future_meal_ids,
            future_meal_macros,
            target_scales,
        )

        metrics = self._compute_forecast_metrics(past_glucose, future_glucose, target_scales, preds)

        self.log("train_quantile_loss", metrics["q_loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_rmse", metrics["rmse"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_iAUC_loss", metrics["iAUC_loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_total_loss", metrics["total_loss"], on_step=True, on_epoch=True, prog_bar=True)

        return metrics["total_loss"]

    def validation_step(self, batch, batch_idx):
        (
            past_glucose,
            past_meal_ids,
            past_meal_macros,
            future_meal_ids,
            future_meal_macros,
            future_glucose,
            target_scales,
        ) = batch

        if target_scales.dim() > 2:
            target_scales = target_scales.view(target_scales.size(0), -1)

        # Forward pass with attention return.
        preds = self(
            past_glucose,
            past_meal_ids,
            past_meal_macros,
            future_meal_ids,
            future_meal_macros,
            target_scales,
            return_attn=True,
        )

        metrics = self._compute_forecast_metrics(past_glucose, future_glucose, target_scales, preds)

        self.log("val_quantile_loss", metrics["q_loss"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_rmse", metrics["rmse"], on_step=False, on_epoch=True, prog_bar=True)

        if not hasattr(self, "val_outputs"):
            self.val_outputs = []
        self.val_outputs.append(
            {
                "val_quantile_loss": metrics["q_loss"],
                "pred_iAUC": metrics["pred_iAUC"],
                "true_iAUC": metrics["true_iAUC"],
            }
        )

        # Save examples and attention weights for logging on first validation batch.
        if batch_idx == 0:
            self.example_forecasts = {
                "past": unscale_tensor(past_glucose, target_scales).detach().cpu(),
                "pred": preds[0].detach().cpu() if isinstance(preds, tuple) else preds.detach().cpu(),
                "truth": (
                    future_glucose * target_scales[:, 1].unsqueeze(1)
                    + target_scales[:, 0].unsqueeze(1)
                ).detach().cpu(),
                "future_meal_ids": future_meal_ids.detach().cpu(),
                "past_meal_ids": past_meal_ids.detach().cpu(),
            }
            self.example_attn_weights = preds[2].detach().cpu() if isinstance(preds, tuple) else None

        return {
            "val_quantile_loss": metrics["q_loss"],
            "pred_iAUC": metrics["pred_iAUC"],
            "true_iAUC": metrics["true_iAUC"],
        }

    def on_validation_epoch_end(self):
        if not hasattr(self, "val_outputs") or len(self.val_outputs) == 0:
            return
        outputs = self.val_outputs

        # ----- Plot Forecast Examples with Attention Heatmaps -----
        if self.example_forecasts is not None:
            fixed_indices, fig = plot_forecast_examples(
                self.example_forecasts,
                self.example_attn_weights,
                self.past_meal_len,
                self.future_meal_len,
                self.quantiles,
                self.logger,
                self.global_step,
                fixed_indices=getattr(self, "fixed_forecast_indices", None)
            )
            self.fixed_forecast_indices = fixed_indices
            plt.close(fig)
            self.example_forecasts = None

        # ----- iAUC Scatter Plot and Correlation -----
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
        # Optionally log additional statistics.
        pass

# -----------------------------------------------------------------------------
# Loss and Metric Helper Functions
# -----------------------------------------------------------------------------
def quantile_loss(predictions, targets, quantiles):
    """
    Compute the quantile (pinball) loss.
    predictions: Tensor of shape [B, T, Q]
    targets: Tensor of shape [B, T]
    quantiles: Tensor of shape [Q]
    """
    targets_expanded = targets.unsqueeze(-1)
    errors = targets_expanded - predictions
    losses = torch.max((quantiles - 1) * errors, quantiles * errors)
    return losses.mean()


def compute_iAUC(median_pred, future_glucose, past_glucose, target_scales, eval_window=None):
    """
    Compute the integrated Area Under the Curve (iAUC) for the median forecast predictions and the ground truth.
    """
    past_glucose_unscaled = unscale_tensor(past_glucose, target_scales)
    future_glucose_unscaled = (
        future_glucose * target_scales[:, 1].unsqueeze(1) + target_scales[:, 0].unsqueeze(1)
    )
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
def plot_forecast_examples(forecasts, attn_weights, past_meal_len, future_meal_len, quantiles, logger, global_step, fixed_indices=None):
    """
    Plot forecast examples along with attention heatmaps.
    """
    past = forecasts["past"]
    pred = forecasts["pred"]  # [B, forecast_horizon, num_quantiles]
    truth = forecasts["truth"]
    future_meal_ids = forecasts["future_meal_ids"]
    past_meal_ids = forecasts.get("past_meal_ids", None)
    num_examples = min(4, past.size(0))

    if fixed_indices is None:
        fixed_indices = random.sample(list(range(past.size(0))), num_examples)
    sampled_indices = fixed_indices

    fig, axs = plt.subplots(num_examples, 2, figsize=(14, 4 * num_examples))
    if num_examples == 1:
        axs = [axs]

    for i, idx in enumerate(sampled_indices):
        ax_ts = axs[i][0]
        past_i = past[idx].cpu().numpy()
        pred_i = pred[idx].cpu().numpy()  # [T_forecast, num_quantiles]
        truth_i = truth[idx].cpu().numpy()
        meals_i = future_meal_ids[idx].cpu().numpy()
        past_meals_i = past_meal_ids[idx].cpu().numpy() if past_meal_ids is not None else None

        T_context = past_i.shape[0]
        T_forecast = pred_i.shape[0]
        x_hist = list(range(-T_context + 1, 1))
        x_forecast = list(range(1, T_forecast + 1))

        # Plot historical glucose.
        ax_ts.plot(x_hist, past_i, marker="o", markersize=2, label="Historical Glucose")

        # Mark historical meal consumption.
        if past_meals_i is not None:
            meal_label_added_hist = False
            for j, x_coord in enumerate(x_hist):
                if (past_meals_i[j] != 0).any():
                    if not meal_label_added_hist:
                        ax_ts.axvline(x=x_coord, color="green", linestyle="--", alpha=0.7, label="Historical Meal Consumption")
                        meal_label_added_hist = True
                    else:
                        ax_ts.axvline(x=x_coord, color="green", linestyle="--", alpha=0.7)

        # Plot ground truth forecast.
        ax_ts.plot(x_forecast, truth_i, marker="o", markersize=2, label="Ground Truth Forecast")

        # Plot quantile forecast regions.
        num_quantiles = pred_i.shape[1]
        base_color = "blue"
        median_index = num_quantiles // 2
        for qi in range(num_quantiles - 1):
            if qi < median_index:
                alpha_val = 0.1 + (qi + 1) * 0.15
            else:
                alpha_val = 0.1 + (num_quantiles - qi - 1) * 0.15
            ax_ts.fill_between(
                x_forecast,
                pred_i[:, qi],
                pred_i[:, qi + 1],
                color=base_color,
                alpha=alpha_val / 4,
                label=f"{quantiles[qi]:.2f}-{quantiles[qi+1]:.2f}" if qi == median_index - 1 else None
            )

        # Highlight the median forecast.
        ax_ts.plot(x_forecast, pred_i[:, median_index], marker="o", markersize=2, color="darkblue", label="Median Forecast")

        # Mark future meal consumption.
        meal_label_added_forecast = False
        for t, x_coord in enumerate(x_forecast):
            if (meals_i[t] != 0).any():
                if not meal_label_added_forecast:
                    ax_ts.axvline(x=x_coord, color="red", linestyle="--", alpha=0.7, label="Future Meal Consumption")
                    meal_label_added_forecast = True
                else:
                    ax_ts.axvline(x=x_coord, color="red", linestyle="--", alpha=0.7)

        ax_ts.set_xlabel("Relative Timestep")
        ax_ts.set_ylabel("Glucose Level")
        ax_ts.set_title(f"Forecast Example {i} (Dataset Index: {idx})")
        ax_ts.legend(fontsize="small")

        # Right Plot: Attention Heatmap
        ax_attn = axs[i][1]
        attn = attn_weights[idx].cpu().numpy()
        T_glucose, _ = attn.shape
        key_tick_labels = [i - (past_meal_len - 1) for i in range(past_meal_len)] + [i + 1 for i in range(future_meal_len)]
        ax_attn.set_xticks(range(len(key_tick_labels)))
        ax_attn.set_xticklabels(key_tick_labels, rotation=90, fontsize=6)
        query_tick_labels = list(range(-T_glucose + 1, 1))
        ax_attn.set_yticks(range(T_glucose))
        ax_attn.set_yticklabels(query_tick_labels, fontsize=6)
        im = ax_attn.imshow(attn, aspect="auto", cmap="viridis")
        ax_attn.set_xlabel("Meal Timestep")
        ax_attn.set_ylabel("Glucose Timestep")
        ax_attn.set_title(f"Attention Weights (Sample {idx})")
        fig.colorbar(im, ax=ax_attn)

    fig.tight_layout()
    logger.experiment.log({"forecast_samples": wandb.Image(fig), "global_step": global_step})
    return fixed_indices, fig


def plot_iAUC_scatter(all_pred_iAUC, all_true_iAUC):
    """
    Plots a scatter plot comparing the predicted and true iAUC values.
    """
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
    ax_scatter.text(
        0.05, 0.95, f'Corr: {corr.item():.2f}',
        transform=ax_scatter.transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.5)
    )
    return fig_scatter, corr

# -----------------------------------------------------------------------------
# DataLoader & Trainer Setup Functions
# -----------------------------------------------------------------------------
def get_dataloaders(config: ExperimentConfig):
    """Creates training, validation, and test datasets and corresponding DataLoaders."""
    (
        training_dataset,
        validation_dataset,
        test_dataset,
        categorical_encoders,
        continuous_scalers,
    ) = create_cached_dataset(
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
    """Returns a PyTorch Lightning Trainer configured with WandB and callbacks."""
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
    """
    Main entry point of the script.
    """
    # Initialize logging.
    logging.getLogger().setLevel(logging.DEBUG if debug else logging.INFO)
    logging.info(f"Starting main with debug={debug}, no_cache={no_cache}, precision={precision}.")

    # Create experiment configuration.
    config = ExperimentConfig(debug_mode=debug, use_cache=not no_cache, precision=precision)

    # Set additional hyperparameters if needed.
    config.min_encoder_length = 32
    config.prediction_length = 32
    config.eval_window = 8

    # DataLoader creation.
    train_loader, val_loader, training_dataset = get_dataloaders(config)

    # Initialize the model.
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

    # Setup callbacks.
    rich_model_summary = RichModelSummary(max_depth=2)
    rich_progress_bar = RichProgressBar()
    callbacks = [rich_model_summary, rich_progress_bar]

    # Get trainer.
    trainer = get_trainer(config, callbacks)

    logging.info("Starting training.")
    trainer.fit(model, train_loader, val_loader)
    logging.info("Training complete.")

    # Evaluate on a single validation batch.
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))
        batch = [x.to(device) for x in batch]
        (past_glucose, past_meal_ids, past_meal_macros,
         future_meal_ids, future_meal_macros, future_glucose, target_scales) = batch

        preds, past_meal_embeds, attn_weights = model(
            past_glucose,
            past_meal_ids,
            past_meal_macros,
            future_meal_ids,
            future_meal_macros,
            target_scales,
            return_attn=True,
        )
    logging.info("Predicted future glucose (first sample): %s", preds[0].cpu().numpy())
    logging.info("Actual future glucose (first sample):   %s", future_glucose[0].cpu().numpy())
    logging.info("Past meal embedding shape: %s", past_meal_embeds.shape)
    logging.info("Cross-attention weight shape: %s", attn_weights.shape)
    logging.info("Attention weights (first sample): %s", attn_weights[0])


if __name__ == "__main__":
    main()
