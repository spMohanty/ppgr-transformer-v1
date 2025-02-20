#!/usr/bin/env python
import os
import random
import pickle
import hashlib
import logging
import warnings
import math

from dataclasses import dataclass, asdict

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

from loguru import logger

from tqdm import tqdm

from utils import create_click_options
from plot_helpers import plot_meal_self_attention, plot_forecast_examples, plot_iAUC_scatter, plot_meal_self_attention

# -----------------------------------------------------------------------------
# Experiment Configuration
# -----------------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    # Dataset / caching settings
    dataset_version: str = "v0.5"
    cache_dir: str = "/scratch/mohanty/food/ppgr-v1/datasets-cache"
    use_cache: bool = True
    debug_mode: bool = False
    dataloader_num_workers: int = 7  # Added configurable dataloader_num_workers parameter

    # Data splitting & sequence parameters
    min_encoder_length: int = 8 * 4    # e.g., 8hrs * 4
    prediction_length: int = 4 * 4     # e.g.,  4hrs * 4
    eval_window: int = 2 * 4            # e.g., 2hrs * 4
    validation_percentage: float = 0.1
    test_percentage: float = 0.1
    
    patch_size: int = 1 * 4 # 1 hour patch size
    patch_stride: int = 1  # 15 min stride
    

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
    food_embed_dim: int = 2048 # the number of dimensions from the pre-trained embeddings to use
    food_embed_adapter_dim: int = 64 # the number of dimensions to project the food embeddings to for visualization purposes
    hidden_dim: int = 256
    num_heads: int = 4
    enc_layers: int = 2
    residual_pred: bool = False
    num_quantiles: int = 7
    loss_iauc_weight: float = 0.00

    # New dropout hyperparameters
    dropout_rate: float = 0.1          # Used for projections, cross-attention, forecast MLP, etc.
    transformer_dropout: float = 0.1   # Used within Transformer layers

    # Training hyperparameters
    batch_size: int = 1024 * 2
    max_epochs: int = 50
    optimizer_lr: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip_val: float = 0.1  # Added gradient clipping parameter

    # WandB logging
    wandb_project: str = "meal-representations-learning-v0"
    wandb_run_name: str = "MealGlucoseForecastModel_Run"

    # Precision
    precision: str = "bf16"

    # Batch size for projecting food embeddings when logging
    food_embedding_projection_batch_size: int = 1024 * 4

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
        food_embed_adapter_dim: int,
        hidden_dim: int,
        num_foods: int,
        macro_dim: int,
        max_meals: int = 11,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout_rate: float = 0.2,
        transformer_dropout: float = 0.1,
        bootstrap_food_id_embeddings: nn.Embedding = None
    ):
        super(MealEncoder, self).__init__()
        self.food_embed_dim = food_embed_dim
        self.food_embed_adapter_dim = food_embed_adapter_dim
        self.hidden_dim = hidden_dim
        self.max_meals = max_meals
        self.num_foods = num_foods
        self.macro_dim = macro_dim
                
        self.food_emb = nn.Embedding(num_foods, food_embed_dim, padding_idx=0)
        self.food_emb_adapter = nn.Linear(food_embed_dim, food_embed_adapter_dim) # A linear projection to make it easier to visualize
        self.food_emb_proj = nn.Linear(food_embed_adapter_dim, hidden_dim)
        self.macro_proj = nn.Linear(macro_dim, hidden_dim, bias=False)
        self.pos_emb = nn.Embedding(max_meals, hidden_dim)
        self.start_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        self.bootstrap_food_id_embeddings(bootstrap_food_id_embeddings)

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
        food_emb = self.food_emb_adapter(food_emb)
        
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

    def bootstrap_food_id_embeddings(self, bootstrap_food_id_embeddings: nn.Embedding):
        # Bootstrap the food id embeddings to pre-computed ones
        if bootstrap_food_id_embeddings is not None:
            with torch.no_grad():
                logger.warning(f"Bootstrapping food id embeddings with {self.food_embed_dim} dimensions of pre-computed embeddings")
                self.food_emb.weight.copy_(bootstrap_food_id_embeddings.weight[:, :self.food_embed_dim])
            logger.warning(f"Food id embeddings have been frozen")
            self.food_emb.weight.requires_grad = False  # Freeze the food id embeddings        

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


class PatchedGlucoseEncoder(nn.Module):
    def __init__(
        self, embed_dim: int, patch_size: int, patch_stride: int, num_heads: int = 4, num_layers: int = 1, max_seq_len: int = 100, dropout_rate: float = 0.2
    ):
        super(PatchedGlucoseEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        # For patch embedding:
        # If using a simple linear projection: (patch_size) -> (embed_dim)
        self.patch_proj = nn.Linear(patch_size, embed_dim)

        # The maximum number of patches you could get is roughly max_seq_len / patch_size
        # so create enough positional embeddings for that many patches.
        max_num_patches = (max_seq_len - patch_size) // patch_stride + 1
        self.pos_emb = nn.Embedding(max_num_patches, embed_dim)


        # You can use a TransformerEncoder or your custom TransformerEncoderWithAttn
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, glucose_seq: torch.Tensor) -> torch.Tensor:
        glucose_seq = glucose_seq.to(next(self.parameters()).dtype)
        B, T = glucose_seq.size()
        
        # 1) Patchify: if you want non-overlapping patches
        # shape => (B, floor(T / patch_size), patch_size)
        # for overlapping patches, you'd do a more advanced fold/unfold or 1D strided approach
        # but let's do the simplest approach first:
        patch_list = []
        for start in range(0, T - self.patch_size + 1, self.patch_stride):
            patch = glucose_seq[:, start:start + self.patch_size]  # shape (B, patch_size)
            patch_list.append(patch.unsqueeze(1))  # shape => (B, 1, patch_size)
        
        if len(patch_list) == 0:
            # Edge case: if T < patch_size. Might need a fallback or zero-padding
            # Just an example fallback:
            patch_list = [F.pad(glucose_seq, (0, self.patch_size - T))[:, None, :]]
        
        patches = torch.cat(patch_list, dim=1)  # shape => (B, N_patches, patch_size)
        # 2) Project each patch into embed_dim
        # We can flatten or keep it as is for linear layer
        # shape => (B*N_patches, patch_size)
        B_np, N_patches, _ = patches.shape
        patches = patches.view(B_np * N_patches, self.patch_size)
        
        # linear projection
        patch_emb = self.patch_proj(patches)  # (B*N_patches, embed_dim)
        patch_emb = self.dropout(patch_emb)
        patch_emb = patch_emb.view(B_np, N_patches, -1)  # => (B, N_patches, embed_dim)

        # 3) Add positional embedding
        # pos_indices => [0..N_patches-1]
        pos_indices = torch.arange(N_patches, device=glucose_seq.device)
        pos_enc = self.pos_emb(pos_indices).unsqueeze(0)  # shape (1, N_patches, embed_dim)
        patch_emb = patch_emb + pos_enc
        
        # 4) Run Transformer on patches
        patch_emb = self.transformer(patch_emb)  # => (B, N_patches, embed_dim)
        
        return patch_emb

# -----------------------------------------------------------------------------
# MealGlucoseForecastModel with configurable dropout parameters
# -----------------------------------------------------------------------------
class MealGlucoseForecastModel(pl.LightningModule):
    def __init__(
        self,
        food_embed_dim: int,
        food_embed_adapter_dim: int,
        hidden_dim: int,
        num_foods: int,
        macro_dim: int,
        max_meals: int = 11,
        glucose_seq_len: int = 20,
        forecast_horizon: int = 4,
        eval_window: int = None,
        num_heads: int = 4,
        enc_layers: int = 1,
        patch_size: int = 1 * 4,
        patch_stride: int = 1,
        residual_pred: bool = True,
        num_quantiles: int = 7,
        loss_iauc_weight: float = 1,
        dropout_rate: float = 0.2,
        transformer_dropout: float = 0.1,
        bootstrap_food_id_embeddings: nn.Embedding = None,
        optimizer_lr: float = 1e-4,            
        weight_decay: float = 1e-5,
        food_embedding_projection_batch_size: int = 1025   # NEW configurable parameter
    ):
        super(MealGlucoseForecastModel, self).__init__()
        self.forecast_horizon = forecast_horizon
        self.eval_window = eval_window if eval_window is not None else forecast_horizon

        self.food_embed_dim = food_embed_dim
        self.food_embed_adapter_dim = food_embed_adapter_dim
        self.hidden_dim = hidden_dim
        self.max_meals = max_meals
        self.num_foods = num_foods
        self.macro_dim = macro_dim
                
        self.residual_pred = residual_pred
        self.num_quantiles = num_quantiles
        self.loss_iauc_weight = loss_iauc_weight

        self.optimizer_lr = optimizer_lr       
        self.weight_decay = weight_decay       

        # NEW: store the configurable batch size for food embeddings logging
        self.food_embedding_projection_batch_size = food_embedding_projection_batch_size

        # Encoders with configurable dropout
        self.meal_encoder = MealEncoder(
            food_embed_dim, food_embed_adapter_dim, hidden_dim, num_foods, macro_dim, max_meals,
            num_heads, enc_layers, dropout_rate=dropout_rate, transformer_dropout=transformer_dropout,
            bootstrap_food_id_embeddings=bootstrap_food_id_embeddings
        )
        # self.glucose_encoder = GlucoseEncoder(
        #     hidden_dim, num_heads, enc_layers, glucose_seq_len, dropout_rate=dropout_rate
        # )
        self.glucose_encoder = PatchedGlucoseEncoder(
            embed_dim=hidden_dim, patch_size=patch_size, patch_stride=patch_stride, num_heads=num_heads, num_layers=enc_layers, max_seq_len=glucose_seq_len, dropout_rate=dropout_rate
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
        # glucose_enc = self.glucose_encoder(past_glucose)  # [B, T_glucose, hidden_dim]
        glucose_enc = self.glucose_encoder(past_glucose) # PatchedGlucoseEncoder

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
        weighted_iAUC_loss = self.loss_iauc_weight * iAUC_loss
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

    def on_train_start(self):
        self.log_food_embeddings("train_start")
        
    def on_train_end(self):
        self.log_food_embeddings("train_end")

    def on_validation_epoch_end(self):
        if not hasattr(self, "val_outputs") or len(self.val_outputs) == 0:
            return
        outputs = self.val_outputs
        
        
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

    def log_food_embeddings(self, label: str):
        logger.info(f"Logging food embeddings for {label}")
        # Retrieve the food embeddings from the embedding layer
        food_emb = self.meal_encoder.food_emb.weight  # shape: (num_foods, food_embed_dim)
        
        # Process the embeddings in batches using the configurable batch size
        batch_size = self.food_embedding_projection_batch_size  # NEW: now configurable through ExperimentConfig
        projected_embeddings = []
        
        with torch.no_grad():
            # Loop over the embeddings in batches
            for i in tqdm(range(0, food_emb.size(0), batch_size), desc="Projecting food embeddings for visualization"):
                batch = food_emb[i : i + batch_size]
                # Apply the layers in the correct order: adapter then projection
                proj_batch = self.meal_encoder.food_emb_adapter(batch)
                projected_embeddings.append(proj_batch.detach().cpu())
        
        # Concatenate all batches into a single array and convert to numpy
        projected_embeddings = torch.cat(projected_embeddings, dim=0).numpy()
        embedding_cols = [f"proj_embedding_{i}" for i in range(projected_embeddings.shape[1])]
        
        # Retrieve food names from your dataset (assuming they are available as a list)
        dataset = self.trainer.val_dataloaders.dataset
        food_names = dataset.food_names  # a list of food names corresponding to each row
        food_group_names = dataset.food_group_names  # a list of food group names corresponding to each row
        
        # Create a DataFrame with the projected embeddings and insert the food names as the first column
        food_embeddings_df = pd.DataFrame(projected_embeddings, columns=embedding_cols)
        food_embeddings_df.insert(0, 'food_group_name', food_group_names) # becomes the second column 
        food_embeddings_df.insert(0, 'food_name', food_names) # becomes the first column 
        # Log the DataFrame as a WandB Table
        self.logger.experiment.log({f"food_embeddings_{label}": wandb.Table(dataframe=food_embeddings_df)})

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.optimizer_lr,
            weight_decay=self.weight_decay
        )

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
    # Log the entire configuration to WandB by converting the dataclass to a dictionary.
    wandb_logger = WandbLogger(
        project=config.wandb_project,
        name=config.wandb_run_name,
        config=asdict(config),  # Now logs the whole config
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
@create_click_options(ExperimentConfig)
def main(**kwargs):
    logging.getLogger().setLevel(logging.DEBUG if kwargs["debug_mode"] else logging.INFO)
    config = ExperimentConfig(**kwargs)

    if config.debug_mode:
        config.dataloader_num_workers = 1 # for debugging

    train_loader, val_loader, training_dataset = get_dataloaders(config)
    
    if config.use_bootstraped_food_embeddings:
        bootstrap_food_id_embeddings = training_dataset.get_food_id_embeddings()
    else:
        bootstrap_food_id_embeddings = None
    
    model = MealGlucoseForecastModel(
        food_embed_dim=config.food_embed_dim,
        food_embed_adapter_dim=config.food_embed_adapter_dim,
        hidden_dim=config.hidden_dim,
        num_foods=training_dataset.num_foods,
        macro_dim=training_dataset.num_nutrients,
        max_meals=training_dataset.max_meals,
        glucose_seq_len=config.min_encoder_length,
        forecast_horizon=config.prediction_length,
        eval_window=config.eval_window,
        num_heads=config.num_heads,
        enc_layers=config.enc_layers,
        patch_size=config.patch_size,
        patch_stride=config.patch_stride,
        residual_pred=config.residual_pred,
        num_quantiles=config.num_quantiles,
        loss_iauc_weight=config.loss_iauc_weight,
        dropout_rate=config.dropout_rate,
        transformer_dropout=config.transformer_dropout,
        bootstrap_food_id_embeddings=bootstrap_food_id_embeddings,
        
        optimizer_lr=config.optimizer_lr,
        weight_decay=config.weight_decay,
        food_embedding_projection_batch_size=config.food_embedding_projection_batch_size
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
