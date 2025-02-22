#!/usr/bin/env python
import os
import random
import pickle
import hashlib
import logging
import warnings
import math
import datetime

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

from typing import Optional

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

from config import ExperimentConfig, generate_experiment_name

from callbacks import LRSchedulerCallback

# -----------------------------------------------------------------------------
#   - "TransformerEncoderLayerWithAttn" for single self-attention + feedforward
#   - "TransformerEncoderWithAttn" to stack multiple layers and optionally return the last layer's attn
# -----------------------------------------------------------------------------
class TransformerEncoderLayerWithAttn(nn.Module):
    """
    A single Transformer encoder layer with self-attention + feed-forward.
    Returns (output, attn_weights) if return_attn=True on the last layer.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Activation function
        self.activation_fn = F.relu if activation == "relu" else F.gelu

    def forward(self, src, src_mask=None, src_key_padding_mask=None, return_attn=False):
        # 1) Self-Attention
        attn_output, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True
        )
        src = src + self.dropout_attn(attn_output)
        src = self.norm1(src)

        # 2) Feed-forward
        ff = self.linear2(self.dropout_ffn(self.activation_fn(self.linear1(src))))
        src = src + ff
        src = self.norm2(src)

        if return_attn:
            return src, attn_weights
        return src


class TransformerEncoderWithAttn(nn.Module):
    """
    Stacks multiple TransformerEncoderLayerWithAttn. 
    If return_attn=True, returns the final layer's attn_weights.
    """
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None, return_attn=False):
        output = src
        attn_weights = None

        for layer_idx, layer in enumerate(self.layers):
            # Only return attention from the final layer if asked
            is_last = (layer_idx == len(self.layers) - 1)
            if return_attn and is_last:
                output, attn_weights = layer(
                    output,
                    src_mask=mask,
                    src_key_padding_mask=src_key_padding_mask,
                    return_attn=True
                )
            else:
                output = layer(
                    output,
                    src_mask=mask,
                    src_key_padding_mask=src_key_padding_mask,
                    return_attn=False
                )

        if self.norm is not None:
            output = self.norm(output)

        if return_attn:
            return output, attn_weights
        return output


# -----------------------------------------------------------------------------
#   - "TransformerDecoderLayerWithAttn" for self-attn + cross-attn + feedforward
#   - "TransformerDecoderWithAttn" to stack multiple layers and optionally return the last layer's cross-attn
# -----------------------------------------------------------------------------
class TransformerDecoderLayerWithAttn(nn.Module):
    """
    A single Transformer decoder layer with self-attention + cross-attention + feed-forward.
    Returns cross-attn weights if return_attn=True.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        # Self-attn
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Cross-attn
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # Feed-forward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_cross = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Activation function
        self.activation_fn = F.relu if activation == "relu" else F.gelu

    def forward(
        self,
        tgt, 
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        return_attn=False
    ):
        # 1) Self-Attn
        x = tgt
        sa_out, _ = self.self_attn(
            x, x, x,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False
        )
        x = x + self.dropout_attn(sa_out)
        x = self.norm1(x)

        # 2) Cross-Attn
        cross_attn_weights = None
        ca_out, ca_weights = self.cross_attn(
            x, memory, memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=True
        )
        x = x + self.dropout_cross(ca_out)
        x = self.norm2(x)

        if return_attn:
            cross_attn_weights = ca_weights  # shape [B, T_tgt, T_mem]

        # 3) Feed-forward
        ff = self.linear2(self.dropout_ffn(self.activation_fn(self.linear1(x))))
        x = x + ff
        x = self.norm3(x)

        if return_attn:
            return x, cross_attn_weights
        return x, None


class TransformerDecoderWithAttn(nn.Module):
    """
    Stacks multiple TransformerDecoderLayerWithAttn.
    If return_attn=True, returns the final layer's cross-attn weights.
    """
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.norm = norm

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        return_attn=False
    ):
        output = tgt
        cross_attn_weights = None

        for layer_idx, layer in enumerate(self.layers):
            is_last = (layer_idx == len(self.layers) - 1)
            if return_attn and is_last:
                output, cross_attn_weights = layer(
                    output,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                    return_attn=True
                )
            else:
                output, _ = layer(
                    output,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                    return_attn=False
                )

        if self.norm is not None:
            output = self.norm(output)

        return output, cross_attn_weights


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
        aggregator_type: str = "set",  # Either "set" or "sum"
        bootstrap_food_id_embeddings: Optional[nn.Embedding] = None
    ):
        super(MealEncoder, self).__init__()
        self.food_embed_dim = food_embed_dim
        self.food_embed_adapter_dim = food_embed_adapter_dim
        self.hidden_dim = hidden_dim
        self.max_meals = max_meals
        self.num_foods = num_foods
        self.macro_dim = macro_dim
        self.aggregator_type = aggregator_type.lower().strip()

        # --------------------
        #  Embedding Layers
        # --------------------
        self.food_emb = nn.Embedding(num_foods, food_embed_dim, padding_idx=0)
        self.food_emb_adapter = nn.Linear(food_embed_dim, food_embed_adapter_dim) # A linear projection to make it easier to visualize
        self.food_emb_proj = nn.Linear(food_embed_adapter_dim, hidden_dim)
        self.macro_proj = nn.Linear(macro_dim, hidden_dim, bias=False)

                
        # Optional aggregator token to collect representations (only used in "set" mode).
        self.start_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # Bootstrapping pre-trained food embeddings if provided
        self.bootstrap_food_id_embeddings(bootstrap_food_id_embeddings)
        
        #  Dropout
        self.dropout = nn.Dropout(dropout_rate)

        #  Transformer stack (only for aggregator_type="set")
        if self.aggregator_type == "set":
            enc_layer = TransformerEncoderLayerWithAttn(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=transformer_dropout,
                activation="relu"
            )
            self.encoder = TransformerEncoderWithAttn(
                enc_layer,
                num_layers=num_layers,
                norm=nn.LayerNorm(hidden_dim)
            )
        else:
            self.encoder = None  # No Transformer for aggregator_type="sum"
            
    
    def forward(
        self,
        meal_ids: torch.LongTensor,
        meal_macros: torch.Tensor,
        return_self_attn: bool = False
    ):
        """
        :param meal_ids:    (B, T, M)  with each M being item indices in that meal
        :param meal_macros: (B, T, M, macro_dim)  macro features for each item
        :param return_self_attn: If True, returns self-attention weights (only works in "set" mode).
        :return: 
          - meal_timestep_emb: (B, T, hidden_dim) aggregator encoding for each meal/time-step
          - self_attn (if "set" mode and return_self_attn=True), else None
        """
        B, T, M = meal_ids.size()

        # 1) Flatten out (B,T) -> (B*T) for item-level embeddings
        meal_ids_flat = meal_ids.view(B * T, M)            # => (B*T, M)
        meal_macros_flat = meal_macros.view(B * T, M, -1)  # => (B*T, M, macro_dim)

        # 2) Embed items
        food_emb = self.food_emb(meal_ids_flat)               # (B*T, M, food_embed_dim)
        food_emb = self.food_emb_adapter(food_emb)            # (B*T, M, food_embed_adapter_dim)
        food_emb = self.food_emb_proj(food_emb)               # (B*T, M, hidden_dim)
        food_emb = self.dropout(food_emb)

        macro_emb = self.macro_proj(meal_macros_flat)         # (B*T, M, hidden_dim)
        macro_emb = self.dropout(macro_emb)

        # Combine them
        meal_token_emb = food_emb + macro_emb  # shape = (B*T, M, hidden_dim)

        # -------------------------------------------------------------------
        #  aggregator_type = "sum"
        # -------------------------------------------------------------------
        if self.aggregator_type == "sum":
            # Permutation-invariant aggregator: sum (or mean) across items
            # (B*T, M, hidden_dim) -> (B*T, hidden_dim)
            # NOTE: You could do meal_token_emb.mean(dim=1) if you prefer averaging
            meal_timestep_emb = meal_token_emb.sum(dim=1)
            meal_timestep_emb = meal_timestep_emb.view(B, T, self.hidden_dim)
            # No self-attn in sum mode
            self_attn = None

            if return_self_attn:
                # We don't have attention to return in sum mode
                return meal_timestep_emb, None
            else:
                return meal_timestep_emb

        # -------------------------------------------------------------------
        #  aggregator_type = "set"
        # -------------------------------------------------------------------
        elif self.aggregator_type == "set":
            # We do a self-attention aggregator with a special start token
            # *Without* adding any positional embeddings to keep it permutation-invariant
            # (i.e. we skip meal_token_emb + pos_enc).

            # Insert aggregator token at position 0
            start_token_expanded = self.start_token.expand(B * T, -1, -1)  # => (B*T, 1, hidden_dim)
            meal_token_emb = torch.cat([start_token_expanded, meal_token_emb], dim=1)
            # => shape (B*T, M+1, hidden_dim)

            # Build a padding mask: treat item_id=0 as padding
            pad_mask = (meal_ids_flat == 0)  # shape (B*T, M)
            zero_col = torch.zeros(B * T, 1, dtype=torch.bool, device=pad_mask.device)
            pad_mask = torch.cat([zero_col, pad_mask], dim=1)  # => (B*T, M+1)

            if self.encoder is None:
                raise ValueError(
                    "aggregator_type='set' requires a defined self.encoder, but it is None."
                )

            # Forward pass through the Transformer
            if return_self_attn:
                meal_attn_out, self_attn = self.encoder(
                    meal_token_emb,
                    src_key_padding_mask=pad_mask,
                    return_attn=True
                )
                # self_attn shape => (B*T, M+1, M+1)
            else:
                meal_attn_out = self.encoder(
                    meal_token_emb,
                    src_key_padding_mask=pad_mask,
                    return_attn=False
                )
                self_attn = None

            # Use the aggregator token's output as the "meal embedding"
            meal_timestep_emb = meal_attn_out[:, 0, :]  # => (B*T, hidden_dim)
            meal_timestep_emb = meal_timestep_emb.view(B, T, self.hidden_dim)

            # If returning attention, reshape it for convenience:
            # from (B*T, M+1, M+1) -> (B, T, M+1, M+1)
            if return_self_attn and self_attn is not None:
                self_attn = self_attn.view(B, T, self_attn.size(-2), self_attn.size(-1))
                return meal_timestep_emb, self_attn
            else:
                return meal_timestep_emb

        else:
            raise ValueError(f"Invalid aggregator_type='{self.aggregator_type}'. Use 'set' or 'sum' only.")

    def bootstrap_food_id_embeddings(self, bootstrap_food_id_embeddings: nn.Embedding):
        # Bootstrap the food id embeddings to pre-computed ones
        if bootstrap_food_id_embeddings is not None:
            with torch.no_grad():
                logger.warning(f"Bootstrapping food id embeddings with {self.food_embed_dim} dimensions of pre-computed embeddings")
                self.food_emb.weight.copy_(bootstrap_food_id_embeddings.weight[:, :self.food_embed_dim])
            self.food_emb.weight.requires_grad = False  # Freeze the food id embeddings
            logger.warning(f"Food id embeddings have been frozen")

# -----------------------------------------------------------------------------
# Glucose Encoder
# -----------------------------------------------------------------------------
class PatchedGlucoseEncoder(nn.Module):
    def __init__(
        self, embed_dim: int, patch_size: int, patch_stride: int, num_heads: int = 4, num_layers: int = 1, max_seq_len: int = 100, dropout_rate: float = 0.2
    ):
        super(PatchedGlucoseEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        # Simple linear projection from each patch to embed_dim
        self.patch_proj = nn.Linear(patch_size, embed_dim)

        # Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout_rate,
            batch_first=True,
        )
        # Encoder
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, glucose_seq: torch.Tensor) -> torch.Tensor:
        glucose_seq = glucose_seq.to(next(self.parameters()).dtype)
        B, T = glucose_seq.size()
        
        # 1) Patchify using unfold for efficient sliding window
        # shape => (B, N_patches, patch_size)
        patches = glucose_seq.unfold(1, self.patch_size, self.patch_stride)
        
        # Handle edge case where T < patch_size
        if patches.size(1) == 0:
            patches = F.pad(glucose_seq, (0, self.patch_size - T))[:, None, :]
        
        # Get patch indices for positional embeddings
        patch_indices = torch.arange(0, T - self.patch_size + 1, self.patch_stride, device=glucose_seq.device)
        if patch_indices.size(0) == 0:
            patch_indices = torch.tensor([0], device=glucose_seq.device)
        
        # 2) Project each patch into embed_dim
        # shape => (B*N_patches, patch_size)
        B_np, N_patches, _ = patches.shape
        patches = patches.reshape(B_np * N_patches, self.patch_size)  # Changed from view to reshape
        
        # linear projection
        patch_emb = self.patch_proj(patches)  # (B*N_patches, embed_dim)
        patch_emb = self.dropout(patch_emb)
        patch_emb = patch_emb.view(B_np, N_patches, -1)  # => (B, N_patches, embed_dim)
        
        # Transformer on patches
        patch_emb = self.transformer(patch_emb)  # => (B, N_patches, embed_dim)
        
        return patch_emb, patch_indices

# -----------------------------------------------------------------------------
# MealGlucoseForecastModel with configurable dropout parameters
# -----------------------------------------------------------------------------
class MealGlucoseForecastModel(pl.LightningModule):
    def __init__(self, config: ExperimentConfig, num_foods: int, macro_dim: int):
        super().__init__()
        # Save hyperparameters correctly
        self.save_hyperparameters({
            'config': asdict(config),
            'num_foods': num_foods,
            'macro_dim': macro_dim
        })
        self.config = config
        self.num_foods = num_foods
        self.macro_dim = macro_dim
        self.forecast_horizon = config.prediction_length
        self.eval_window = config.eval_window if config.eval_window is not None else config.prediction_length

        self.residual_pred = config.residual_pred
        self.num_quantiles = config.num_quantiles
        self.loss_iauc_weight = config.loss_iauc_weight

        self.hidden_dim = config.hidden_dim
        self.optimizer_lr = config.optimizer_lr
        self.weight_decay = config.weight_decay
        self.gradient_clip_val = config.gradient_clip_val
        self.food_embedding_projection_batch_size = config.food_embedding_projection_batch_size
        self.disable_plots = config.disable_plots
        
        # 1) Meal Encoder
        self.meal_encoder = MealEncoder(
            food_embed_dim=config.food_embed_dim,
            food_embed_adapter_dim=config.food_embed_adapter_dim,
            hidden_dim=config.hidden_dim,
            num_foods=num_foods,
            macro_dim=macro_dim,
            max_meals=config.max_meals,
            num_heads=config.num_heads,
            num_layers=config.transformer_encoder_layers,
            dropout_rate=config.dropout_rate,
            transformer_dropout=config.transformer_dropout,
            bootstrap_food_id_embeddings=None, # This is initialized in the from_dataset method
            aggregator_type=config.meal_aggregator_type
        )

        # 2) Glucose Encoder (patch-based)
        self.glucose_encoder = PatchedGlucoseEncoder(
            embed_dim=config.hidden_dim,
            patch_size=config.patch_size,
            patch_stride=config.patch_stride,
            num_heads=config.num_heads,
            num_layers=config.transformer_encoder_layers,
            max_seq_len=config.min_encoder_length,
            dropout_rate=config.dropout_rate
        )

        # 3) Transformer Decoder
        dec_layer = TransformerDecoderLayerWithAttn(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 2,
            dropout=config.transformer_dropout,
            activation="relu"
        )
        self.decoder = TransformerDecoderWithAttn(
            dec_layer,
            num_layers=config.transformer_decoder_layers,
            norm=nn.LayerNorm(config.hidden_dim)
        )

        # Query embeddings for each forecast horizon step
        self.future_time_queries = nn.Embedding(config.prediction_length, config.hidden_dim)

        # Final projection: hidden_dim -> num_quantiles
        self.forecast_linear = nn.Linear(config.hidden_dim, config.num_quantiles)

        # Register the quantiles as a buffer
        self.register_buffer("quantiles", torch.linspace(0.05, 0.95, steps=config.num_quantiles))
        
        # global time-based positional embedding
        max_time = config.min_encoder_length + config.prediction_length + 2000  # a safe upper bound
        self.time_emb = nn.Embedding(max_time, config.hidden_dim)
        

    def forward(
        self,
        past_glucose,        # [B, T_glucose]
        past_meal_ids,       # [B, T_pastMeal, M]
        past_meal_macros,    # [B, T_pastMeal, M, macro_dim]
        future_meal_ids,     # [B, T_futureMeal, M]
        future_meal_macros,   # [B, T_futureMeal, M, macro_dim]
        target_scales,        # [B, 2] for unscale
        return_attn: bool = False,
        return_meal_self_attn: bool = False
    ):
        """
        Returns either a single tensor [B, T_future, Q]
        or a tuple if return_attn=True: 
          (pred_future, past_meal_enc, attn_past, future_meal_enc, attn_future, meal_self_attn_past, meal_self_attn_future)
        """
        device = past_glucose.device
        B = past_glucose.size(0)
        
        # 1) Encode glucose => shape [B, G_patches, hidden_dim]
        glucose_enc, patch_indices = self.glucose_encoder(past_glucose)
        n_glucose_patches = glucose_enc.size(1)  # how many patches
        
        # glucose_enc => [B, n_glucose_patches, hidden_dim]
        # We'll broadcast the patch_indices shape => [B, n_patches], do the same embed
        glucose_enc = glucose_enc + self.time_emb(patch_indices) # Postional Embedding
        
        
        # 2) Encode past & future meals => [B, T_pastMeal, hidden_dim], [B, T_futureMeal, hidden_dim]
        if return_meal_self_attn:
            past_meal_enc, meal_self_attn_past = self.meal_encoder(
                past_meal_ids, past_meal_macros, return_self_attn=True
            )
            future_meal_enc, meal_self_attn_future = self.meal_encoder(
                future_meal_ids, future_meal_macros, return_self_attn=True
            )
        else:
            past_meal_enc = self.meal_encoder(past_meal_ids, past_meal_macros)
            future_meal_enc = self.meal_encoder(future_meal_ids, future_meal_macros)
            meal_self_attn_past = None
            meal_self_attn_future = None

        T_past = past_meal_enc.size(1)
        T_future = future_meal_enc.size(1)

        # 3) Build time indices for each block
        # We'll place them contiguously in time:
        #   Past meals:    t in [0 ... T_past-1]
        #   Future meals:  t in [T_past ... T_past+T_future-1]
        #   Glucose patch: t in [T_past+T_future ... T_past+T_future + n_glucose_patches - 1]
        #   (We won't add the queries here; see below.)

        # a) Past meal time
        past_indices = torch.arange(T_past, device=device).unsqueeze(0).expand(B, -1)
        # b) Future meal time
        future_indices = torch.arange(T_future, device=device).unsqueeze(0).expand(B, -1) + T_past
        

        # Add time embeddings
        # shape => [B, T_past, hidden_dim]
        past_meal_enc = past_meal_enc + self.time_emb(past_indices)
        future_meal_enc = future_meal_enc + self.time_emb(future_indices)


        ########################################################################
        ## MEMORY
        ########################################################################
        
        # 4) Combine them in a single "memory" => shape [B, (T_past + T_future + G_patches), hidden_dim]
        memory = torch.cat([past_meal_enc, future_meal_enc, glucose_enc], dim=1)

        # 5) Prepare queries for each forecast horizon step
        # Instead of self.future_time_queries, let's also unify them with the same time_emb
        # But if you prefer, you can still keep future_time_queries. 
        # For full unification, we can do:
        #   queries at time [T_past+T_future + G_patches ... T_past+T_future + G_patches + forecast_horizon - 1]


        # So t_future_idx in [0..forecast_horizon-1], but let's offset it:
        t_future_indices = torch.arange(self.forecast_horizon, device=device).unsqueeze(0).expand(B, -1)
        t_fufure_global_indices = t_future_indices + T_past

        # Now we can build queries from scratch:
        # shape => [B, forecast_horizon, hidden_dim]
        query_emb = self.future_time_queries.weight[t_future_indices]  # or gather
        # or we can do something like:
        # query_emb = torch.zeros(B, self.forecast_horizon, self.hidden_dim, device=device)
        # query_emb += self.time_emb(t_future_idx)
        # etc., depending on how you prefer.

        # If you want to add the same time_emb to the existing future_time_queries:
        query_emb = self.future_time_queries(t_future_indices)  # [B, T_future, hidden_dim]
        query_emb = query_emb + self.time_emb(t_fufure_global_indices)

        # 6) Decode
        if return_attn:
            decoder_output, cross_attn = self.decoder(
                tgt=query_emb,
                memory=memory,
                return_attn=True
            )
        else:
            decoder_output, cross_attn = self.decoder(
                tgt=query_emb,
                memory=memory,
                return_attn=False
            )

        # cross_attn => [B, T_future, T_mem]

        # 7) Slice out cross-attn for past vs future
        attn_past = None
        attn_future = None
        if cross_attn is not None:
            past_len = past_meal_enc.shape[1]
            future_len = future_meal_enc.shape[1]
            # memory = [past(0..past_len-1), future(past_len..past_len+future_len-1), glucose(...)]
            attn_past = cross_attn[:, :, :past_len]  # => [B, T_future, past_len]
            attn_future = cross_attn[:, :, past_len : past_len + future_len]

        # 8) Final projection => [B, T_future, num_quantiles]
        pred_future = self.forecast_linear(decoder_output)

        # Residual
        if self.residual_pred:
            last_val = past_glucose[:, -1].unsqueeze(1).unsqueeze(-1)
            pred_future = pred_future + last_val

        # Unscale
        pred_future = unscale_tensor(pred_future, target_scales)

        if return_attn:
            return (
                pred_future,
                past_meal_enc, attn_past,
                future_meal_enc, attn_future,
                meal_self_attn_past, meal_self_attn_future
            )
        else:
            return pred_future

    def _shared_step(self, batch, batch_idx, phase: str):
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
        
        # Log metrics with phase prefix
        for key, value in metrics["metrics"].items():
            self.log(f"{phase}_{key}", value, on_step=True, on_epoch=True, prog_bar=True)
        
        # Store outputs for phase end
        if not hasattr(self, f"{phase}_outputs"):
            setattr(self, f"{phase}_outputs", [])
        getattr(self, f"{phase}_outputs").append({
            f"{phase}_q_loss": metrics["metrics"]["q_loss"],
            f"{phase}_pred_iAUC_{self.eval_window}": metrics[f"pred_iAUC_{self.eval_window}"],
            f"{phase}_true_iAUC_{self.eval_window}": metrics[f"true_iAUC_{self.eval_window}"],
        })
        
        # Store example data for plotting
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

    def _shared_phase_end(self, phase: str):
        outputs = getattr(self, f"{phase}_outputs", [])
        if len(outputs) == 0:
            return

        if (not self.disable_plots) and (self.example_forecasts is not None):
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
        
        all_pred_iAUC = torch.cat([output[f"{phase}_pred_iAUC_{self.eval_window}"] for output in outputs], dim=0)
        all_true_iAUC = torch.cat([output[f"{phase}_true_iAUC_{self.eval_window}"] for output in outputs], dim=0)
        fig_scatter, corr = plot_iAUC_scatter(all_pred_iAUC, all_true_iAUC)
        self.logger.experiment.log({
            f"{phase}_iAUC_eh{self.eval_window}_scatter": wandb.Image(fig_scatter),
            f"{phase}_iAUC_eh{self.eval_window}_correlation": corr.item(),
            "global_step": self.global_step
        })
        plt.close(fig_scatter)
        setattr(self, f"{phase}_outputs", [])

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def on_validation_epoch_end(self):
        self._shared_phase_end("val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "test")

    def on_test_end(self):
        self._shared_phase_end("test")

    def training_step(self, batch, batch_idx):
        (past_glucose, past_meal_ids, past_meal_macros,
         future_meal_ids, future_meal_macros, future_glucose, target_scales) = batch
        
        # Forward pass
        preds = self(
            past_glucose,
            past_meal_ids,
            past_meal_macros,
            future_meal_ids,
            future_meal_macros,
            target_scales
        )
        
        # Compute metrics
        metrics = self._compute_forecast_metrics(past_glucose, future_glucose, target_scales, preds)
        
        # Log metrics
        for key in metrics["metrics"]:
            self.log(f"train_step_{key}", metrics["metrics"][key], on_step=True, on_epoch=True, prog_bar=True)
            
        return metrics["metrics"]["total_loss"]

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
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.optimizer_lr,
            weight_decay=self.weight_decay
        )
        
        return optimizer

    def on_train_epoch_end(self):
        pass

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, config: ExperimentConfig, num_foods: int, macro_dim: int, **kwargs):
        return super().load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            config=config,
            num_foods=num_foods,
            macro_dim=macro_dim,
            **kwargs
        )

    @classmethod
    def from_dataset(cls, dataset, config: ExperimentConfig):
        """
        Create a MealGlucoseForecastModel instance from a dataset and config.
        """
        model = cls(
            config=config,
            num_foods=dataset.num_foods,
            macro_dim=dataset.num_nutrients,
        )
        
        # If you want to bootstrap with pretrained embeddings:
        if config.use_bootstraped_food_embeddings:
            pretrained_weights = dataset.get_food_id_embeddings()  # returns a Tensor
            model.meal_encoder.bootstrap_food_id_embeddings(pretrained_weights)
        return model

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
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        num_workers=config.dataloader_num_workers, 
        pin_memory=True, 
        persistent_workers=True)
    
    return train_loader, val_loader, test_loader, training_dataset

def get_trainer(config: ExperimentConfig, callbacks):    
    wandb_logger = WandbLogger(
        project=config.wandb_project,
        name=config.wandb_run_name,
        config=asdict(config),
        log_model=True,
    )
    precision_value = int(config.precision) if config.precision == "32" else "bf16"
    trainer = pl.Trainer(
        profiler="simple",
        max_epochs=config.max_epochs,
        enable_checkpointing=True,  # Enable checkpointing
        logger=wandb_logger,
        callbacks=callbacks,
        precision=precision_value,
        gradient_clip_val=config.gradient_clip_val
    )
    return trainer


def prepare_callbacks(config: ExperimentConfig, model: MealGlucoseForecastModel, train_loader: DataLoader):
    optimizer = model.configure_optimizers()
    
    rich_model_summary = RichModelSummary(max_depth=2)
    rich_progress_bar = RichProgressBar()
    
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = os.path.join(config.checkpoint_base_dir, config.wandb_run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Add ModelCheckpoint callback with configurable parameters
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor=config.checkpoint_monitor,
        mode=config.checkpoint_mode,
        save_top_k=config.checkpoint_top_k,
        filename="best-{epoch}-{val_q_loss:.2f}",
        save_last=True
    )    
    
    lr_scheduler = LRSchedulerCallback(
        optimizer=optimizer,
        base_lr=config.optimizer_lr,
        total_steps=config.max_epochs * len(train_loader),
        pct_start=config.optimizer_lr_scheduler_pct_start,
    )
    callbacks = [rich_model_summary, rich_progress_bar, lr_scheduler, checkpoint_callback]
    return callbacks

# -----------------------------------------------------------------------------
# Main Training & Evaluation Entry Point
# -----------------------------------------------------------------------------
@click.command()
@create_click_options(ExperimentConfig)
def main(**kwargs):
    logging.getLogger().setLevel(logging.DEBUG if kwargs["debug_mode"] else logging.INFO)
    config = ExperimentConfig(**kwargs)
    
    # Generate experiment name based on modified parameters
    experiment_name = generate_experiment_name(config, kwargs)
    config.wandb_run_name = experiment_name
    
    logger.info(f"Starting experiment: {experiment_name}")
    
    if config.debug_mode:
        config.dataloader_num_workers = 1  # for debugging

    train_loader, val_loader, test_loader, training_dataset = get_dataloaders(config)
    
    model = MealGlucoseForecastModel.from_dataset(training_dataset, config)
    
    # Compile the model
    callbacks = prepare_callbacks(config, model, train_loader)
    trainer = get_trainer(config, callbacks)
    logging.info("Starting training.")
    
    trainer.fit(model, train_loader, val_loader)
    logging.info("Training complete.")
    
    # Load best model checkpoint
    best_model_path = trainer.checkpoint_callback.best_model_path
    if best_model_path:
        logging.info(f"Loading best model from: {best_model_path}")
        model = MealGlucoseForecastModel.load_from_checkpoint(
            best_model_path,
            config=config,
            num_foods=training_dataset.num_foods,
            macro_dim=training_dataset.num_nutrients
        )
    
    # Test phase
    logging.info("Starting testing.")
    # Evaluate on test set
    test_results = trainer.test(model, test_loader)
    logging.info("Test results: %s", test_results)
    
    # Detailed evaluation similar to validation
    model.eval()
    with torch.no_grad():
        batch = next(iter(test_loader))
        batch = [x.to(model.device) for x in batch]
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
        
    logging.info("Test set predictions (first sample): %s", pred_future[0].cpu().numpy())
    logging.info("Test set actual values (first sample): %s", future_glucose[0].cpu().numpy())
    logging.info("Test set past cross-attention shape: %s", attn_past.shape)
    logging.info("Test set future cross-attention shape: %s", attn_future.shape)
    logging.info("Test set meal self-attention (past) shape: %s", meal_self_attn_past.shape)
    logging.info("Test set meal self-attention (future) shape: %s", meal_self_attn_future.shape)
    
if __name__ == "__main__":
    main()
