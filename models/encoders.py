"""
Encoders for meal and glucose data.
"""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from .transformer_blocks import TransformerEncoderLayer, TransformerEncoder


class MealEncoder(nn.Module):
    """
    Encodes meal data either using a simple sum aggregation or a self-attention based approach.
    """
    def __init__(
        self,
        food_embed_dim: int,
        hidden_dim: int,
        num_foods: int,
        food_macro_dim: int,
        food_names: List[str],
        food_group_names: List[str],
        max_meals: int = 11,
        num_heads: int = 4,
        num_layers: int = 1,
        layers_share_weights: bool = False,
        dropout_rate: float = 0.2,
        transformer_dropout: float = 0.1,
        aggregator_type: str = "set",  # Either "set" or "sum"
        ignore_food_macro_features: bool = False,
        bootstrap_food_id_embeddings: Optional[nn.Embedding] = None,
        freeze_food_id_embeddings: bool = True
    ):
        super().__init__()
        self.food_embed_dim = food_embed_dim
        self.hidden_dim = hidden_dim
        self.max_meals = max_meals
        self.num_foods = num_foods
        self.food_macro_dim = food_macro_dim
        self.food_names = food_names
        self.food_group_names = food_group_names
        self.aggregator_type = aggregator_type.lower().strip()
        self.ignore_food_macro_features = ignore_food_macro_features

        # --------------------
        #  Embedding Layers
        # --------------------
        self.food_emb = nn.Embedding(num_foods, food_embed_dim, padding_idx=0)
        self.food_emb_proj = nn.Linear(food_embed_dim, hidden_dim)
        
        if not self.ignore_food_macro_features:
            # We need the macro projection only if we are using all the macro features
            self.macro_proj = nn.Linear(food_macro_dim, hidden_dim, bias=False)

        # Optional aggregator token to collect representations (only used in "set" mode).
        self.start_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # Bootstrapping pre-trained food embeddings if provided
        self._bootstrap_food_id_embeddings(bootstrap_food_id_embeddings, freeze_embeddings=freeze_food_id_embeddings)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

        # Transformer stack (only for aggregator_type="set")
        if self.aggregator_type == "set":
            enc_layer = TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=transformer_dropout,
                activation="relu"
            )
            self.encoder = TransformerEncoder(
                enc_layer,
                num_layers=num_layers,
                norm=nn.LayerNorm(hidden_dim),
                layers_share_weights=layers_share_weights
            )
        else:
            self.encoder = None  # No Transformer for aggregator_type="sum"
            
    def forward(
        self,
        meal_ids: torch.LongTensor,
        meal_macros: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_self_attn: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode meals for each timestep.
        
        Args:
            meal_ids: (B, T, M) with each M being item indices in that meal
            meal_macros: (B, T, M, food_macro_dim) macro features for each item
            mask: (B, T) boolean mask where True indicates valid timesteps
            return_self_attn: If True, returns self-attention weights (only works in "set" mode)
            
        Returns:
            meal_timestep_emb: (B, T, hidden_dim) aggregator encoding for each meal/time-step
            self_attn: Attention weights if requested and in "set" mode, else None
        """
        B, T, M = meal_ids.size()

        # 1) Flatten out (B,T) -> (B*T) for item-level embeddings
        meal_ids_flat = meal_ids.view(B * T, M)            # => (B*T, M)
        meal_macros_flat = meal_macros.view(B * T, M, -1)  # => (B*T, M, food_macro_dim)

        # 2) Embed items
        food_emb = self.food_emb(meal_ids_flat)               # (B*T, M, food_embed_dim)
        food_emb = self.food_emb_proj(food_emb)               # (B*T, M, hidden_dim)
        food_emb = self.dropout(food_emb)
        
        if self.ignore_food_macro_features:
            # Scale food embeddings by weight (first column of macros)
            food_scaled_weight = meal_macros_flat[:, :, 0].unsqueeze(-1)
            meal_token_emb = food_emb * food_scaled_weight
        else:
            macro_emb = self.macro_proj(meal_macros_flat)     # (B*T, M, hidden_dim)
            macro_emb = self.dropout(macro_emb)
            # Combine 
            meal_token_emb = food_emb + macro_emb  # shape = (B*T, M, hidden_dim)
        
        # Process with the chosen aggregator strategy
        if self.aggregator_type == "sum":
            return self._process_sum_aggregator(meal_token_emb, B, T, mask)
        elif self.aggregator_type == "set":
            return self._process_set_aggregator(meal_token_emb, meal_ids_flat, B, T, mask, return_self_attn)
        else:
            raise ValueError(f"Invalid aggregator_type='{self.aggregator_type}'. Use 'set' or 'sum' only.")

    def _process_sum_aggregator(self, meal_token_emb, B, T, mask=None):
        """Process using simple summation aggregator."""
        # Simple sum over items
        meal_timestep_emb = meal_token_emb.sum(dim=1)  # (B*T, hidden_dim)
        meal_timestep_emb = meal_timestep_emb.view(B, T, self.hidden_dim)
        
        # Apply mask if provided
        if mask is not None:
            # Zero out invalid timesteps
            meal_timestep_emb = meal_timestep_emb * mask.unsqueeze(-1).float()
        
        return meal_timestep_emb, None

    def _process_set_aggregator(self, meal_token_emb, meal_ids_flat, B, T, mask=None, return_self_attn=False):
        """Process using transformer-based set aggregator."""
        if self.encoder is None:
            raise ValueError("aggregator_type='set' requires a defined self.encoder, but it is None.")
        
        # Find timesteps with at least one non-zero meal_id (non-empty meals)
        has_meals = (meal_ids_flat != 0).any(dim=1)  # (B*T)
        
        # If all timesteps are empty, return zeros
        if not has_meals.any():
            meal_timestep_emb = torch.zeros(B, T, self.hidden_dim, device=meal_token_emb.device)
            return meal_timestep_emb, None
        
        # Get indices of non-empty timesteps
        non_empty_indices = has_meals.nonzero().squeeze(-1)  # (N_non_empty)
        
        # Filter meal_token_emb and meal_ids_flat to only include non-empty timesteps
        filtered_meal_token_emb = meal_token_emb[non_empty_indices]  # (N_non_empty, M, hidden_dim)
        filtered_meal_ids = meal_ids_flat[non_empty_indices]  # (N_non_empty, M)
        
        # Insert aggregator token at position 0
        N_non_empty = filtered_meal_token_emb.size(0)
        start_token_expanded = self.start_token.expand(N_non_empty, -1, -1)  # => (N_non_empty, 1, hidden_dim)
        filtered_meal_token_emb = torch.cat([start_token_expanded, filtered_meal_token_emb], dim=1)
        # => shape (N_non_empty, M+1, hidden_dim)

        # Build a padding mask: treat item_id=0 as padding
        pad_mask = (filtered_meal_ids == 0)  # shape (N_non_empty, M)
        zero_col = torch.zeros(N_non_empty, 1, dtype=torch.bool, device=pad_mask.device)
        pad_mask = torch.cat([zero_col, pad_mask], dim=1)  # => (N_non_empty, M+1)
        
        # Incorporate timestep mask if provided (for non-empty timesteps)
        if mask is not None:
            # Get timestep mask for the non-empty timesteps
            flat_mask = mask.view(B * T)
            filtered_timestep_mask = flat_mask[non_empty_indices].unsqueeze(1)  # (N_non_empty, 1)
            
            # Expand to match the meal items + aggregator token
            filtered_timestep_mask = filtered_timestep_mask.expand(-1, pad_mask.size(1))  # (N_non_empty, M+1)
            
            # Combine with padding mask
            timestep_mask_combined = ~filtered_timestep_mask  # True means positions to ignore
            timestep_mask_combined[:, 0] = False  # Keep aggregator token valid
            pad_mask = pad_mask | timestep_mask_combined

        # Forward pass through the Transformer (only for non-empty timesteps)
        meal_attn_out, self_attn = self.encoder(
            filtered_meal_token_emb,
            src_key_padding_mask=pad_mask,
            need_weights=return_self_attn
        )

        # Use the aggregator token's output as the "meal embedding"
        filtered_meal_emb = meal_attn_out[:, 0, :]  # => (N_non_empty, hidden_dim)
        
        # Create the full meal embeddings tensor filled with zeros
        meal_timestep_emb = torch.zeros(B * T, self.hidden_dim, device=meal_token_emb.device)
        
        # Place the computed embeddings in the right positions
        meal_timestep_emb[non_empty_indices] = filtered_meal_emb
        
        # Reshape to (B, T, hidden_dim)
        meal_timestep_emb = meal_timestep_emb.view(B, T, self.hidden_dim)
        
        # Apply timestep mask if provided
        if mask is not None:
            meal_timestep_emb = meal_timestep_emb * mask.unsqueeze(-1).float()
        
        # Process attention weights if needed
        if self_attn is not None and return_self_attn:
            # Create a full attention tensor filled with zeros
            full_self_attn = torch.zeros(
                B * T, self_attn.size(-2), self_attn.size(-1), 
                device=self_attn.device
            )
            
            # Place the computed attention weights in the right positions
            full_self_attn[non_empty_indices] = self_attn
            
            # Reshape to (B, T, M+1, M+1)
            self_attn = full_self_attn.view(B, T, self_attn.size(-2), self_attn.size(-1))
        
        return meal_timestep_emb, self_attn

    def _bootstrap_food_id_embeddings(self, bootstrap_food_id_embeddings: nn.Embedding, freeze_embeddings: bool = True):
        """Bootstrap the food id embeddings from pre-computed ones."""
        if bootstrap_food_id_embeddings is not None:
            with torch.no_grad():
                logger.warning(f"Bootstrapping food id embeddings with {self.food_embed_dim} dimensions of pre-computed embeddings")
                self.food_emb.weight.copy_(bootstrap_food_id_embeddings.weight[:, :self.food_embed_dim])
            if freeze_embeddings:
                self.food_emb.weight.requires_grad = not freeze_embeddings
                logger.warning(f"Food id embeddings have been frozen")


class PatchedGlucoseEncoder(nn.Module):
    """
    Encodes glucose time series data using a patched approach with transformer encoder.
    """
    def __init__(
        self, 
        embed_dim: int, 
        patch_size: int, 
        patch_stride: int, 
        num_heads: int = 4, 
        num_layers: int = 1, 
        layers_share_weights: bool = False,
        max_seq_len: int = 100, 
        dropout_rate: float = 0.2
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        # Simple linear projection from each patch to embed_dim
        self.patch_proj = nn.Linear(patch_size, embed_dim)

        # Encoder Layer
        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout_rate,
            activation="relu"
        )
        # Encoder
        self.transformer = TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim),
            layers_share_weights=layers_share_weights
        )
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, glucose_seq: torch.Tensor, mask: Optional[torch.Tensor] = None, return_self_attn: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            glucose_seq: (B, T) time series of glucose values
            mask: (B, T) boolean mask where True indicates valid data points
            
        Returns:
            patch_emb: (B, N_patches, embed_dim) encoded patches
            patch_indices: (N_patches,) indices of patches for positional embeddings
        """
        glucose_seq = glucose_seq.to(next(self.parameters()).dtype)
        B, T = glucose_seq.size()
        
        # 1) Patchify using unfold for efficient sliding window
        # shape => (B, N_patches, patch_size)
        patches = glucose_seq.unfold(1, self.patch_size, self.patch_stride)
        
        # Handle edge case where T < patch_size
        if patches.size(1) == 0:
            patches = F.pad(glucose_seq, (0, self.patch_size - T))[:, None, :]
        
        # Extract dimensions right away to ensure N_patches is defined
        B_np, N_patches, _ = patches.shape
        
        # Get patch indices for positional embeddings
        patch_indices = torch.arange(0, T - self.patch_size + 1, self.patch_stride, device=glucose_seq.device)
        if patch_indices.size(0) == 0:
            patch_indices = torch.tensor([0], device=glucose_seq.device)
            
        # 2) Handle masking if provided
        key_padding_mask = None
        if mask is not None:
            # Patchify the mask
            mask_patches = mask.unfold(1, self.patch_size, self.patch_stride)
            
            # Handle edge case where T < patch_size
            if mask_patches.size(1) == 0:
                mask_patches = F.pad(mask, (0, self.patch_size - T))[:, None, :]
            
            # A patch is valid only if at least one point in it is valid
            patch_valid = mask_patches.any(dim=2)  # [B, N_patches]
            
            # Create key padding mask for transformer (True means positions to IGNORE)
            key_padding_mask = ~patch_valid  # [B, N_patches]
            
            # Apply mask to patches (zero out invalid points)
            patches = patches * mask_patches.float()
        
        # 3) Project each patch into embed_dim
        patches = patches.reshape(B_np * N_patches, self.patch_size)
        patch_emb = self.patch_proj(patches)  # (B*N_patches, embed_dim)
        patch_emb = self.dropout(patch_emb)
        patch_emb = patch_emb.view(B_np, N_patches, self.embed_dim)  # => (B, N_patches, embed_dim)
        
        # 4) Handle the masks
        # Create a standard causal mask (non-batch dimension)
        causal_mask = torch.triu(
            torch.full((N_patches, N_patches), float('-inf')), diagonal=1
        ).to(glucose_seq.device)
        
        # Pass only the causal mask - it will be broadcast to all batches automatically
        # This is key: mask parameter expects shape [seq_len, seq_len] not [batch, seq_len, seq_len]
        patch_emb, attn_weights = self.transformer(
            patch_emb,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask.float(),
            need_weights=return_self_attn
        )
        
        return patch_emb, attn_weights, patch_indices