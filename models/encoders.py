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
                norm=nn.LayerNorm(hidden_dim)
            )
        else:
            self.encoder = None  # No Transformer for aggregator_type="sum"
            
    def forward(
        self,
        meal_ids: torch.LongTensor,
        meal_macros: torch.Tensor,
        return_self_attn: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode meals for each timestep.
        
        Args:
            meal_ids: (B, T, M) with each M being item indices in that meal
            meal_macros: (B, T, M, food_macro_dim) macro features for each item
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
            return self._process_sum_aggregator(meal_token_emb, B, T)
        elif self.aggregator_type == "set":
            return self._process_set_aggregator(meal_token_emb, meal_ids_flat, B, T, return_self_attn)
        else:
            raise ValueError(f"Invalid aggregator_type='{self.aggregator_type}'. Use 'set' or 'sum' only.")

    def _process_sum_aggregator(self, meal_token_emb, B, T):
        """Process using simple summation aggregator."""
        # Simple sum over items
        meal_timestep_emb = meal_token_emb.sum(dim=1)
        meal_timestep_emb = meal_timestep_emb.view(B, T, self.hidden_dim)
        return meal_timestep_emb, None

    def _process_set_aggregator(self, meal_token_emb, meal_ids_flat, B, T, return_self_attn):
        """Process using transformer-based set aggregator."""
        if self.encoder is None:
            raise ValueError("aggregator_type='set' requires a defined self.encoder, but it is None.")
            
        # Insert aggregator token at position 0
        start_token_expanded = self.start_token.expand(B * T, -1, -1)  # => (B*T, 1, hidden_dim)
        meal_token_emb = torch.cat([start_token_expanded, meal_token_emb], dim=1)
        # => shape (B*T, M+1, hidden_dim)

        # Build a padding mask: treat item_id=0 as padding
        pad_mask = (meal_ids_flat == 0)  # shape (B*T, M)
        zero_col = torch.zeros(B * T, 1, dtype=torch.bool, device=pad_mask.device)
        pad_mask = torch.cat([zero_col, pad_mask], dim=1)  # => (B*T, M+1)

        # Forward pass through the Transformer
        meal_attn_out, self_attn = self.encoder(
            meal_token_emb,
            src_key_padding_mask=pad_mask,
            need_weights=return_self_attn
        )

        # Use the aggregator token's output as the "meal embedding"
        meal_timestep_emb = meal_attn_out[:, 0, :]  # => (B*T, hidden_dim)
        meal_timestep_emb = meal_timestep_emb.view(B, T, self.hidden_dim)
        
        if self_attn is not None:
            # Reshape the attention weights to (B, T, M+1, M+1)
            self_attn = self_attn.view(B, T, self_attn.size(-2), self_attn.size(-1))
            
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
        max_seq_len: int = 100, 
        add_causal_mask: bool = True, 
        dropout_rate: float = 0.2
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.add_causal_mask = add_causal_mask

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

    def forward(self, glucose_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            glucose_seq: (B, T) time series of glucose values
            
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
        
        # Get patch indices for positional embeddings
        patch_indices = torch.arange(0, T - self.patch_size + 1, self.patch_stride, device=glucose_seq.device)
        if patch_indices.size(0) == 0:
            patch_indices = torch.tensor([0], device=glucose_seq.device)
        
        # 2) Project each patch into embed_dim
        B_np, N_patches, _ = patches.shape
        patches = patches.reshape(B_np * N_patches, self.patch_size)
        
        # Linear projection
        patch_emb = self.patch_proj(patches)  # (B*N_patches, embed_dim)
        patch_emb = self.dropout(patch_emb)
        patch_emb = patch_emb.view(B_np, N_patches, -1)  # => (B, N_patches, embed_dim)
        
        # Create a causal mask if needed
        if self.add_causal_mask:
            # The mask shape is (N_patches, N_patches)
            causal_mask = torch.triu(
                torch.full((N_patches, N_patches), float('-inf')), diagonal=1
            ).to(glucose_seq.device)
        else:
            causal_mask = None
            
        # Transformer on patches
        patch_emb = self.transformer(patch_emb, mask=causal_mask)  # => (B, N_patches, embed_dim)
        
        return patch_emb, patch_indices