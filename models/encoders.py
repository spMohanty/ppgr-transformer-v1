"""
Encoders for meal and glucose data.
"""
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from .transformer_blocks import TransformerEncoderLayer, TransformerEncoder

from pytorch_forecasting.models.nn import MultiEmbedding

class UserEncoder(nn.Module):
    """
    Simple MLP-based user encoder for static features.
    
    Args:
        categorical_variable_sizes: Dictionary mapping categorical variable names to their vocabulary sizes
        real_variables: List of names for real-valued variables
        hidden_dim: Dimension of the output embeddings
        dropout_rate: Dropout rate for regularization
        use_batch_norm: Whether to apply batch normalization to real variables
    """
    def __init__(
        self,
        categorical_variable_sizes: Dict[str, int],
        real_variables: List[str],
        hidden_dim: int,
        output_dim: int,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True
    ):
        super().__init__()
        
        self.categorical_variable_sizes = categorical_variable_sizes
        self.real_variables = real_variables
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_cat_features = len(categorical_variable_sizes)
        self.num_real_features = len(real_variables)
        
        # Individual embeddings for each categorical feature
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(size, hidden_dim)
            for size in categorical_variable_sizes.values()
        ])
        
        # Simple processing for real variables
        self.real_projection = nn.Linear(self.num_real_features, hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity()
        
        # Simple feed-forward network for final projection
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * (self.num_cat_features + 1), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, user_categoricals: torch.Tensor, user_reals: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the user encoder.
        
        Args:
            user_categoricals: Tensor of shape (batch_size, num_categorical_features)
            user_reals: Tensor of shape (batch_size, num_real_features)
            
        Returns:
            user_embeddings: Tensor of shape (batch_size, hidden_dim)
        """
        batch_size = user_categoricals.size(0)
        device = user_categoricals.device
        
        # Process categorical variables
        cat_embeddings_list = []
        for i, embedding_layer in enumerate(self.cat_embeddings):
            cat_feature = user_categoricals[:, i].long()
            cat_embeddings_list.append(embedding_layer(cat_feature))
        
        # Process real variables
        if self.num_real_features > 0:
            real_values = user_reals.float()  # [batch_size, num_reals]
            real_embeddings = self.real_projection(real_values)
            real_embeddings = self.batch_norm(real_embeddings)
        else:
            real_embeddings = torch.zeros(batch_size, self.hidden_dim, device=device)
        
        # Combine all embeddings
        all_embeddings = cat_embeddings_list + [real_embeddings]
        combined = torch.cat(all_embeddings, dim=1)
        
        # Project to final embedding
        user_embeddings = self.projection(combined)
        
        return self.dropout(user_embeddings)

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
        positional_embedding: nn.Module,  # Add positional embedding parameter
        max_meals: int = 11,
        num_heads: int = 4,
        num_layers: int = 1,
        layers_share_weights: bool = False,
        dropout_rate: float = 0.2,
        transformer_dropout: float = 0.1,
        aggregator_type: str = "set",  # Either "set" or "sum"
        ignore_food_macro_features: bool = False,
        bootstrap_food_id_embeddings: Optional[nn.Embedding] = None,
        freeze_food_id_embeddings: bool = True,
        add_residual_connection_before_meal_timestep_embedding: bool = True
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
        self.add_residual_connection_before_meal_timestep_embedding = add_residual_connection_before_meal_timestep_embedding
        self.positional_embedding = positional_embedding  # Store the positional embedding
        
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
        positions: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_self_attn: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode meals for each timestep.
        
        Args:
            meal_ids: (B, T, M) with each M being item indices in that meal
            meal_macros: (B, T, M, food_macro_dim) macro features for each item
            positions: (B, T) position indices for each timestep - important for distinguishing past/future meals
            mask: (B, T) boolean mask where True indicates valid timesteps
            return_self_attn: If True, returns self-attention weights (only works in "set" mode)
            
        Returns:
            meal_timestep_emb: (B, T, hidden_dim) aggregator encoding for each meal/time-step
            self_attn: Attention weights if requested and in "set" mode, else None
        """
        B, T, M = meal_ids.size()
        
        # Initialize output tensor with zeros
        meal_timestep_emb = torch.zeros(B, T, self.hidden_dim, device=meal_ids.device)
        self_attn_out = None
        
        # Reshape for easier processing
        meal_ids_flat = meal_ids.view(B * T, M)            # => (B*T, M)
        
        # Find timesteps with at least one non-zero meal_id (non-empty meals)
        has_meals = (meal_ids_flat != 0).any(dim=1)        # (B*T)
        
        # If all timesteps are empty, return zeros
        if not has_meals.any():
            return meal_timestep_emb, None
        
        # Get indices of non-empty timesteps
        non_empty_indices = has_meals.nonzero().squeeze(-1)  # (N_non_empty)
        
        # Filter to only include non-empty timesteps
        filtered_meal_ids = meal_ids_flat[non_empty_indices]  # (N_non_empty, M)
        filtered_meal_macros = meal_macros.view(B * T, M, -1)[non_empty_indices]  # (N_non_empty, M, food_macro_dim)
        
        # Get filtered mask for non-empty timesteps
        filtered_mask = None
        if mask is not None:
            flat_mask = mask.view(B * T)
            filtered_mask = flat_mask[non_empty_indices]  # (N_non_empty)
        
        # Get filtered positions for non-empty timesteps
        filtered_positions = None
        if positions is not None:
            flat_positions = positions.view(B * T)
            filtered_positions = flat_positions[non_empty_indices]  # (N_non_empty)
        
        # 1) Embed items only for non-empty timesteps
        food_emb = self.food_emb(filtered_meal_ids)              # (N_non_empty, M, food_embed_dim)
        food_emb = self.food_emb_proj(food_emb)                  # (N_non_empty, M, hidden_dim)
        food_emb = self.dropout(food_emb)
        
        if self.ignore_food_macro_features:
            # Scale food embeddings by weight (first column of macros)
            food_scaled_weight = filtered_meal_macros[:, :, 0].unsqueeze(-1)
            meal_token_emb = food_emb * food_scaled_weight
        else:
            macro_emb = self.macro_proj(filtered_meal_macros)    # (N_non_empty, M, hidden_dim)
            macro_emb = self.dropout(macro_emb)
            # Combine 
            meal_token_emb = food_emb + macro_emb  # shape = (N_non_empty, M, hidden_dim)

        # Process with the chosen aggregator strategy
        if self.aggregator_type == "sum":
            # Simple sum over items for non-empty timesteps
            filtered_meal_emb = meal_token_emb.sum(dim=1)  # (N_non_empty, hidden_dim)
            
        elif self.aggregator_type == "set":
            if self.add_residual_connection_before_meal_timestep_embedding:
                # Create simple average representation for residual connection
                # Mask out padding tokens (where meal_id is 0)
                item_mask = (filtered_meal_ids != 0).float().unsqueeze(-1)  # (N_non_empty, M, 1)
                avg_meal_emb = (meal_token_emb * item_mask).sum(dim=1) / (item_mask.sum(dim=1) + 1e-10)  # (N_non_empty, hidden_dim)
            
            # Insert aggregator token at position 0
            N_non_empty = meal_token_emb.size(0)
            start_token_expanded = self.start_token.expand(N_non_empty, -1, -1)  # => (N_non_empty, 1, hidden_dim)
            meal_token_emb_with_agg = torch.cat([start_token_expanded, meal_token_emb], dim=1)
            # => shape (N_non_empty, M+1, hidden_dim)

            # Build a padding mask: treat item_id=0 as padding
            pad_mask = (filtered_meal_ids == 0)  # shape (N_non_empty, M)
            zero_col = torch.zeros(N_non_empty, 1, dtype=torch.bool, device=pad_mask.device)
            pad_mask = torch.cat([zero_col, pad_mask], dim=1)  # => (N_non_empty, M+1)
            
            # Incorporate timestep mask if provided (for non-empty timesteps)
            if filtered_mask is not None:
                # Expand to match the meal items + aggregator token
                filtered_timestep_mask = filtered_mask.unsqueeze(1).expand(-1, pad_mask.size(1))  # (N_non_empty, M+1)
                
                # Combine with padding mask
                timestep_mask_combined = ~filtered_timestep_mask  # True means positions to ignore
                timestep_mask_combined[:, 0] = False  # Keep aggregator token valid
                pad_mask = pad_mask | timestep_mask_combined
            
            # Generate position indices for each token in each meal
            if filtered_positions is not None:
                # Use provided positions - each meal token gets the timestep position
                # This expands the timestep position [N_non_empty] to [N_non_empty, M+1]
                # so all tokens (including aggregator) in same meal get same position
                positions = filtered_positions.unsqueeze(1).expand(-1, meal_token_emb_with_agg.size(1))
            else:
                # Fall back to using non-empty indices as positions
                positions = non_empty_indices.unsqueeze(1).expand(-1, meal_token_emb_with_agg.size(1))
            
            # Apply positional embeddings - ALL tokens in the same meal get the same position
            meal_token_emb_with_agg = self.positional_embedding(meal_token_emb_with_agg, positions)

            # Forward pass through the Transformer
            meal_attn_out, attn_weights = self.encoder(
                meal_token_emb_with_agg,
                src_key_padding_mask=pad_mask,
                need_weights=return_self_attn
            )

            # Use the aggregator token's output as the "meal embedding"
            transformer_meal_emb = meal_attn_out[:, 0, :]  # => (N_non_empty, hidden_dim)
            
            # Apply residual connection: combine transformer output with average embedding
            filtered_meal_emb = transformer_meal_emb
            if self.add_residual_connection_before_meal_timestep_embedding:
                filtered_meal_emb += avg_meal_emb  # (N_non_empty, hidden_dim)
            
            # Process attention weights if needed
            if attn_weights is not None and return_self_attn:
                # Create a full attention tensor filled with zeros
                full_self_attn = torch.zeros(
                    B * T, attn_weights.size(-2), attn_weights.size(-1), 
                    device=attn_weights.device
                )
                
                # Place the computed attention weights in the right positions
                full_self_attn[non_empty_indices] = attn_weights
                
                # Reshape to (B, T, M+1, M+1)
                self_attn_out = full_self_attn.view(B, T, attn_weights.size(-2), attn_weights.size(-1))
        
        else:
            raise ValueError(f"Invalid aggregator_type='{self.aggregator_type}'. Use 'set' or 'sum' only.")

        # Apply filtered mask if provided
        if filtered_mask is not None:
            filtered_meal_emb = filtered_meal_emb * filtered_mask.unsqueeze(-1).float()
        
        # Place embeddings back in the full tensor
        meal_timestep_emb_flat = torch.zeros(B * T, self.hidden_dim, device=meal_ids.device)
        meal_timestep_emb_flat[non_empty_indices] = filtered_meal_emb
        meal_timestep_emb = meal_timestep_emb_flat.view(B, T, self.hidden_dim)
        
        return meal_timestep_emb, self_attn_out

    def _bootstrap_food_id_embeddings(self, bootstrap_food_id_embeddings: nn.Embedding, freeze_embeddings: bool = True):
        """Bootstrap the food id embeddings from pre-computed ones."""
        if bootstrap_food_id_embeddings is not None:
            with torch.no_grad():
                logger.warning(f"Bootstrapping food id embeddings with {self.food_embed_dim} dimensions of pre-computed embeddings")
                self.food_emb.weight.copy_(bootstrap_food_id_embeddings.weight[:, :self.food_embed_dim])
            if freeze_embeddings:
                self.food_emb.weight.requires_grad = not freeze_embeddings
                logger.warning(f"Food id embeddings have been frozen")


class GlucosePatcher(nn.Module):
    """
    Module that handles patching of 1D glucose time series data.
    This module has no trainable parameters and simply transforms the data.
    """
    def __init__(self, patch_size: int, patch_stride: int):
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        
    def forward(self, glucose_seq: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Transform glucose time series into patches.
        
        Args:
            glucose_seq: (B, T) time series of glucose values
            mask: (B, T) boolean mask where True indicates valid data points
            
        Returns:
            patches: (B, N_patches, patch_size) patched data
            patch_indices: (N_patches,) indices where each patch starts
            patch_mask: (B, N_patches) boolean mask for valid patches
        """
        glucose_seq = glucose_seq.float()  # Ensure float type
        B, T = glucose_seq.size()
        
        # Create patches using unfold
        patches = glucose_seq.unfold(1, self.patch_size, self.patch_stride)
        
        # Handle edge case where T < patch_size
        if patches.size(1) == 0:
            patches = F.pad(glucose_seq, (0, self.patch_size - T))[:, None, :]
        
        # Get dimensions
        _, N_patches, _ = patches.shape
        
        # Calculate patch indices
        patch_indices = torch.arange(0, T - self.patch_size + 1, self.patch_stride, device=glucose_seq.device)
        if patch_indices.size(0) == 0:
            patch_indices = torch.tensor([0], device=glucose_seq.device)
        
        # Handle masking if provided
        patch_mask = None
        if mask is not None:
            # Patchify the mask
            mask_patches = mask.unfold(1, self.patch_size, self.patch_stride)
            
            # Handle edge case where T < patch_size
            if mask_patches.size(1) == 0:
                mask_patches = F.pad(mask, (0, self.patch_size - T))[:, None, :]
            
            # A patch is valid only if at least one point in it is valid
            patch_mask = mask_patches.any(dim=2)  # [B, N_patches]
            
            # Apply mask to patches (zero out invalid points)
            patches = patches * mask_patches.float()
        
        return patches, patch_indices, patch_mask


class PatchEncoder(nn.Module):
    """
    Encodes pre-patched data using a transformer.
    """
    def __init__(
        self,
        patch_size: int,
        embed_dim: int,
        positional_embedding: nn.Module,
        num_heads: int = 4,
        num_layers: int = 1,
        layers_share_weights: bool = False,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.positional_embedding = positional_embedding
        
        # Linear projection from patches to embedding space
        self.patch_proj = nn.Linear(patch_size, embed_dim)
        
        # Encoder layer
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
    
    def forward(
        self, 
        patches: torch.Tensor, 
        positions: torch.Tensor,
        patch_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode patches into the embedding space.
        
        Args:
            patches: (B, N_patches, patch_size) patched data
            positions: (B, N_patches) position indices for each patch
            patch_mask: (B, N_patches) boolean mask where True indicates valid patches
            return_attn: If True, returns attention weights
            
        Returns:
            encodings: (B, N_patches, embed_dim) encoded patches
            attn_weights: Optional attention weights if requested
        """
        B, N_patches, _ = patches.shape
        
        # Reshape for linear projection
        flat_patches = patches.reshape(B * N_patches, self.patch_size)
        
        # Project patches to embedding dimension
        patch_emb = self.patch_proj(flat_patches)  # (B*N_patches, embed_dim)
        patch_emb = self.dropout(patch_emb)
        patch_emb = patch_emb.view(B, N_patches, self.embed_dim)  # (B, N_patches, embed_dim)
        
        # Apply positional embeddings
        patch_emb = self.positional_embedding(patch_emb, positions)
        
        # Create causal mask for transformer
        causal_mask = torch.triu(
            torch.full((N_patches, N_patches), float('-inf')), diagonal=1
        ).to(patches.device)
        
        # Create key padding mask if patch mask is provided
        key_padding_mask = None
        if patch_mask is not None:
            # In transformer, key_padding_mask=True means positions to IGNORE
            key_padding_mask = ~patch_mask
            # Convert to float to match mask type - this was missing
            key_padding_mask = key_padding_mask.float()
        
        # Pass through transformer
        encodings, attn_weights = self.transformer(
            patch_emb,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
            need_weights=return_attn
        )
        
        return encodings, attn_weights


class PatchedGlucoseEncoder(nn.Module):
    """
    Encodes glucose time series data using a patched approach with transformer encoder.
    This is the main interface that composes the GlucosePatcher and PatchEncoder.
    """
    def __init__(
        self, 
        embed_dim: int, 
        patch_size: int, 
        patch_stride: int, 
        positional_embedding: nn.Module, 
        num_heads: int = 4, 
        num_layers: int = 1, 
        layers_share_weights: bool = False,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        
        # Create components
        self.patcher = GlucosePatcher(patch_size, patch_stride)
        self.encoder = PatchEncoder(
            patch_size=patch_size,
            embed_dim=embed_dim,
            positional_embedding=positional_embedding,
            num_heads=num_heads,
            num_layers=num_layers,
            layers_share_weights=layers_share_weights,
            dropout_rate=dropout_rate
        )
    
    def forward(
        self, 
        glucose_seq: torch.Tensor, 
        positions: Optional[torch.Tensor] = None, 
        mask: Optional[torch.Tensor] = None, 
        return_self_attn: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Encode glucose time series data.
        
        Args:
            glucose_seq: (B, T) time series of glucose values
            positions: (B, T) position indices for each time point
            mask: (B, T) boolean mask where True indicates valid data points
            return_self_attn: If True, returns self-attention weights
            
        Returns:
            encodings: (B, N_patches, embed_dim) encoded patches
            attn_weights: Optional attention weights if requested
            patch_indices: (N_patches,) indices where each patch starts
        """
        # Step 1: Create patches
        patches, patch_indices, patch_mask = self.patcher(glucose_seq, mask)
        
        # Step 2: Handle positions for patches
        B, N_patches, _ = patches.shape
        
        if positions is not None:
            # Simplify position handling - just use provided positions
            # Make sure dimensions match
            if positions.size(1) != N_patches:
                # Ensure positions matches the number of patches by interpolation
                if positions.size(1) > N_patches:
                    # Subsample positions if we have too many
                    indices = torch.linspace(0, positions.size(1)-1, N_patches, device=positions.device).long()
                    patch_positions = positions[:, indices]
                else:
                    # Upsample positions if we have too few
                    # Just repeat the last position for any missing positions
                    last_pos = positions[:, -1:].expand(-1, N_patches - positions.size(1))
                    patch_positions = torch.cat([positions, last_pos], dim=1)
            else:
                patch_positions = positions
        else:
            # If no positions provided, use patch indices directly
            patch_positions = patch_indices.unsqueeze(0).expand(B, -1)
        
        # Step 3: Encode patches
        encodings, attn_weights = self.encoder(
            patches=patches,
            positions=patch_positions,
            patch_mask=patch_mask,
            return_attn=return_self_attn
        )
        
        return encodings, attn_weights, patch_indices

class TemporalEncoder(nn.Module):
    """
    Encodes temporal features (both categorical and real-valued) at each timestep.
    
    Args:
        categorical_variable_sizes: Dictionary mapping categorical variable names to their vocabulary sizes
        real_variables: List of names for real-valued variables
        hidden_dim: Dimension of the output embeddings
        dropout_rate: Dropout rate for regularization
        use_batch_norm: Whether to apply batch normalization to real variables
        positional_embedding: Optional positional embedding module
    """
    def __init__(
        self,
        categorical_variable_sizes: Dict[str, int],
        real_variables: List[str],
        hidden_dim: int,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        positional_embedding: Optional[nn.Module] = None
    ):
        super().__init__()
        
        self.categorical_variable_sizes = categorical_variable_sizes
        self.real_variables = real_variables
        self.hidden_dim = hidden_dim
        self.num_cat_features = len(categorical_variable_sizes)
        self.num_real_features = len(real_variables)
        self.positional_embedding = positional_embedding
        
        # Individual embeddings for each categorical feature
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(size, hidden_dim)
            for size in categorical_variable_sizes.values()
        ])
        
        # Simple processing for real variables
        self.real_projection = nn.Linear(self.num_real_features, hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity()
        
        # Simple feed-forward network for final projection
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * (self.num_cat_features + 1), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, 
                temporal_categoricals: Optional[torch.Tensor] = None, 
                temporal_reals: Optional[torch.Tensor] = None,
                positions: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None
               ) -> torch.Tensor:
        """
        Forward pass of the temporal encoder.
        
        Args:
            temporal_categoricals: Tensor of shape (batch_size, seq_length, num_categorical_features) or None
            temporal_reals: Tensor of shape (batch_size, seq_length, num_real_features) or None
            positions: Optional tensor of shape (batch_size, seq_length) with position indices
            mask: Optional mask of shape (batch_size, seq_length) where True indicates valid timesteps
            
        Returns:
            temporal_embeddings: Tensor of shape (batch_size, seq_length, hidden_dim)
        """
        # Check if at least one input is provided
        if temporal_categoricals is None and temporal_reals is None:
            raise ValueError("Both temporal_categoricals and temporal_reals cannot be None")
        
        # Determine batch_size and seq_length from non-None input
        if temporal_categoricals is not None:
            batch_size, seq_length, _ = temporal_categoricals.size()
            device = temporal_categoricals.device
        else:
            batch_size, seq_length, _ = temporal_reals.size()
            device = temporal_reals.device
        
        # Process categorical variables
        cat_embeddings_list = []
        if temporal_categoricals is not None:
            # Reshape to process all timesteps at once
            cat_flat = temporal_categoricals.reshape(batch_size * seq_length, -1)
            
            for i, embedding_layer in enumerate(self.cat_embeddings):
                cat_feature = cat_flat[:, i].long()
                cat_embeddings_list.append(embedding_layer(cat_feature))
        
        # Process real variables
        if self.num_real_features > 0 and temporal_reals is not None:
            real_flat = temporal_reals.reshape(batch_size * seq_length, -1)
            real_values = real_flat.float()  # [batch_size * seq_length, num_reals]
            real_embeddings = self.real_projection(real_values)
            
            # For batch norm, reshape to [batch_size * seq_length, hidden_dim]
            if isinstance(self.batch_norm, nn.BatchNorm1d):
                real_embeddings = self.batch_norm(real_embeddings)
        else:
            real_embeddings = torch.zeros(batch_size * seq_length, self.hidden_dim, device=device)
        
        # Combine all embeddings
        if not cat_embeddings_list:
            # If no categorical embeddings, just use real embeddings
            combined = real_embeddings
        else:
            # Otherwise combine categorical and real embeddings
            all_embeddings = cat_embeddings_list + [real_embeddings]
            combined = torch.cat(all_embeddings, dim=1)
        
        # Project to final embedding
        temporal_embeddings_flat = self.projection(combined)
        temporal_embeddings_flat = self.dropout(temporal_embeddings_flat)
        
        # Reshape back to sequence form
        temporal_embeddings = temporal_embeddings_flat.reshape(batch_size, seq_length, self.hidden_dim)
        
        # Apply positional embeddings if provided
        if self.positional_embedding is not None and positions is not None:
            temporal_embeddings = self.positional_embedding(temporal_embeddings, positions)
        
        # Apply mask if provided
        if mask is not None:
            temporal_embeddings = temporal_embeddings * mask.unsqueeze(-1).float()
        
        return temporal_embeddings