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


class MicrobiomeEncoder(nn.Module):
    """
    Encoder for microbiome features.
    
    Takes microbiome features and projects them to the hidden dimension
    using two linear layers with activation.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
            nn.Linear(hidden_dim, output_dim),
        )
        
    def forward(self, microbiome_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the microbiome encoder.
        
        Args:
            microbiome_features: Microbiome features [B, num_microbiome_features]
            
        Returns:
            Encoded microbiome features [B, output_dim]
        """
        # Ensure microbiome features have the same dtype as model parameters
        microbiome_features = microbiome_features.to(dtype=self.projection[0].weight.dtype)        
        return self.projection(microbiome_features)


class UserEncoder(nn.Module):
    """
    Simple MLP-based user encoder for static features.
    
    returns [B, num_user_{cat+real}_features, hidden_dim] if project_to_single_vector=False
    returns [B, 1, hidden_dim] if project_to_single_vector=True
    
    Args:
        categorical_variable_sizes: Dictionary mapping categorical variable names to their vocabulary sizes
        real_variables: List of names for real-valued variables
        hidden_dim: Dimension of the intermediate representations
        dropout_rate: Dropout rate for regularization
        project_to_single_vector: Whether to project all user features to a single vector
    """
    def __init__(
        self,
        categorical_variable_sizes: Dict[str, int],
        real_variables: List[str],
        hidden_dim: int,
        dropout_rate: float = 0.1,
        project_to_single_vector: bool = False,
    ):
        super().__init__()
        
        self.categorical_variable_sizes = categorical_variable_sizes
        self.real_variables = real_variables
        self.hidden_dim = hidden_dim
        self.num_cat_features = len(categorical_variable_sizes)
        self.num_real_features = len(real_variables)
        self.total_features = self.num_cat_features + self.num_real_features
        self.project_to_single_vector = project_to_single_vector
        
        # Individual embeddings for each categorical feature, named by variable
        self.cat_embeddings = nn.ModuleDict({
            var_name: nn.Embedding(size, self.hidden_dim)
            for var_name, size in categorical_variable_sizes.items()
        })
        
        # Individual projection layers for each real feature, named by variable
        if self.num_real_features > 0:
            self.real_projections = nn.ModuleDict({
                var_name: nn.Sequential(
                    nn.Linear(1, self.hidden_dim),
                    nn.LayerNorm(self.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                ) for var_name in real_variables
            })
        
        # Type embeddings to differentiate between each individual feature
        # Each feature (both categorical and real) gets its own type embedding
        self.type_embeddings = nn.Embedding(self.total_features, hidden_dim)
        
        # Optional projection layer to combine all features into a single vector
        if self.project_to_single_vector:
            self.final_projection = nn.Sequential(
                nn.Linear(hidden_dim * self.total_features, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
                
    def forward(self, user_categoricals: torch.Tensor, user_reals: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the user encoder.
        
        Args:
            user_categoricals: Tensor of shape (batch_size, num_categorical_features)
            user_reals: Tensor of shape (batch_size, num_real_features)
            
        Returns:
            user_embeddings: Tensor of shape (batch_size, num_features, hidden_dim) if project_to_single_vector=False
                           or (batch_size, 1, hidden_dim) if project_to_single_vector=True
        """
        batch_size = user_categoricals.size(0)
        
        # Get the dtype from the first parameter of the model
        dtype = next(self.parameters()).dtype
        device = user_categoricals.device
        
        # Process categorical variables using named embedding layers
        cat_embeddings_list = []
        for i, var_name in enumerate(self.categorical_variable_sizes.keys()):
            cat_feature = user_categoricals[:, i].long()
            # Get embeddings and add type embedding for this specific categorical feature
            feature_emb = self.cat_embeddings[var_name](cat_feature)  # [B, hidden_dim]
            type_id = torch.full((batch_size,), i, device=device, dtype=torch.long)
            feature_emb = feature_emb + self.type_embeddings(type_id)  # [B, hidden_dim]
            cat_embeddings_list.append(feature_emb)
        
        # Stack categorical embeddings if any exist
        if cat_embeddings_list:
            cat_embeddings = torch.stack(cat_embeddings_list, dim=1)  # [B, num_cat, hidden_dim]
        else:
            cat_embeddings = torch.empty((batch_size, 0, self.hidden_dim), device=device, dtype=dtype)
        
        # Process real features using named projection layers
        real_embeddings_list = []
        if self.num_real_features > 0:
            for i, var_name in enumerate(self.real_variables):
                # Get single real feature and ensure correct dtype
                real_feature = user_reals[:, i].to(dtype=dtype).unsqueeze(-1)  # [B, 1]
                # Project through named projection layer
                feature_emb = self.real_projections[var_name](real_feature)  # [B, hidden_dim]
                # Add type embedding for this specific real feature
                type_id = torch.full((batch_size,), self.num_cat_features + i, device=device, dtype=torch.long)
                feature_emb = feature_emb + self.type_embeddings(type_id)  # [B, hidden_dim]
                real_embeddings_list.append(feature_emb)
            
            # Stack all real embeddings
            real_embeddings = torch.stack(real_embeddings_list, dim=1)  # [B, num_real, hidden_dim]
        else:
            real_embeddings = torch.empty((batch_size, 0, self.hidden_dim), device=device, dtype=dtype)
        
        # Combine categorical and real embeddings
        combined = torch.cat([cat_embeddings, real_embeddings], dim=1)  # [B, num_features, hidden_dim]
        
        if self.project_to_single_vector:
            # Flatten and project to single vector
            flattened = combined.reshape(batch_size, -1)  # [B, num_features * hidden_dim]
            projected = self.final_projection(flattened)  # [B, hidden_dim]
            return projected.unsqueeze(1)  # [B, 1, hidden_dim]
        else:
            return combined  # [B, num_features, hidden_dim]

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
        
        # Define macro feature names
        self.macro_feature_names = [
            "food__eaten_quantity_in_gram", 
            "food__energy_kcal_eaten",
            "food__carb_eaten", 
            "food__fat_eaten",
            "food__protein_eaten", 
            "food__fiber_eaten",
            "food__alcohol_eaten"
        ]
        
        # --------------------
        #  Embedding Layers
        # --------------------
        self.food_emb = nn.Embedding(num_foods, food_embed_dim, padding_idx=0)
        self.food_emb_proj = nn.Linear(food_embed_dim, hidden_dim)
        
        if not self.ignore_food_macro_features:
            # Individual projection layers for each macro feature
            self.macro_projections = nn.ModuleDict({
                macro_name: nn.Linear(1, hidden_dim, bias=False)
                for macro_name in self.macro_feature_names[:food_macro_dim]
            })

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
            # Extract mask values for non-empty timesteps only
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
            # Process each macro feature separately using its dedicated projection layer
            macro_embs = []
            for idx, feature_name in enumerate(self.macro_feature_names[:self.food_macro_dim]):
                # Extract the specific macro feature and ensure correct shape
                macro_feature = filtered_meal_macros[:, :, idx].unsqueeze(-1)  # (N_non_empty, M, 1)
                # Project with the dedicated layer for this feature
                projected_feature = self.macro_projections[feature_name](macro_feature)  # (N_non_empty, M, hidden_dim)
                macro_embs.append(projected_feature)
            
            # Combine all macro embeddings by stacking and summing them explicitly along the macro dimension
            # First stack all macro embeddings: shape becomes (N_non_empty, M, num_macros, hidden_dim)
            stacked_macro_embs = torch.stack(macro_embs, dim=2)
            # Sum across the macro dimension (dim=2)
            macro_emb = torch.sum(stacked_macro_embs, dim=2)  # (N_non_empty, M, hidden_dim)
            macro_emb = self.dropout(macro_emb)
            
            # Combine with food embeddings
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
            if filtered_mask is not None and self.aggregator_type == "set":
                # Expand to match the meal items + aggregator token
                filtered_timestep_mask = filtered_mask.unsqueeze(1).expand(-1, pad_mask.size(1))  # (N_non_empty, M+1)
                
                # Combine with padding mask
                # In PyTorch transformers, src_key_padding_mask=True means positions to IGNORE
                # filtered_mask has True for valid positions, so we invert it for transformer
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
                    B * T,
                    attn_weights.size(-2),
                    attn_weights.size(-1),
                    device=attn_weights.device,
                    dtype=attn_weights.dtype,
                )
                
                # Place the computed attention weights in the right positions
                full_self_attn[non_empty_indices] = attn_weights
                
                # Reshape to (B, T, M+1, M+1)
                self_attn_out = full_self_attn.view(B, T, attn_weights.size(-2), attn_weights.size(-1))
        
        else:
            raise ValueError(f"Invalid aggregator_type='{self.aggregator_type}'. Use 'set' or 'sum' only.")

        # Apply filtered mask if provided
        if filtered_mask is not None:
            # In this codebase, mask=True means valid values
            # multiplying by mask zeros out invalid positions
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
            patch_indices: (N_patches,) indices where each patch ends (not starts)
            patch_mask: (B, N_patches) boolean mask for valid patches
        """
        glucose_seq = glucose_seq.float()  # Ensure float type
        B, T = glucose_seq.size()
        
        # Create patches using unfold
        patches = glucose_seq.unfold(1, self.patch_size, self.patch_stride)
        
        # No patches? Handle the edge case
        if patches.size(1) == 0:
            # Create a single patch
            patches = F.pad(glucose_seq, (0, self.patch_size - T))[:, None, :]
        
        # Get indices where each patch STARTS
        start_indices = torch.arange(0, T - self.patch_size + 1, self.patch_stride, device=glucose_seq.device)
        
        # Convert to indices where each patch ENDS
        patch_indices = start_indices + (self.patch_size - 1)
        
        # If no patches were created, use a single index at the end
        if patch_indices.size(0) == 0:
            patch_indices = torch.tensor([T-1], device=glucose_seq.device)
        
        # Handle masking if provided
        patch_mask = None
        if mask is not None:
            # Patchify the mask
            mask_patches = mask.unfold(1, self.patch_size, self.patch_stride)
            
            # Handle edge case where T < patch_size
            if mask_patches.size(1) == 0:
                mask_patches = F.pad(mask, (0, self.patch_size - T))[:, None, :]
            
            # A patch is valid only if at least one point in it is valid
            # In this codebase, mask=True means valid positions
            # patch_mask will be True for patches with at least one valid point
            patch_mask = mask_patches.any(dim=2)  # [B, N_patches]
            
            # Apply mask to patches (zero out invalid points)
            # mask_patches has True for valid points, multiplying zeros out invalid points
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
            # In PyTorch transformers, src_key_padding_mask=True means positions to IGNORE
            # patch_mask has True for valid positions, so we need to invert it
            key_padding_mask = ~patch_mask
        
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
            patch_indices: (N_patches,) indices where each patch ends (not starts)
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
    
    def forward(
        self, 
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
            # In this codebase, mask=True means valid values
            # multiplying by mask zeros out invalid positions
            temporal_embeddings = temporal_embeddings * mask.unsqueeze(-1).float()
        
        return temporal_embeddings

class SimpleGlucoseEncoder(nn.Module):
    """
    A simple encoder that transforms each glucose value to hidden_dim using a linear layer.
    No patching is performed in this encoder.
    """
    def __init__(
        self,
        embed_dim: int,
        positional_embedding: Optional[nn.Module] = None,
        dropout_rate: float = 0.1,
        add_type_embedding: bool = True  # Option to add type embedding
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.positional_embedding = positional_embedding
        self.add_type_embedding = add_type_embedding
        
        # Transformation from 1D glucose value to hidden_dim
        self.glucose_projection = nn.Linear(1, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Add type embedding if requested
        if add_type_embedding:
            self.type_embedding = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
    def forward(
        self,
        glucose_values: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_self_attn: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for the glucose encoder.
        
        Args:
            glucose_values: Tensor of shape [batch_size, seq_len]
            positions: Optional tensor of shape [batch_size, seq_len] for positional encoding
            mask: Optional mask tensor of shape [batch_size, seq_len]
            return_self_attn: Whether to return self-attention weights (not used in this encoder)
            
        Returns:
            Tuple of:
            - glucose_embeddings: Tensor of shape [batch_size, seq_len, embed_dim]
            - None: No self-attention weights in this encoder
        """
        batch_size, seq_len = glucose_values.shape
        
        # Convert to [batch_size, seq_len, 1] for the linear projection
        glucose_values = glucose_values.unsqueeze(-1)
        
        # Project glucose values to embedding dimension
        glucose_embeddings = self.glucose_projection(glucose_values)
        
        # Add type embedding if requested
        if self.add_type_embedding:
            glucose_embeddings = glucose_embeddings + self.type_embedding
        
        # Apply dropout
        glucose_embeddings = self.dropout(glucose_embeddings)
        
        # Apply layer normalization
        glucose_embeddings = self.layer_norm(glucose_embeddings)
        
        # Apply positional encoding if provided
        if self.positional_embedding is not None and positions is not None:
            glucose_embeddings = self.positional_embedding(glucose_embeddings, positions)
            
        # Apply mask if provided
        if mask is not None:
            # In this codebase, mask=True means valid values
            # multiplying by mask zeros out invalid positions
            glucose_embeddings = glucose_embeddings * mask.unsqueeze(-1).float()
        
        return glucose_embeddings
    
    
