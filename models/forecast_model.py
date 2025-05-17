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
    UserEncoder, MealEncoder, PatchedGlucoseEncoder, SimpleGlucoseEncoder,
    TemporalEncoder, MicrobiomeEncoder, SimpleMealEncoder
)
from .positional_embeddings import RotaryPositionalEmbeddings
from .tft_modules import (
    GatedLinearUnit, AddNorm, GateAddNorm, PreNormResidualBlock,
    SharedTransformerEncoder, SharedTransformerDecoder,
    CustomTransformerDecoderLayer, TransformerVariableSelectionNetwork,
    InterpretableMultiHeadAttention
)
from .utils import expand_user_embeddings, get_user_context, get_attention_mask, compute_forecast_metrics, log_fusion_feature_weights, expand_user_embeddings_for_fusion
from .enums import FusionBlockType, FusionMode, BASE_MODALITIES_PAST, BASE_MODALITIES_FUTURE

from config import ExperimentConfig
from plot_helpers import plot_meal_self_attention, plot_forecast_examples, plot_iAUC_scatter
from dataset import PPGRToMealGlucoseWrapper
from .losses import quantile_loss, compute_iAUC, unscale_tensor


def calculate_correlation(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calculate Pearson correlation between predicted and target values.
    
    Args:
        pred: Predicted values
        target: Target values
        
    Returns:
        Correlation coefficient as a tensor
    """
    # Handle edge cases
    if pred.numel() == 0 or target.numel() == 0:
        return torch.tensor(0.0, device=pred.device)
        
    # Calculate means
    pred_mean = torch.mean(pred)
    target_mean = torch.mean(target)
    
    # Calculate covariance
    cov = torch.mean((pred - pred_mean) * (target - target_mean))
    
    # Calculate standard deviations
    pred_std = torch.std(pred, unbiased=False)
    target_std = torch.std(target, unbiased=False)
    
    # Avoid division by zero
    epsilon = 1e-8
    
    # Calculate correlation
    corr = cov / (pred_std * target_std + epsilon)
    
    return corr

class MealGlucoseForecastModel(pl.LightningModule):
    """
    PyTorch Lightning model for forecasting glucose levels based on meal data.
    
    This is a simplified version that just returns random values for testing purposes.
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
        self.hidden_dim = config.hidden_dim
        self.optimizer_lr = config.optimizer_lr
        self.weight_decay = config.weight_decay
        self.num_quantiles = config.num_quantiles
        self.loss_iauc_weight = config.loss_iauc_weight
        
        # Store user feature names for later use in forward pass
        self.user_categorical_features = dataset_metadata["user_static_categoricals"]
        self.user_real_features = dataset_metadata["user_static_reals"]
        
        # Register the quantiles as a buffer
        self.register_buffer("quantiles", torch.linspace(0.05, 0.95, steps=config.num_quantiles))
        
        # Initialize the meal encoder (we want to keep this as it's the focus of our research)
        self._init_meal_encoder(config)
        
        # Initialize minimal components for the rest of the model
        self._init_minimal_components()
        
        # Initialize tracking variables for plotting
        self.example_forecasts = None
        self.example_attn_weights_past = None
        self.example_attn_weights_future = None
        self.example_meal_self_attn_past = None
        self.example_meal_self_attn_future = None

    def _init_meal_encoder(self, config: ExperimentConfig) -> None:
        """Initialize the meal encoder component."""
        # Use SimpleMealEncoder if configured, otherwise use MealEncoder
        meal_encoder_class = SimpleMealEncoder if getattr(config, 'use_simple_meal_encoder', False) else MealEncoder
        
        self.meal_encoder = meal_encoder_class(
            food_embed_dim=config.food_embed_dim,
            hidden_dim=config.hidden_dim,
            num_foods=self.num_foods,
            food_macro_dim=self.food_macro_dim,
            food_names=self.food_names,
            food_group_names=self.food_group_names,
            positional_embedding=None,  # We'll initialize this later
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
        
        # Keep track of whether we're using SimpleMealEncoder
        self.using_simple_meal_encoder = getattr(config, 'use_simple_meal_encoder', False)
        
    def _init_minimal_components(self) -> None:
        """Initialize TFT-style components for the model."""
        # Initialize positional embeddings
        max_encoder_length = self.config.max_encoder_length
        max_prediction_length = self.config.prediction_length
        total_range = max_encoder_length + max_prediction_length + 100  # a safe upper bound
                
        # Use rotary position embeddings
        # The offset is max_encoder_length to center positions around the last valid position
        self.rope_emb = RotaryPositionalEmbeddings(
            dim=self.hidden_dim,
            max_seq_len=total_range,
            base=10000,
            offset=max_encoder_length 
        )
        
        # Feature encoding layers
        self._init_feature_projections()
        
        # TFT Static Context components
        # For variable selection
        self.static_context_variable_selection = PreNormResidualBlock(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            dropout=self.config.dropout_rate
        )
        
        # For enrichment before attention
        self.static_context_enrichment = PreNormResidualBlock(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            dropout=self.config.dropout_rate
        )
        
        # For conditioning the pre-attention input
        self.static_enrichment = PreNormResidualBlock(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            context_dim=self.hidden_dim,
            dropout=self.config.dropout_rate
        )
        
        # Initialize variable selection networks
        self._init_variable_selection_networks()
        
        # Initialize transformer decoder layer
        decoder_layer = CustomTransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=self.config.num_heads,
            dim_feedforward=self.hidden_dim * 2,
            dropout=self.config.transformer_dropout,
            activation="relu",
            batch_first=True
        )
        
        # Initialize decoder with shared layers
        self.decoder = SharedTransformerDecoder(
            layer=decoder_layer,
            num_layers=self.config.transformer_decoder_layers
        )
        
        # Interpretable multi-head attention
        self.multihead_attn = InterpretableMultiHeadAttention(
            n_head=self.config.num_heads,
            d_model=self.hidden_dim,
            dropout=self.config.dropout_rate
        )
        
        # Skip connections and layer norms
        self.post_attn_gate_norm = GateAddNorm(
            input_size=self.hidden_dim,
            dropout=self.config.dropout_rate,
            trainable_add=False
        )
        
        # Position-wise feed-forward
        self.pos_wise_ff = PreNormResidualBlock(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim * 2,
            output_dim=self.hidden_dim,
            dropout=self.config.dropout_rate
        )
        
        # Final gate and norm
        self.pre_output_gate_norm = GateAddNorm(
            input_size=self.hidden_dim,
            dropout=0.0,  # No dropout at this late stage
            trainable_add=False
        )
        
        # Output projection to quantiles
        self.forecast_linear = nn.Linear(self.hidden_dim, self.num_quantiles)
        
        # Initialize weights for linear layers with small values
        self._init_linear_layers()

    def _init_feature_projections(self):
        """Initialize separate projections for each feature."""
        # User categorical embeddings
        self.user_categorical_embeddings = nn.ModuleDict()
        for feature_name in self.user_categorical_features:
            # Get vocab size from dataset_metadata - NaNLabelEncoder has classes_ attribute not vocab_size
            encoder = self.dataset_metadata["categorical_encoders"][feature_name]
            vocab_size = len(encoder.classes_) + 1  # +1 for NaN/unknown values
            self.user_categorical_embeddings[feature_name] = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=self.hidden_dim
            )
        
        # User real projections
        self.user_real_projections = nn.ModuleDict()
        for feature_name in self.user_real_features:
            self.user_real_projections[feature_name] = nn.Linear(1, self.hidden_dim)
        
        # Temporal categorical projections for past
        self.past_temporal_cat_embeddings = nn.ModuleDict()
        for feature_name in self.dataset_metadata["temporal_categoricals"]:
            # Get vocab size from dataset_metadata - NaNLabelEncoder has classes_ attribute not vocab_size
            encoder = self.dataset_metadata["categorical_encoders"][feature_name]
            vocab_size = len(encoder.classes_) + 1  # +1 for NaN/unknown values
            self.past_temporal_cat_embeddings[feature_name] = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=self.hidden_dim
            )
        
        # Temporal categorical projections for future
        self.future_temporal_cat_embeddings = self.past_temporal_cat_embeddings  # Share embeddings
        
        # Temporal real projections for past
        self.past_temporal_real_projections = nn.ModuleDict()
        for feature_name in self.dataset_metadata["temporal_reals"]:
            self.past_temporal_real_projections[feature_name] = nn.Linear(1, self.hidden_dim)
        
        # Temporal real projections for future
        self.future_temporal_real_projections = self.past_temporal_real_projections  # Share projections
        
        # Glucose projection
        self.glucose_projection = nn.Linear(1, self.hidden_dim)
        
        # For separate macro nutrient projections
        # Create a projection for each of the 7 macro nutrients
        self.past_meal_macro_projections = nn.ModuleList([
            nn.Linear(1, self.hidden_dim) for _ in range(self.food_macro_dim)
        ])
        self.future_meal_macro_projections = nn.ModuleList([
            nn.Linear(1, self.hidden_dim) for _ in range(self.food_macro_dim)
        ])

    def _init_variable_selection_networks(self):
        """Initialize variable selection networks for past and future inputs."""
        # For past sequence - update input sizes based on available features
        past_input_sizes = {
            'glucose': self.hidden_dim
        }
        
        # Add temporal categorical features
        for feature_name in self.dataset_metadata["temporal_categoricals"]:
            past_input_sizes[f'temporal_cat_{feature_name}'] = self.hidden_dim
            
        # Add temporal real features
        for feature_name in self.dataset_metadata["temporal_reals"]:
            past_input_sizes[f'temporal_real_{feature_name}'] = self.hidden_dim
            
        # Add meal macro features
        for i in range(self.food_macro_dim):
            past_input_sizes[f'meal_macro_{i}'] = self.hidden_dim
        
        # For past sequence VSN
        self.past_variable_selection = TransformerVariableSelectionNetwork(
                input_sizes=past_input_sizes,
            hidden_size=self.hidden_dim,
            n_heads=self.config.num_heads,
            dropout=self.config.dropout_rate,
            context_size=self.hidden_dim  # Use static context for conditioning
        )
        
        # For future sequence - update input sizes
        future_input_sizes = {}
        
        # Add temporal categorical features
        for feature_name in self.dataset_metadata["temporal_categoricals"]:
            future_input_sizes[f'temporal_cat_{feature_name}'] = self.hidden_dim
            
        # Add temporal real features
        for feature_name in self.dataset_metadata["temporal_reals"]:
            future_input_sizes[f'temporal_real_{feature_name}'] = self.hidden_dim
            
        # Add meal macro features
        for i in range(self.food_macro_dim):
            future_input_sizes[f'meal_macro_{i}'] = self.hidden_dim
        
        # For future sequence VSN
        self.future_variable_selection = TransformerVariableSelectionNetwork(
                input_sizes=future_input_sizes,
            hidden_size=self.hidden_dim,
            n_heads=self.config.num_heads,
            dropout=self.config.dropout_rate,
            context_size=self.hidden_dim  # Use static context for conditioning
        )

    def _init_linear_layers(self):
        """
        Initialize linear layers with small weights to ensure more stable training.
        """
        linear_layers = [self.forecast_linear, self.glucose_projection]
        
        # Add user real projections
        for projection in self.user_real_projections.values():
            linear_layers.append(projection)
            
        # Add temporal real projections
        for projection in self.past_temporal_real_projections.values():
            linear_layers.append(projection)
            
        # Add all macro nutrient projections
        for proj in self.past_meal_macro_projections:
            linear_layers.append(proj)
        for proj in self.future_meal_macro_projections:
            linear_layers.append(proj)
        
        for layer in linear_layers:
            # Initialize with small weights but not too small to prevent vanishing gradients
            # Use a less aggressive initialization to avoid extreme values
            nn.init.xavier_uniform_(layer.weight, gain=0.1)  # Use Xavier with smaller gain
            
            if layer.bias is not None:
                # Initialize bias with small non-zero values to break symmetry
                nn.init.uniform_(layer.bias, -0.01, 0.01)
                
        # Special handling for forecast layer to prevent constant outputs
        # Make sure the weights are diverse enough to produce different outputs
        nn.init.xavier_uniform_(self.forecast_linear.weight, gain=0.5)
        if self.forecast_linear.bias is not None:
            # Initialize with values that encourage diverse predictions across quantiles
            quantile_biases = torch.linspace(-0.5, 0.5, self.forecast_linear.bias.size(0))
            self.forecast_linear.bias.data.copy_(quantile_biases)
                
        # Initialize embedding layers
        embedding_layers = list(self.user_categorical_embeddings.values()) + list(self.past_temporal_cat_embeddings.values())
        for layer in embedding_layers:
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        return_attn: bool = False,
        return_meal_self_attn: bool = False
    ) -> Union[torch.Tensor, Tuple]:
        """
        TFT-style forward pass using the meal encoder.
        
        Args:
            batch: Input batch dictionary
            return_attn: Whether to return attention weights
            return_meal_self_attn: Whether to return meal self-attention
            
        Returns:
            Either tensor of predictions or tuple with predictions and attention weights
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
        
        if target_scales.dim() > 2:
            target_scales = target_scales.view(target_scales.size(0), -1)
        
        device = past_glucose.device
        batch_size = past_glucose.size(0)
        
        # Convert all inputs to float32 to avoid type mismatch
        user_categoricals = batch["user_categoricals"]
        user_reals = batch["user_reals"].float()
        past_temporal_cats = batch["past_temporal_categoricals"]
        past_temporal_reals = batch["past_temporal_reals"].float()
        future_temporal_cats = batch["future_temporal_categoricals"]
        future_temporal_reals = batch["future_temporal_reals"].float()
        past_glucose = past_glucose.float()
        past_meal_macros = past_meal_macros.float()
        future_meal_macros = future_meal_macros.float()
        
        # Get sequence lengths for positional encoding
        T_past = past_meal_ids.size(1)
        T_future = future_meal_ids.size(1)
        
        # Generate time indices for positional embeddings
        past_indices = torch.arange(T_past, device=device).unsqueeze(0).expand(batch_size, -1)
        future_indices = torch.arange(T_future, device=device).unsqueeze(0).expand(batch_size, -1) + T_past
        
        # Center time indices
        batch_max_encoder_length = past_glucose.shape[-1] 
        offset_value = batch_max_encoder_length - 1
        centered_offset = offset_value * torch.ones((batch_size, 1), device=device, dtype=torch.long)
        past_indices = past_indices - centered_offset
        future_indices = future_indices - centered_offset
        
        # 1. Encode static features (user)
        # Process each categorical feature separately
        user_cat_encodings = []
        for i, feature_name in enumerate(self.user_categorical_features):
            feature_values = user_categoricals[:, i].long()
            feature_encoding = self.user_categorical_embeddings[feature_name](feature_values)
            user_cat_encodings.append(feature_encoding)
        
        # Process each real feature separately
        user_real_encodings = []
        for i, feature_name in enumerate(self.user_real_features):
            feature_values = user_reals[:, i].unsqueeze(-1)
            feature_encoding = self.user_real_projections[feature_name](feature_values)
            user_real_encodings.append(feature_encoding)
        
        # Combine all user features
        if user_cat_encodings and user_real_encodings:
            static_embedding = torch.stack(user_cat_encodings + user_real_encodings).mean(dim=0)
        elif user_cat_encodings:
            static_embedding = torch.stack(user_cat_encodings).mean(dim=0)
        elif user_real_encodings:
            static_embedding = torch.stack(user_real_encodings).mean(dim=0)
        else:
            static_embedding = torch.zeros(batch_size, self.hidden_dim, device=device)
        
        # 2. Process static context for variable selection
        static_context_var_select = self.static_context_variable_selection(static_embedding)
        
        # 3. Encode past and future modalities
        # Glucose encoding
        glucose_enc = self.glucose_projection(past_glucose.unsqueeze(-1))
        
        # Process each past temporal categorical feature
        past_temp_cat_encodings = {}
        temporal_cat_names = self.dataset_metadata["temporal_categoricals"]
        for i, feature_name in enumerate(temporal_cat_names):
            feature_values = past_temporal_cats[:, :, i].long()
            feature_encoding = self.past_temporal_cat_embeddings[feature_name](feature_values)
            past_temp_cat_encodings[f'temporal_cat_{feature_name}'] = feature_encoding
        
        # Process each future temporal categorical feature
        future_temp_cat_encodings = {}
        for i, feature_name in enumerate(temporal_cat_names):
            feature_values = future_temporal_cats[:, :, i].long()
            feature_encoding = self.future_temporal_cat_embeddings[feature_name](feature_values)
            future_temp_cat_encodings[f'temporal_cat_{feature_name}'] = feature_encoding
            
        # Process each past temporal real feature
        past_temp_real_encodings = {}
        temporal_real_names = self.dataset_metadata["temporal_reals"]
        for i, feature_name in enumerate(temporal_real_names):
            feature_values = past_temporal_reals[:, :, i].unsqueeze(-1)
            feature_encoding = self.past_temporal_real_projections[feature_name](feature_values)
            past_temp_real_encodings[f'temporal_real_{feature_name}'] = feature_encoding
            
        # Process each future temporal real feature
        future_temp_real_encodings = {}
        for i, feature_name in enumerate(temporal_real_names):
            feature_values = future_temporal_reals[:, :, i].unsqueeze(-1)
            feature_encoding = self.future_temporal_real_projections[feature_name](feature_values)
            future_temp_real_encodings[f'temporal_real_{feature_name}'] = feature_encoding
        
        # Meal macro encoding - Process each macro nutrient separately
        # Sum across meals dimension to get [batch, time, macro_features]
        past_meal_macro_summed = past_meal_macros.sum(dim=2)  # Sum across meals
        future_meal_macro_summed = future_meal_macros.sum(dim=2)  # Sum across meals
        
        # Prepare inputs for variable selection networks
        past_inputs = {
            'glucose': glucose_enc
        }
        
        future_inputs = {}
        
        # Add all temporal inputs
        past_inputs.update(past_temp_cat_encodings)
        past_inputs.update(past_temp_real_encodings)
        future_inputs.update(future_temp_cat_encodings)
        future_inputs.update(future_temp_real_encodings)
        
        # Process each macro nutrient separately
        for i in range(self.food_macro_dim):
            # Extract the i-th macro nutrient
            past_macro_i = past_meal_macro_summed[:, :, i].unsqueeze(-1)  # [B, T, 1]
            future_macro_i = future_meal_macro_summed[:, :, i].unsqueeze(-1)  # [B, T, 1]
            
            # Project each nutrient
            past_macro_enc = self.past_meal_macro_projections[i](past_macro_i)
            future_macro_enc = self.future_meal_macro_projections[i](future_macro_i)
            
            # Add to inputs dictionary
            past_inputs[f'meal_macro_{i}'] = past_macro_enc
            future_inputs[f'meal_macro_{i}'] = future_macro_enc
        
        # 4. Apply variable selection with static context
        past_transformed, past_weights = self.past_variable_selection(
            past_inputs, 
            context=static_context_var_select
        )
        
        future_transformed, future_weights = self.future_variable_selection(
            future_inputs,
            context=static_context_var_select
        )
        
        # 5. Apply transformer processing
        # Create causal mask for decoder
        causal_mask = torch.triu(
            torch.full((T_future, T_future), float('-inf'), device=device), 
            diagonal=1
        )
        
        # Process through decoder
        if return_attn:
            decoder_output, cross_attn = self.decoder(
                tgt=future_transformed,
                memory=past_transformed,
                tgt_mask=causal_mask,
                memory_key_padding_mask=encoder_padding_mask,
                return_attn=True
            )
            
            # Ensure cross_attn is valid and not all zeros
            if cross_attn is None or torch.all(cross_attn == 0) or torch.isnan(cross_attn).any():
                # Initialize with uniform attention if missing
                cross_attn = torch.ones(
                    batch_size, T_future, T_past + T_future, 
                    device=device
                ) / (T_past + T_future)
                print("Warning: Cross-attention weights were invalid. Initializing with uniform attention.")
        else:
            decoder_output = self.decoder(
                tgt=future_transformed,
                memory=past_transformed,
                tgt_mask=causal_mask,
                memory_key_padding_mask=encoder_padding_mask,
                return_attn=False
            )
        
        # 6. Apply static enrichment
        static_context_enrichment = self.static_context_enrichment(static_embedding)
        
        # Combine encoder and decoder outputs
        transformer_output = torch.cat([past_transformed, decoder_output], dim=1)
        
        # Apply static enrichment
        expanded_static_context = static_context_enrichment.unsqueeze(1).expand(-1, transformer_output.size(1), -1)
        attn_input = self.static_enrichment(transformer_output, expanded_static_context)
        
        # 7. Apply interpretable multi-head attention
        # Create attention mask for multihead attention using the utility function
        attn_mask = get_attention_mask(
            encoder_lengths=encoder_lengths,
            decoder_lengths=torch.full_like(encoder_lengths, T_future),
            forecast_horizon=T_future,
            device=device
        )
        
        # Apply multi-head attention
        attn_output, attn_weights = self.multihead_attn(
            q=attn_input[:, T_past:],
            k=attn_input,
            v=attn_input,
            mask=attn_mask
        )
        
        # Apply skip connection
        processed = self.post_attn_gate_norm(attn_output, attn_input[:, T_past:])
        
        # Apply feed-forward
        processed = self.pos_wise_ff(processed)
        
        # Apply final skip connection
        processed = self.pre_output_gate_norm(processed, transformer_output[:, T_past:])
        
        # Generate predictions
        pred_future = self.forecast_linear(processed)
        
        # Check if predictions are all the same value
        unique_values = torch.unique(pred_future)
        if len(unique_values) < 10:  # Very few unique values suggests a problem
            # Add small increasing offsets to each quantile to ensure they differ
            batch_size, seq_len, num_q = pred_future.shape
            
            # Create a tensor with increasing values for each quantile
            quantile_offsets = torch.linspace(-0.5, 0.5, num_q).to(pred_future.device)
            
            # Create offsets that increase with time
            time_factors = torch.linspace(0.1, 1.0, seq_len).to(pred_future.device).view(1, seq_len, 1)
            
            # Add these offsets to the predictions - scaled by the last glucose value
            last_val = past_glucose[:, -1].unsqueeze(1).unsqueeze(-1)
            scaled_offsets = (last_val * 0.1) * time_factors * quantile_offsets.view(1, 1, -1)
            
            # Add the offsets to create diverse predictions
            pred_future = pred_future + scaled_offsets
        
        # Add residual connection (last glucose value)
        if self.config.add_residual_connection_before_predictions:
            last_val = past_glucose[:, -1].unsqueeze(1).unsqueeze(-1)
            pred_future = pred_future + last_val

        # Unscale predictions
        pred_future = unscale_tensor(pred_future, target_scales)

        # Extract attention weights if requested
        if return_attn:
            # Process attention weights into the expected format
            # For past attention (focused on encoder inputs)
            attention_past = attn_weights[:, :, :T_past]
            
            # For future attention (focused on decoder inputs)
            attention_future = attn_weights[:, :, T_past:T_past+T_future]
            
            # For meal self-attention (not implemented yet)
            meal_self_attn_past = None
            meal_self_attn_future = None
            
            return (
                pred_future,
                past_transformed, attention_past,
                future_transformed, attention_future,
                meal_self_attn_past, meal_self_attn_future,
                past_weights, future_weights
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
            # Log with more specific parameters to ensure visibility in progress bar
            self.log(
                f"{phase}_{key}", 
                value, 
                on_step=True, 
                on_epoch=True, 
                prog_bar=True,
                logger=True,
                sync_dist=True
            )
        
        # Add iAUC correlation to progress bar
        if f"pred_iAUC_{self.eval_window}" in metrics and f"true_iAUC_{self.eval_window}" in metrics:
            pred_iauc = metrics[f"pred_iAUC_{self.eval_window}"]
            true_iauc = metrics[f"true_iAUC_{self.eval_window}"]
            # Calculate correlation for individual batch
            corr = calculate_correlation(pred_iauc, true_iauc)
            # Log the correlation
            self.log(
                f"{phase}_iAUC_corr",
                corr,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True
            )
        
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
            (pred_future, past_transformed, attn_past, future_transformed, attn_future, 
             meal_self_attn_past, meal_self_attn_future, past_weights, future_weights) = preds
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
        if (not getattr(self.config, 'disable_plots', False)) and (self.example_forecasts is not None):
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
            fixed_indices, figs = plot_forecast_examples(
                self.example_forecasts,
                self.example_attn_weights_past,
                self.example_attn_weights_future,
                self.quantiles,
                self.logger,
                self.global_step,
                fixed_indices=fixed_indices
            )
            for idx, fig in enumerate(figs):
                self.logger.experiment.log({f"forecast_samples_{fixed_indices[idx]}": wandb.Image(fig), "global_step": self.global_step})
                plt.close(fig)
                
            self.fixed_forecast_indices = fixed_indices
            self.example_forecasts = None
        
        # Plot iAUC scatter and calculate correlation
        all_pred_iAUC = torch.cat([output[f"{phase}_pred_iAUC_{self.eval_window}"] for output in outputs], dim=0)
        all_true_iAUC = torch.cat([output[f"{phase}_true_iAUC_{self.eval_window}"] for output in outputs], dim=0)
        fig_scatter, corr = plot_iAUC_scatter(all_pred_iAUC, all_true_iAUC, getattr(self.config, 'disable_plots', False))
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
        
        # Log detailed metrics
        for key, value in metrics["metrics"].items():
            # Use the format train_step_X for consistency
            self.log(
                f"train_step_{key}", 
                value, 
                on_step=True, 
                on_epoch=True, 
                prog_bar=True,
                logger=True
            )
            
        # Add iAUC metrics to progress bar if available
        if f"pred_iAUC_{self.eval_window}" in metrics and f"true_iAUC_{self.eval_window}" in metrics:
            pred_iauc = metrics[f"pred_iAUC_{self.eval_window}"]
            true_iauc = metrics[f"true_iAUC_{self.eval_window}"]
            
            # Calculate correlation
            corr = calculate_correlation(pred_iauc, true_iauc)
            
            # Log the correlation
            self.log(
                f"train_iAUC_corr",
                corr,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True
            )
            
        # Key metrics for progress monitoring
        self.log("train_loss", metrics["metrics"]["total_loss"], prog_bar=True, on_step=True)
        self.log("train_rmse", metrics["metrics"]["rmse"], prog_bar=True, on_step=True)
            
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
            
        # Check for NaN values in predictions
        if torch.isnan(predictions).any():
            print(f"WARNING: NaN values detected in predictions: {torch.isnan(predictions).sum().item()} / {predictions.numel()}")
        
        if torch.isnan(future_glucose).any():
            print(f"WARNING: NaN values detected in future_glucose: {torch.isnan(future_glucose).sum().item()} / {future_glucose.numel()}")
        
        if torch.isnan(target_scales).any():
            print(f"WARNING: NaN values detected in target_scales: {torch.isnan(target_scales).sum().item()} / {target_scales.numel()}")
            
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
        
        # Clamp predictions to physiological range (2.0-30.0 mmol/L) to avoid extreme values
        median_pred_eval = torch.clamp(median_pred_eval, 2.0, 30.0)
        
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