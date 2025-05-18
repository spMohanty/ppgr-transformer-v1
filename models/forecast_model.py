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
import traceback
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
    TransformerVariableSelectionNetwork, InterpretableMultiHeadAttention
)

from config import ExperimentConfig
from plot_helpers import plot_meal_self_attention, plot_forecast_examples, plot_iAUC_scatter
from dataset import PPGRToMealGlucoseWrapper
from .losses import quantile_loss, compute_iAUC, unscale_tensor, calculate_correlation

from pytorch_forecasting.models.nn import MultiEmbedding
from pytorch_forecasting.utils import get_embedding_size, create_mask



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
        self.hidden_continuous_dim = config.hidden_continuous_dim
        self.num_quantiles = config.num_quantiles
        self.loss_iauc_weight = config.loss_iauc_weight
        
        # Store user feature names for later use in forward pass
        self.user_categorical_features = dataset_metadata["user_static_categoricals"]
        self.user_real_features = dataset_metadata["user_static_reals"]
        
        # Register the quantiles as a buffer
        self.register_buffer("quantiles", torch.linspace(0.05, 0.95, steps=config.num_quantiles))
        
        # Initialize the meal encoder (we want to keep this as it's the focus of our research)
        # self._init_meal_encoder(config)
                
        self.setup_static_categorical_embeddings()
        self.setup_prescalers()
        self.setup_static_context_encoders()
        self.setup_transformer_encoder_decoder_layers()
        self.setup_variable_selection_networks()
        self.setup_output_layers()
        
        self.positional_embeddings = RotaryPositionalEmbeddings(
            dim=self.config.hidden_dim,
            base=10000,
            offset=self.config.max_encoder_length # Center positions around the last valid position
        )
        
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
            num_heads=config.meal_transformer_num_heads,
            num_layers=config.meal_transformer_encoder_layers,
            layers_share_weights=config.meal_transformer_encoder_layers_share_weights,
            dropout_rate=config.dropout_rate,
            transformer_dropout=config.transformer_dropout,
            ignore_food_macro_features=config.ignore_food_macro_features,
            bootstrap_food_id_embeddings=None,  # This is initialized in the from_dataset method
            freeze_food_id_embeddings=config.freeze_food_id_embeddings,
            aggregator_type=config.meal_aggregator_type
        )
        
        # Keep track of whether we're using SimpleMealEncoder
        self.using_simple_meal_encoder = getattr(config, 'use_simple_meal_encoder', False)
        
    def setup_static_context_encoders(self):
        ## Static Encoders
        # for variable selection
        self.static_context_variable_selection = PreNormResidualBlock(
            input_dim=self.config.hidden_dim,
            hidden_dim=self.config.hidden_dim,
            output_dim=self.config.hidden_dim,
            dropout=self.config.dropout_rate,
            context_dim=None,
        )
        
        # for post lstm static enrichment
        self.static_context_enrichment = PreNormResidualBlock(
            input_dim=self.config.hidden_dim,
            hidden_dim=self.config.hidden_dim,
            output_dim=self.config.hidden_dim,
            dropout=self.config.dropout_rate,
            context_dim=None,
        )
        
        self.multihead_attn = InterpretableMultiHeadAttention(
            d_model=self.config.hidden_dim,
            n_head=self.config.transformer_encoder_decoder_num_heads,
            dropout=self.config.dropout_rate,
        )        
        
        # static enrichment just before the multihead attn
        self.static_enrichment = PreNormResidualBlock(
            input_dim=self.config.hidden_dim,
            hidden_dim=self.config.hidden_dim,
            output_dim=self.config.hidden_dim,
            dropout=self.config.dropout_rate,
            context_dim=self.config.hidden_dim,
        )
    
    def setup_transformer_encoder_decoder_layers(self):
        # Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.hidden_dim,
            nhead=self.config.transformer_encoder_decoder_num_heads,
            dim_feedforward=self.config.transformer_encoder_decoder_hidden_size,
            dropout=self.config.transformer_dropout,
            batch_first=True,  
        )
        self.transformer_encoder = SharedTransformerEncoder(
            layer=encoder_layer,
            num_layers=self.config.transformer_encoder_decoder_num_layers,
        )

        # Decoder Layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.config.hidden_dim,
            nhead=self.config.transformer_encoder_decoder_num_heads,
            dim_feedforward=self.config.transformer_encoder_decoder_hidden_size,
            dropout=self.config.transformer_dropout,
            batch_first=True,
        )
        self.transformer_decoder = SharedTransformerDecoder(
            layer=decoder_layer,
            num_layers=self.config.transformer_encoder_decoder_num_layers,
        )
        
        # skip connection for transformer encoder and decoder
        self.post_transformer_gate_encoder = GatedLinearUnit(
            self.config.hidden_dim, dropout=self.config.dropout_rate
        )
        self.post_transformer_gate_decoder = self.post_transformer_gate_encoder
        self.post_transformer_add_norm_encoder = AddNorm(
            self.config.hidden_dim, trainable_add=False
        )
        self.post_transformer_add_norm_decoder = self.post_transformer_add_norm_encoder
    
    def setup_static_categorical_embeddings(self):
        user_static_categoricals = []
        user_static_categoricals += self.dataset_metadata["user_static_categoricals"]
        
        self.user_static_categoricals_embeddings = MultiEmbedding(
            embedding_sizes = {
                name: (
                    len(self.dataset_metadata["categorical_encoders"][name].classes_), 
                    get_embedding_size(
                        len(self.dataset_metadata["categorical_encoders"][name].classes_), 
                        self.config.hidden_continuous_dim
                    )
                )
                for name in user_static_categoricals
            },
            x_categoricals=user_static_categoricals,
            max_embedding_size=self.config.hidden_dim,
        )
    
    def setup_prescalers(self):
        """Setup linear transformations for continuous features (prescalers)."""
        self.prescalers = nn.ModuleDict()
        
        # Define categories to process
        categories = [
            "user_static_reals",  # User static features
            "temporal_reals",     # Temporal features
            "food_reals",         # Food features
            "target_columns"      # Target variables
        ]
        
        # Default value for target_columns if not present
        default_values = {
            "target_columns": ["val"]
        }
        
        # Create prescalers for all continuous features
        for category in categories:
            # Get features with appropriate default
            features = self.dataset_metadata.get(category, default_values.get(category, []))
            
            for name in features:
                if name not in self.prescalers:  # Avoid duplicates
                    self.prescalers[name] = nn.Linear(1, self.config.hidden_continuous_dim)
                    logger.info(f"Created prescaler for '{name}' from category '{category}'")
                else:
                    logger.warning(f"Prescaler for '{name}' already exists")
        # Log summary
        logger.info(f"Created {len(self.prescalers)} prescalers, each mapping to hidden_continuous_dim={self.config.hidden_continuous_dim}")
        logger.info(f"Complete prescalers list: {list(self.prescalers.keys())}")
    
    def setup_variable_selection_networks(self):
        
        # variable selection for static variables
        
        # User static
        static_categorical_input_sizes = {
            name: self.user_static_categoricals_embeddings.output_size[name]
            for name in self.user_categorical_features
        }
        static_real_input_sizes = {
            name: self.hidden_continuous_dim
            for name in self.user_real_features
        }
        # TODO: Add Microbiome here 
        
        static_input_sizes = {**static_categorical_input_sizes, **static_real_input_sizes}
        self.static_variable_selection = TransformerVariableSelectionNetwork(
            input_sizes=static_input_sizes,
            hidden_size=self.config.hidden_dim,
            n_heads=self.config.variable_selection_network_n_heads,
            input_embedding_flags={
                name: True for name in self.user_categorical_features
            },
            dropout=self.config.dropout_rate,
            prescalers=self.prescalers,
        )

        # Define time-varying categories and reals for encoder and decoder
        self.time_varying_categoricals_encoder = self.dataset_metadata["temporal_categoricals"]
        self.time_varying_reals_encoder = self.dataset_metadata["temporal_reals"] + ["val"]  + self.dataset_metadata["food_reals"]
        
        self.time_varying_categoricals_decoder = self.dataset_metadata["temporal_categoricals"]
        self.time_varying_reals_decoder = self.dataset_metadata["temporal_reals"] + self.dataset_metadata["food_reals"]
        
        # Create MultiEmbedding for time varying categoricals if not already created
        self.temporal_categorical_embeddings = MultiEmbedding(
            embedding_sizes={
                name: (
                    len(self.dataset_metadata["categorical_encoders"][name].classes_),
                    get_embedding_size(
                        len(self.dataset_metadata["categorical_encoders"][name].classes_),
                        self.config.hidden_continuous_dim
                    )
                )
                for name in self.dataset_metadata["temporal_categoricals"]
            },
            x_categoricals=self.dataset_metadata["temporal_categoricals"],
            max_embedding_size=self.config.hidden_continuous_dim,
        ) # TODO: This needs some more thought - doesnt make sense wrt encoder/decoder yet
        
        # variable selection for encoder and decoder
        encoder_input_sizes = {
            name: self.temporal_categorical_embeddings.output_size[name]
            for name in self.time_varying_categoricals_encoder
        }
        encoder_input_sizes.update({
            name: self.config.hidden_continuous_dim
            for name in self.time_varying_reals_encoder
        })

        decoder_input_sizes = {
            name: self.temporal_categorical_embeddings.output_size[name]
            for name in self.time_varying_categoricals_decoder
        }
        decoder_input_sizes.update({
            name: self.hidden_continuous_dim
            for name in self.time_varying_reals_decoder
        })

        # create single variable grns that are shared across decoder and encoder
        if self.config.share_single_variable_networks:
            self.shared_single_variable_grns = nn.ModuleDict()
            for name, input_size in encoder_input_sizes.items():
                self.shared_single_variable_grns[name] = PreNormResidualBlock(
                    input_dim=input_size,
                    hidden_dim=min(input_size, self.config.hidden_dim),
                    output_dim=self.config.hidden_dim,
                    dropout=self.config.dropout_rate,
                )
            for name, input_size in decoder_input_sizes.items():
                if name not in self.shared_single_variable_grns:
                    self.shared_single_variable_grns[name] = PreNormResidualBlock(
                        input_dim=input_size,
                        hidden_dim=min(input_size, self.config.hidden_dim),
                        output_dim=self.config.hidden_dim,
                        dropout=self.config.dropout_rate,
                    )

        self.encoder_variable_selection = TransformerVariableSelectionNetwork(
            input_sizes=encoder_input_sizes,
            hidden_size=self.config.hidden_dim,
            n_heads=self.config.variable_selection_network_n_heads,
            input_embedding_flags={
                name: True for name in self.time_varying_categoricals_encoder
            },
            dropout=self.config.dropout_rate,
            context_size=self.config.hidden_dim,
            prescalers=self.prescalers,
            single_variable_grns=(
                {}
                if not self.config.share_single_variable_networks
                else self.shared_single_variable_grns
            ),
        )

        self.decoder_variable_selection = TransformerVariableSelectionNetwork(
            input_sizes=decoder_input_sizes,
            hidden_size=self.config.hidden_dim,
            n_heads=self.config.variable_selection_network_n_heads,
            input_embedding_flags={
                name: True for name in self.time_varying_categoricals_decoder
            },
            dropout=self.config.dropout_rate,
            context_size=self.config.hidden_dim,
            prescalers=self.prescalers,
            single_variable_grns=(
                {}
                if not self.config.share_single_variable_networks
                else self.shared_single_variable_grns
            ),
        )    
    
    def setup_output_layers(self):
        # post multihead attn processing before the output processing
        self.pos_wise_ff = PreNormResidualBlock(
            input_dim=self.config.hidden_dim,
            hidden_dim=self.config.hidden_dim,
            output_dim=self.config.hidden_dim,
            dropout=self.config.dropout_rate,
            context_dim=None,
        )
        
        self.post_attn_gate_norm = GateAddNorm(
            self.config.hidden_dim, dropout=self.config.dropout_rate, trainable_add=False
        )
        
        # output processing -> no dropout at this late stage
        self.pre_output_gate_norm = GateAddNorm(
            self.config.hidden_dim, dropout=0.0, trainable_add=False
        )
        
        # Define number of targets (default to 1)
        self.n_targets = 1
        
        # Define output size (number of quantiles)
        self.output_size = self.num_quantiles
        
        # Create output layer
        self.output_layer = nn.Linear(
            self.config.hidden_dim, self.output_size
        )

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        return_attn: bool = False,
        return_meal_self_attn: bool = False
    ) -> Union[torch.Tensor, Tuple]:
        """
        Direct implementation of TFT-style forward pass following model.py structure.
        
        Args:
            batch: Input batch dictionary
            return_attn: Whether to return attention weights
            return_meal_self_attn: Whether to return meal self-attention
            
        Returns:
            Either tensor of predictions or tuple with predictions and attention weights
        """
        # Extract key tensors from batch
        encoder_lengths = batch["encoder_lengths"]  # [B]
        decoder_lengths = torch.full_like(encoder_lengths, batch["future_meal_ids"].size(1))
        encoder_padding_mask = batch["encoder_padding_mask"]  # [B, T_past]
        target_scales = batch["target_scales"]  # [B, 2]
        
        # Original variables needed for data preparation
        past_glucose = batch["past_glucose"]  # [B, T_past]
        past_meal_ids = batch["past_meal_ids"]  # [B, T_past, M]
        past_meal_macros = batch["past_meal_macros"]  # [B, T_past, M, food_macro_dim]
        future_meal_ids = batch["future_meal_ids"]  # [B, T_future, M]
        future_meal_macros = batch["future_meal_macros"]  # [B, T_future, M, food_macro_dim]
        future_glucose = batch.get("future_glucose")  # [B, T_future], could be None during inference
        
        user_categoricals = batch["user_categoricals"]  # [B, num_user_cat_features]
        user_reals = batch["user_reals"]  # [B, num_user_real_features]
        user_microbiome_embeddings = batch.get("user_microbiome_embeddings")
        past_temporal_cats = batch["past_temporal_categoricals"] # [B, T_past, num_temporal_cat_features]
        past_temporal_reals = batch["past_temporal_reals"] # [B, T_past, num_temporal_real_features]
        future_temporal_cats = batch["future_temporal_categoricals"] # [B, T_future, num_temporal_cat_features]
        future_temporal_reals = batch["future_temporal_reals"] # [B, T_future, num_temporal_real_features]
        
        # Ensure target_scales has the right shape
        if target_scales.dim() > 2:
            target_scales = target_scales.view(target_scales.size(0), -1)
        
        # Basic properties
        device = past_glucose.device
        batch_size = past_glucose.size(0)
        T_past = past_glucose.shape[1]
        T_future = future_meal_ids.shape[1]
        max_encoder_length = T_past
        timesteps = T_past + T_future
                
        
        # Encoder Variables
        encoder_variables = [
            "target_columns",
            "temporal_categoricals",
            "temporal_reals",
            "food_reals"
        ]
        # Decoder Variables
        decoder_variables = [
            "temporal_categoricals",
            "temporal_reals",
            "food_reals"
        ]
        
        # 1. Create input_vectors dictionary
        input_vectors = {}
        
        # Gather Target Variables (Val is for the Glucose Val)
        input_vectors["val"] = torch.cat(
            [past_glucose.to(self.dtype), future_glucose.to(self.dtype)], dim=1).unsqueeze(-1)
        
        # Process temporal Variables
        temporal_cat_embeddings = self.temporal_categorical_embeddings(
                torch.cat([past_temporal_cats, future_temporal_cats], dim=1).int()
        )
        temporal_real_values = {}        
        for idx, name in enumerate(self.dataset_metadata["temporal_reals"]):
            temporal_real_values[name] = torch.cat([
                    past_temporal_reals[..., idx].to(self.dtype), 
                    future_temporal_reals[..., idx].to(self.dtype)
                ], dim=1).unsqueeze(-1)
        input_vectors.update(temporal_cat_embeddings) # Add to Inpute Vectors
        input_vectors.update(temporal_real_values) # Add to Input Vectors

        # Gather User Specific Variables
        user_static_cat_embeddings = self.user_static_categoricals_embeddings(
            torch.cat([user_categoricals], dim=1).int()
        )
        user_real_values = {
            name: user_reals[..., idx].unsqueeze(-1).to(self.dtype)
            for idx, name in enumerate(self.user_real_features)
        }
        input_vectors.update(user_static_cat_embeddings) # Add to Input Vectors
        input_vectors.update(user_real_values) # Add to Input Vectors
        
        # Gather Food Variables
        food_real_values = {}
        for idx, name in enumerate(self.dataset_metadata["food_reals"]):
            food_real_values[name] = torch.cat([
                    past_meal_macros.sum(dim=2)[:,:,0],
                    future_meal_macros.sum(dim=2)[:,:,0]
                ], dim=1).unsqueeze(-1)
        input_vectors.update(food_real_values)
                
        ## TODO: Add Meal Embeddings here

        ##########  Static Variable Selection  ##########
        # Static Variable Selection
        static_embedding_variables = {
            **user_static_cat_embeddings,
            **user_real_values
        }
        # Calculate Static Embedding
        static_embedding, static_variable_selection = self.static_variable_selection(
            static_embedding_variables
        )
        def _expand_static_context(static_context, timesteps):
            return static_context.unsqueeze(1).expand(-1, timesteps, -1)
        static_context_variable_selection = _expand_static_context(static_embedding, timesteps)

        ##########  Encoder Variable Selection  ##########
        embeddings_varying_encoder = {}
        for variable_group_name in encoder_variables:
            for variable in self.dataset_metadata[variable_group_name]:
                embeddings_varying_encoder[variable] = input_vectors[variable][:, :max_encoder_length]
                        
        embeddings_varying_encoder, encoder_sparse_weights = (
            self.encoder_variable_selection(
                embeddings_varying_encoder,
                static_context_variable_selection[:, :max_encoder_length],
            )
        )

        ##########  Decoder Variable Selection  ##########
        embeddings_varying_decoder = {}
        for variable_group_name in decoder_variables:
            for variable in self.dataset_metadata[variable_group_name]:
                embeddings_varying_decoder[variable] = input_vectors[variable][:, max_encoder_length:]  
                
        embeddings_varying_decoder, decoder_sparse_weights = (
            self.decoder_variable_selection(
                embeddings_varying_decoder,
                static_context_variable_selection[:, max_encoder_length:],
            )
        )
        

        ##########  Positional Encodings  ##########
        # Generate position indices for encoder and decoder
        past_indices = torch.arange(T_past, device=device).unsqueeze(0).expand(batch_size, -1)
        future_indices = torch.arange(T_future, device=device).unsqueeze(0).expand(batch_size, -1) + T_past
        
        # Center indices
        offset_value = max_encoder_length - 1
        centered_offset = offset_value * torch.ones((batch_size, 1), device=device, dtype=torch.long)
        past_indices = past_indices - centered_offset
        future_indices = future_indices - centered_offset
        
        # Apply rotary positional embeddings 
        embeddings_varying_encoder = self.positional_embeddings(embeddings_varying_encoder, past_indices)
        embeddings_varying_decoder = self.positional_embeddings(embeddings_varying_decoder, future_indices)
        
        
        # --------------------------------------------------------------------
        # Process inputs with Transformer
        # --------------------------------------------------------------------
        # We have "embeddings_varying_encoder" for the historical part (B x T_past x hidden)
        # and "embeddings_varying_decoder" for the future part (B x T_future x hidden).

        # Construct padding masks for the transformer (True = ignore)
        # We want shape: (B, T_past) and (B, T_future)
        encoder_padding_mask = create_mask(
            max_encoder_length, encoder_lengths
        )  # True where "padding"
        decoder_padding_mask = create_mask(
            embeddings_varying_decoder.shape[1], decoder_lengths
        )
        
        # create a causal mask for the decoder:
        T_future = embeddings_varying_decoder.shape[1]
        causal_mask = nn.Transformer().generate_square_subsequent_mask(T_future).to(
            embeddings_varying_decoder.device
        )

        # Pass through the encoder
        transformer_encoder_output = self.transformer_encoder(
            src=embeddings_varying_encoder,  # B x T_enc x hidden
            src_key_padding_mask=encoder_padding_mask,  # B x T_enc
        )  # -> B x T_enc x hidden

        # Pass through the decoder
        transformer_decoder_output = self.transformer_decoder(
            tgt=embeddings_varying_decoder,        # B x T_dec x hidden
            memory=transformer_encoder_output,                 # B x T_enc x hidden
            tgt_mask=causal_mask,                  # T_dec x T_dec, standard
            tgt_key_padding_mask=decoder_padding_mask,  # B x T_dec
            memory_key_padding_mask=encoder_padding_mask,  # B x T_enc
        )  # -> B x T_dec x hidden
        
        
        # Add post transformer gating
        transformer_output_encoder = self.post_transformer_gate_encoder(transformer_encoder_output)
        transformer_output_encoder = self.post_transformer_add_norm_encoder(
            transformer_output_encoder, embeddings_varying_encoder
        )

        transformer_output_decoder = self.post_transformer_gate_decoder(transformer_decoder_output)
        transformer_output_decoder = self.post_transformer_add_norm_decoder(
            transformer_output_decoder, embeddings_varying_decoder
        )

        transformer_output = torch.cat([transformer_output_encoder, transformer_output_decoder], dim=1)
        
        # static enrichment
        static_context_enrichment = self.static_context_enrichment(static_embedding)
        attn_input = self.static_enrichment(transformer_output,
                                            _expand_static_context(
                                                static_context_enrichment, 
                                                timesteps))

        # multihead attn over entire sequence
        attn_output, attn_output_weights = self.multihead_attn(
            q=attn_input[:, max_encoder_length:],  # only for predictions
            k=attn_input,
            v=attn_input,
            mask=self.get_attention_mask(
                encoder_lengths=encoder_lengths, decoder_lengths=decoder_lengths
            ),  # Mask with True for positions to attend to, False for positions to mask out
        )
    
                
        attn_output = self.post_attn_gate_norm(
            attn_output, attn_input[:, max_encoder_length:]
        )
        output = self.pos_wise_ff(attn_output)
        output = self.pre_output_gate_norm(
            output, transformer_output[:, max_encoder_length:]
        )
        
        # Ensure the final output layer always runs in full precision
        with torch.amp.autocast("cuda", enabled=False):
            # Final linear        
            if self.n_targets > 1:
                output = []
                for layer in self.output_layer:
                    output.append(layer(output))
            else:
                output = self.output_layer(output)

        
        # Unscale predictions to original range
        pred_future = unscale_tensor(output, target_scales)
        
        # 12. Return appropriate outputs
        if return_attn:
            # Extract attention weights for past and future
            attention_weights = attn_output_weights  # Shape: [batch_size, forecast_len, num_heads, total_timesteps]
            
            # Average across heads (this is what plot_helpers expects)
            attention_weights = attention_weights.mean(dim=2)  # Now shape: [batch_size, forecast_len, total_timesteps]

            
            attention_past = attention_weights[:, :, :max_encoder_length]
            attention_future = attention_weights[:, :, max_encoder_length:]
            
            # Meal self-attention (not used in this version)
            meal_self_attn_past = None
            meal_self_attn_future = None
            
            return (
                pred_future,
                embeddings_varying_encoder, attention_past,
                transformer_decoder_output, attention_future,
                meal_self_attn_past, meal_self_attn_future,
                encoder_sparse_weights, decoder_sparse_weights
            )
        else:
            return pred_future

    def _shared_step(self, batch, batch_idx, phase: str) -> torch.Tensor:
        """
        Shared step function for training, validation and testing.
        Similar to the step function in model.py.
        
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
            
        # Forward pass with attention weights - comparable to model.py's forward call
        preds = self(batch,
                    return_attn=True,
                    return_meal_self_attn=True)
        
        # Compute metrics - similar to model.py's step function
        metrics = self._compute_forecast_metrics(past_glucose, future_glucose, target_scales, preds)
        
        # Log metrics with phase prefix
        for key, value in metrics["metrics"].items():
            # Log with more specific parameters to ensure visibility in progress bar
            self.log(f"{phase}_{key}", value, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        # Add iAUC correlation to progress bar - custom addition beyond model.py
        if f"pred_iAUC_{self.eval_window}" in metrics and f"true_iAUC_{self.eval_window}" in metrics:
            pred_iauc = metrics[f"pred_iAUC_{self.eval_window}"]
            true_iauc = metrics[f"true_iAUC_{self.eval_window}"]
            # Calculate correlation for individual batch
            corr = calculate_correlation(pred_iauc, true_iauc)
            # Log the correlation
            self.log(
                f"{phase}_iAUC_eh{self.eval_window}_correlation",
                corr.item(),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True
            )
        
        # Store outputs for phase end processing - similar to model.py's step outputs collection
        if not hasattr(self, f"{phase}_outputs"):
            setattr(self, f"{phase}_outputs", [])
            
        getattr(self, f"{phase}_outputs").append({
            f"{phase}_q_loss": metrics["metrics"]["q_loss"],
            f"{phase}_pred_iAUC_{self.eval_window}": metrics[f"pred_iAUC_{self.eval_window}"],
            f"{phase}_true_iAUC_{self.eval_window}": metrics[f"true_iAUC_{self.eval_window}"],
        })
        
        # Store example data for plotting on first batch - similar to model.py's plotting
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
        
        return metrics["metrics"]["total_loss"]

    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        Similar to model.py's validation_step.
        """
        return self._shared_step(batch, batch_idx, "val")

    def on_validation_epoch_end(self):
        """
        End of validation epoch processing.
        Similar to model.py's on_validation_epoch_end but with custom plotting.
        """
        self._shared_phase_end("val")

    def test_step(self, batch, batch_idx):
        """
        Test step.
        Similar to model.py's test_step.
        """
        return self._shared_step(batch, batch_idx, "test")

    def on_test_epoch_end(self):
        """
        End of test epoch processing.
        Similar to model.py's on_test_epoch_end but with custom plotting.
        """
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
            # Use the format train_X for consistency with val/test
            self.log(f"train_{key}", value, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
                        
        # Add iAUC metrics to progress bar if available
        if f"pred_iAUC_{self.eval_window}" in metrics and f"true_iAUC_{self.eval_window}" in metrics:
            pred_iauc = metrics[f"pred_iAUC_{self.eval_window}"]
            true_iauc = metrics[f"true_iAUC_{self.eval_window}"]
            
            # Calculate correlation
            corr = calculate_correlation(pred_iauc, true_iauc)
            
            # Log the correlation
            self.log(
                f"train_iAUC_eh{self.eval_window}_correlation",
                corr.item(),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True
            )
            
            # Also log iAUC loss metrics
            self.log(
                f"train_iAUC_eh{self.eval_window}_loss",
                metrics["metrics"][f"iAUC_eh{self.eval_window}_loss"],
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True
            )
            
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
                f"rmse_eh{self.eval_window}": rmse,
                f"mae_eh{self.eval_window}": mae,
                f"mape_eh{self.eval_window}": mape,
                f"smape_eh{self.eval_window}": smape,
                f"iAUC_eh{self.eval_window}_loss": iAUC_loss,
                f"iAUC_eh{self.eval_window}_weighted_loss": weighted_iAUC_loss,
                "total_loss": total_loss,
            },
            f"pred_iAUC_{self.eval_window}": pred_iAUC,
            f"true_iAUC_{self.eval_window}": true_iAUC,        
        }
    
    def configure_optimizers(self):
        assert self.config.optimizer == "adamw", "Only adamw optimizer is supported atm"
        assert self.config.optimizer_lr_scheduler in ["onecycle", "none"], "Only onecycle scheduler is supported atm or none or auto"
        
        assert self.config.optimizer_lr_scheduler_max_lr_multiplier >= 1.0, "lr_scheduler_max_lr_multiplier must be >= 1.0"
        
        optimizer = torch.optim.AdamW(self.parameters(), 
                          lr=self.config.optimizer_lr, 
                          weight_decay=self.config.optimizer_weight_decay)

        # Return only the optimizer if lr_scheduler is "none"
        if self.config.optimizer_lr_scheduler == "none":
            return optimizer
        
        # Estimate total steps if trainer is not attached yet
        total_steps = self.trainer.estimated_stepping_batches
        
        lr_scheduler = {}
        if self.config.optimizer_lr_scheduler == "onecycle":
            # Configure the OneCycleLR scheduler
            lr_scheduler = {
                'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=self.config.optimizer_lr * self.config.optimizer_lr_scheduler_max_lr_multiplier,           # Peak learning rate during the cycle
                    total_steps=total_steps,  # Total number of training steps
                    pct_start=self.config.optimizer_lr_scheduler_pct_start,         # Fraction of steps spent increasing the LR
                    anneal_strategy=self.config.optimizer_lr_scheduler_anneal_strategy, # Cosine annealing for LR decay
                    cycle_momentum=self.config.optimizer_lr_scheduler_cycle_momentum   # Set to True if you wish to cycle momentum
                ),
                'interval': 'step',        # Update the scheduler every training step
            }

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler
        }    
    
    
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
            # model.meal_encoder._bootstrap_food_id_embeddings(
            #     pretrained_weights, 
            #     freeze_embeddings=config.freeze_food_id_embeddings
            # )
            logger.info("Ignoring loading food embeddings until we bring back the MealEncoder")
            
        return model

    def _shared_phase_end(self, phase: str) -> None:
        """
        Shared epoch end processing for all phases.
        Similar to model.py's on_validation/test_epoch_end with
        additional visualization capabilities.
        
        Args:
            phase: Phase name ('train', 'val', or 'test')
        """
        outputs = getattr(self, f"{phase}_outputs", [])
        if len(outputs) == 0:
            return

        # Generate plots if not disabled and we have examples
        # This is an enhancement over model.py's visualization
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
            
            # Plot forecast examples - similar to model.py's plot_prediction
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
        # This is an enhancement beyond model.py's capabilities
        all_pred_iAUC = torch.cat([output[f"{phase}_pred_iAUC_{self.eval_window}"] for output in outputs], dim=0)
        all_true_iAUC = torch.cat([output[f"{phase}_true_iAUC_{self.eval_window}"] for output in outputs], dim=0)
        fig_scatter, corr = plot_iAUC_scatter(all_pred_iAUC, all_true_iAUC, getattr(self.config, 'disable_plots', False))
        if fig_scatter is not None:
            self.logger.experiment.log({
                f"{phase}_iAUC_eh{self.eval_window}_scatter": wandb.Image(fig_scatter),
            })
            plt.close(fig_scatter)
        
        # Log metrics - similar to model.py's logging
        self.logger.experiment.log({f"{phase}_iAUC_eh{self.eval_window}_correlation": corr.item()})
        
        # Also log epoch-level metrics for the progress bar with highlight for important ones
        self.log(f"{phase}_epoch_iAUC_eh{self.eval_window}_correlation", corr.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        # Calculate and log average q_loss at epoch level
        avg_q_loss = torch.stack([output[f"{phase}_q_loss"] for output in outputs]).mean()
        self.log(f"{phase}_epoch_q_loss", avg_q_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        # Clear outputs for the next epoch
        setattr(self, f"{phase}_outputs", [])

    def get_attention_mask(
        self, encoder_lengths: torch.LongTensor, decoder_lengths: torch.LongTensor
    ):
        """
        Returns causal mask to apply for self-attention layer.
        Critical to ensure that the decoder does not attend to future steps
        and unknowingly lead to information leakage. 
        
        Taken from:
            https://github.com/sktime/pytorch-forecasting/blob/5685c59f13aaa6aaba7181430272819c11fe7725/pytorch_forecasting/models/temporal_fusion_transformer/_tft.py#L459
        """
        decoder_length = decoder_lengths.max()
        # indices to which is attended
        attend_step = torch.arange(decoder_length, device=self.device)
        # indices for which is predicted
        predict_step = torch.arange(0, decoder_length, device=self.device)[:, None]
        # do not attend to steps to self or after prediction
        decoder_mask = (
            (attend_step >= predict_step)
            .unsqueeze(0)
            .expand(encoder_lengths.size(0), -1, -1)
        )

        # do not attend to steps where data is padded
        encoder_mask = (
            create_mask(encoder_lengths.max(), encoder_lengths)
            .unsqueeze(1)
            .expand(-1, decoder_length, -1)
        )
        # combine masks along attended time - first encoder and then decoder
        mask = torch.cat(
            (
                encoder_mask,
                decoder_mask,
            ),
            dim=2,
        )
        return mask
