"""
Utility functions for models.
"""
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Union, Tuple, Optional
from torchmetrics.functional.regression import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    symmetric_mean_absolute_percentage_error,
)
from .losses import quantile_loss, compute_iAUC, unscale_tensor

def expand_user_embeddings(
    user_embeddings: torch.Tensor, 
    timesteps: int, 
    user_categorical_features: List[str],
    user_real_features: List[str],
    use_projected_user_features: bool,
    prefix: str = "user_"
) -> Dict[str, torch.Tensor]:
    """
    Helper function to expand user embeddings across time and return a dictionary of expanded embeddings.
    
    Args:
        user_embeddings: User embeddings tensor of shape [B, num_features, hidden_dim] or [B, 1, hidden_dim]
        timesteps: Number of timesteps to expand the embeddings to
        user_categorical_features: List of categorical feature names
        user_real_features: List of real-valued feature names
        use_projected_user_features: Whether user features are projected to a single vector
        prefix: Prefix to use for the user feature keys (default: "user_")
        
    Returns:
        Dictionary of expanded user embeddings
    """
    user_inputs = {}
    
    # Check if user embeddings are projected to a single vector
    if use_projected_user_features:
        # When projected, user_embeddings has shape [B, 1, hidden_dim]
        # Expand to match timesteps
        user_per_timestep = user_embeddings.expand(-1, timesteps, -1)  # [B, timesteps, hidden_dim]
        user_inputs["user"] = user_per_timestep
    else:
        # When not projected, user_embeddings has shape [B, num_features, hidden_dim]
        # Expand all user features at once
        # [B, num_features, hidden_dim] -> [B, num_features, timesteps, hidden_dim]
        user_embeddings_expanded = user_embeddings.unsqueeze(2).expand(-1, -1, timesteps, -1)
        
        # Add categorical features
        for i, feature_name in enumerate(user_categorical_features):
            # Just assign the already expanded embedding
            user_inputs[f"{prefix}{feature_name}"] = user_embeddings_expanded[:, i]  # [B, timesteps, hidden_dim]
            
        # Add real features
        num_cat_features = len(user_categorical_features)
        for i, feature_name in enumerate(user_real_features):
            # Real features start after categorical features
            feature_idx = i + num_cat_features
            # Just assign the already expanded embedding
            user_inputs[f"{prefix}{feature_name}"] = user_embeddings_expanded[:, feature_idx]  # [B, timesteps, hidden_dim]
    
    return user_inputs

def get_user_context(
    user_embeddings: torch.Tensor, 
    use_projected_user_features: bool
) -> torch.Tensor:
    """
    Calculate user context for VSN from user embeddings.
    
    Args:
        user_embeddings: User embeddings tensor of shape [B, num_features, hidden_dim] or [B, 1, hidden_dim]
        use_projected_user_features: Whether user features are projected to a single vector
        
    Returns:
        User context tensor of shape [B, hidden_dim]
    """
    if use_projected_user_features:
        # When projected, squeeze out the singleton dimension
        return user_embeddings.squeeze(1)  # [B, hidden_dim]
    else:
        # When not projected, take the mean across features
        return user_embeddings.mean(dim=1)  # [B, hidden_dim]

def get_attention_mask(
    forecast_horizon: torch.LongTensor, 
    T_past: int,
    T_future: int,
    device: torch.device
) -> torch.Tensor:
    """
    Returns causal mask to apply for cross-attention in the transformer decoder.
    
    This mask ensures that predictions for position i can only attend to:
    1. All historical data (always accessible to all decoder positions)
    2. Future meal data only up to position i (to prevent future information leakage)
    
    This enforces causality in the time dimension, similar to the TFT implementation.
    
    Args:
        forecast_horizon: Length of the prediction horizon
        T_past: Length of the historical/encoder sequence
        T_future: Length of the future sequence with known features (e.g., meals)
        device: Device to create the mask on
        
    Returns:
        Attention mask of shape [forecast_horizon, T_past+T_future] where values of
        float('-inf') indicate positions that should be masked out during attention.
    """
    decoder_length = forecast_horizon
    
    # Indices to which each query position can attend
    attend_step = torch.arange(T_past + T_future, device=device)
    
    # Indices for which predictions are made
    predict_step = torch.arange(0, decoder_length, device=device)[:, None]
    
    # Create mask where True means position can be attended to
    mask = torch.ones((decoder_length, T_past + T_future), device=device)
    
    # Allow all positions to attend to past
    mask[:, :T_past] = 1.0
    
    # For future positions, create causal mask
    future_mask = attend_step[None, T_past:] <= predict_step + T_past
    mask[:, T_past:] = future_mask.float()
    
    # Convert to format expected by attention mechanism
    # (0 -> -inf, 1 -> 0)
    mask = mask.masked_fill(mask == 0, float('-inf'))
    
    return mask

def compute_forecast_metrics(
    past_glucose: torch.Tensor, 
    future_glucose: torch.Tensor, 
    target_scales: torch.Tensor, 
    preds: Union[torch.Tensor, Tuple],
    quantiles: torch.Tensor,
    eval_window: int,
    loss_iauc_weight: float = 0.0
) -> Dict[str, Any]:
    """
    Compute metrics for forecasting performance.
    
    Args:
        past_glucose: Past glucose values
        future_glucose: Future glucose values (target)
        target_scales: Scaling factors
        preds: Model predictions (either tensor or tuple with tensor as first element)
        quantiles: Tensor of quantile values
        eval_window: Evaluation window size
        loss_iauc_weight: Weight for iAUC loss in total loss
        
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
    q_loss = quantile_loss(predictions, future_glucose_unscaled, quantiles)
    
    # Compute RMSE with median predictions
    median_idx = len(quantiles) // 2
    median_pred = predictions[:, :, median_idx]
    median_pred_eval = median_pred[:, :eval_window]
    future_glucose_unscaled_eval = future_glucose_unscaled[:, :eval_window]
    
    # Flatten predictions and targets to 1D contiguous tensors to avoid view errors in torchmetrics
    pred_flat = median_pred_eval.contiguous().reshape(-1)
    target_flat = future_glucose_unscaled_eval.contiguous().reshape(-1)
    rmse = mean_squared_error(pred_flat, target_flat, squared=False)
    mae = mean_absolute_error(pred_flat, target_flat)
    mape = mean_absolute_percentage_error(pred_flat, target_flat)
    smape = symmetric_mean_absolute_percentage_error(pred_flat, target_flat)
    
    # Compute iAUC metrics
    pred_iAUC, true_iAUC = compute_iAUC(
        median_pred, future_glucose, past_glucose, target_scales, eval_window=eval_window
    )
    
    # Compute losses
    iAUC_loss = F.mse_loss(pred_iAUC, true_iAUC)
    weighted_iAUC_loss = loss_iauc_weight * iAUC_loss
    total_loss = q_loss + weighted_iAUC_loss
    
    return {
        "metrics": {
            "q_loss": q_loss,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "smape": smape,
            f"iAUC_eh{eval_window}_loss": iAUC_loss,
            f"iAUC_eh{eval_window}_weighted_loss": weighted_iAUC_loss,
            "total_loss": total_loss,
        },
        f"pred_iAUC_{eval_window}": pred_iAUC,
        f"true_iAUC_{eval_window}": true_iAUC,        
    }

def log_fusion_feature_weights(
    feature_weights: torch.Tensor,
    base_modalities_count: int,
    user_categorical_features: List[str],
    user_real_features: List[str],
    use_projected_user_features: bool,
    logger_fn=None,
    prefix: str = "",
    base_modality_names: List[str] = None
) -> Dict[str, float]:
    """
    Calculate and log fusion feature weights for both VSN and CrossModalFusion blocks.
    
    This function works with weights from either VariableSelectionNetwork or 
    CrossModalFusion blocks, as they both use a standardized format.
    
    Args:
        feature_weights: The weights from the fusion block [B, T, num_features]
        base_modalities_count: Number of base modalities (e.g., glucose, meal, temporal, microbiome)
        user_categorical_features: List of categorical feature names
        user_real_features: List of real-valued feature names
        use_projected_user_features: Whether user features are projected to a single vector
        logger_fn: Function to log the weights (e.g., self.log). If None, weights are only returned
        prefix: Prefix for logging
        base_modality_names: Optional list of base modality names. If None, defaults to ["glucose", "meal", "temporal", "microbiome"]
        
    Returns:
        Dictionary of calculated feature weights
    """
    weight_dict = {}
    
    # Base modalities first - use provided names or default
    if base_modality_names is None:
        # Default modality names, up to the number of base modalities we have
        all_modalities = ["glucose", "meal", "temporal", "microbiome"]
        base_modality_names = all_modalities[:min(len(all_modalities), base_modalities_count)]
    
    # Log base modality weights
    for i, name in enumerate(base_modality_names):
        if i < feature_weights.shape[2]:  # Make sure we don't exceed the number of features
            weight = feature_weights[:, :, i].mean().item()
            weight_dict[f"{prefix}_{name}_weight"] = weight
            if logger_fn:
                logger_fn(f"{prefix}_{name}_weight", weight)
    
    # Total number of user features
    num_user_features = len(user_categorical_features) + len(user_real_features)
    
    # Compute user feature weights
    if use_projected_user_features:
        # In single vector mode, there's only one user feature column to log
        # It comes right after the base modalities
        user_feature_idx = base_modalities_count
        if user_feature_idx < feature_weights.shape[2]:  # Check bounds
            weight = feature_weights[:, :, user_feature_idx].mean().item()
            weight_dict[f"{prefix}_user_aggregate_weight"] = weight
            if logger_fn:
                logger_fn(f"{prefix}_user_aggregate_weight", weight)
    else:
        # In multi-vector mode, user features are individual columns
        user_feature_start_idx = base_modalities_count
        user_feature_end_idx = user_feature_start_idx + num_user_features
        
        # If we have enough weights columns
        if user_feature_end_idx <= feature_weights.shape[2]:
            user_weights = feature_weights[:, :, user_feature_start_idx:user_feature_end_idx]
            agg_weight = user_weights.mean().item()
            weight_dict[f"{prefix}_user_aggregate_weight"] = agg_weight
            if logger_fn:
                logger_fn(f"{prefix}_user_aggregate_weight", agg_weight)
    
    return weight_dict

# Keep backward compatibility with old function name
log_vsn_feature_weights = log_fusion_feature_weights 

def expand_user_embeddings_for_fusion(
    user_embeddings: torch.Tensor,
    T_past: int,
    T_future: int,
    user_categorical_features: List[str],
    user_real_features: List[str],
    project_to_single_vector: bool,
    user_prefix: str = 'user_'
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Expand user embeddings across time dimensions for fusion blocks.
    
    This function takes pre-computed user embeddings and expands them across
    the time dimension for both past and future sequences. It handles both
    single-vector projection and individual feature cases.
    
    Args:
        user_embeddings: User embeddings tensor of shape [B, num_features, hidden_dim] or [B, hidden_dim]
        T_past: Length of past sequence
        T_future: Length of future sequence
        user_categorical_features: List of categorical feature names
        user_real_features: List of real-valued feature names
        project_to_single_vector: Whether to project all user features to a single vector
        user_prefix: Prefix to use for user feature keys
        
    Returns:
        Tuple of (past_user_dict, future_user_dict) containing expanded user embeddings
        ready to update into modality dictionaries
    """
    past_user_dict = {}
    future_user_dict = {}
    
    # Process user embeddings based on projection setting
    if project_to_single_vector:
        # Project all user features to a single vector
        if user_embeddings.dim() == 3:  # [B, num_features, hidden_dim]
            user_emb_combined = user_embeddings.mean(dim=1, keepdim=True)  # [B, 1, hidden_dim]
        else:  # Already [B, hidden_dim]
            user_emb_combined = user_embeddings.unsqueeze(1)  # [B, 1, hidden_dim]
            
        # Expand to match sequence lengths
        past_user_dict['user'] = user_emb_combined.expand(-1, T_past, -1)  # [B, T_past, hidden_dim]
        future_user_dict['user'] = user_emb_combined.expand(-1, T_future, -1)  # [B, T_future, hidden_dim]
    else:
        # Add each user feature as a separate modality
        for i, feature in enumerate(user_categorical_features + user_real_features):
            # Get the feature embedding
            feature_emb = user_embeddings[:, i, :].unsqueeze(1)  # [B, 1, hidden_dim]
            
            # Expand to match sequence lengths
            past_user_dict[f'{user_prefix}{feature}'] = feature_emb.expand(-1, T_past, -1)  # [B, T_past, hidden_dim]
            future_user_dict[f'{user_prefix}{feature}'] = feature_emb.expand(-1, T_future, -1)  # [B, T_future, hidden_dim]
    
    return past_user_dict, future_user_dict 