"""
Loss functions and metrics for the glucose forecast model.
"""
import torch
import torch.nn.functional as F


def quantile_loss(predictions: torch.Tensor, targets: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    """
    Compute the quantile loss for multiple quantiles.
    
    Args:
        predictions: Tensor of shape (B, T, num_quantiles)
        targets: Tensor of shape (B, T)
        quantiles: Tensor of shape (num_quantiles,)
        
    Returns:
        Scalar loss value
    """
    targets_expanded = targets.unsqueeze(-1)
    errors = targets_expanded - predictions
    losses = torch.max((quantiles - 1) * errors, quantiles * errors)
    return losses.mean()


def compute_iAUC(
    median_pred: torch.Tensor, 
    future_glucose: torch.Tensor, 
    past_glucose: torch.Tensor, 
    target_scales: torch.Tensor, 
    eval_window: int = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the incremental Area Under the Curve (iAUC) for glucose predictions.
    
    Args:
        median_pred: Median prediction tensor (B, T)
        future_glucose: Future glucose values tensor (B, T)
        past_glucose: Past glucose values tensor (B, T_past)
        target_scales: Scaling factors (B, 2) for min/max
        eval_window: Number of time steps to evaluate (optional)
        
    Returns:
        Tuple of predicted iAUC and true iAUC tensors (B,)
    """
    # Unscale past glucose
    past_glucose_unscaled = past_glucose * target_scales[:, 1].unsqueeze(1) + target_scales[:, 0].unsqueeze(1)
    
    # Unscale future glucose
    future_glucose_unscaled = future_glucose * target_scales[:, 1].unsqueeze(1) + target_scales[:, 0].unsqueeze(1)
    
    # Apply evaluation window if specified
    if eval_window is not None:
        median_pred_eval = median_pred[:, :eval_window]
        future_glucose_unscaled_eval = future_glucose_unscaled[:, :eval_window]
    else:
        median_pred_eval = median_pred
        future_glucose_unscaled_eval = future_glucose_unscaled
    
    # Calculate baseline as the mean of the last two past glucose readings
    baseline = past_glucose_unscaled[:, -2:].mean(dim=1)
    
    # Calculate differences from baseline
    pred_diff = median_pred_eval - baseline.unsqueeze(1)
    true_diff = future_glucose_unscaled_eval - baseline.unsqueeze(1)
    
    # Compute iAUC using trapezoidal rule, ignoring values below baseline
    pred_iAUC = torch.trapz(torch.clamp(pred_diff, min=0), dx=1, dim=1)
    true_iAUC = torch.trapz(torch.clamp(true_diff, min=0), dx=1, dim=1)
    
    return pred_iAUC, true_iAUC


def unscale_tensor(tensor: torch.Tensor, target_scales: torch.Tensor) -> torch.Tensor:
    """
    Unscale a tensor using the provided scaling factors.
    
    Args:
        tensor: Tensor to unscale
        target_scales: Scaling factors (B, 2) containing [mean, std] for each batch item
        
    Returns:
        Unscaled tensor
    """
    # Check if inputs contain NaNs
    if torch.isnan(tensor).any():
        # Replace NaNs with zeros
        tensor = torch.nan_to_num(tensor, nan=0.0)
        
    if torch.isnan(target_scales).any():
        # Replace NaNs with reasonable defaults (no scaling)
        target_scales = torch.nan_to_num(target_scales, nan=1.0)
    
    # Extract scale parameters
    mean_val = target_scales[:, 0]
    std_val = target_scales[:, 1]
    
    # Handle division by zero or very small values in standard deviation
    # Ensure std_val doesn't contain zeros, negatives, or extremely small values
    eps = 1e-6
    std_val = torch.clamp(std_val, min=eps)
    
    # Handle different tensor shapes
    if tensor.dim() == 2:  # (B, T)
        unscaled = tensor * std_val.unsqueeze(1) + mean_val.unsqueeze(1)
    elif tensor.dim() == 3:  # (B, T, Q)
        unscaled = tensor * std_val.unsqueeze(1).unsqueeze(2) + mean_val.unsqueeze(1).unsqueeze(2)
    else:
        raise ValueError(f"Unsupported tensor dimension: {tensor.dim()}")
    
    # Clip extreme values to prevent NaNs in subsequent calculations
    # Glucose values typically range from ~40 to ~400 mg/dL
    # Or approximately 2.5 to 22 mmol/L in SI units
    min_glucose = 2.0   # Below physiological range but allow for some prediction error
    max_glucose = 30.0  # Above physiological range but allow for some prediction error
    
    return torch.clamp(unscaled, min=min_glucose, max=max_glucose)


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