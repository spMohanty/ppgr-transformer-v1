import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import argparse
from typing import Optional, Union, Dict, List, Tuple

import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

def plot_forecast(
    true_history: np.ndarray,
    true_future: Optional[np.ndarray],
    median_forecast: np.ndarray,
    quantile_forecasts: np.ndarray,
    encoder_attention_map: Optional[np.ndarray] = None,
    decoder_attention_map: Optional[np.ndarray] = None,
    meal_flags: Optional[np.ndarray] = None,
    loss_value: Optional[float] = None,
    show_observed_future: bool = True,
) -> plt.Figure:
    """
    Plot historical + observed + forecast with uncertainty bands and optional attention heatmap.
    
    Parameters
    ----------
    true_history : 1D array, length = H
        The observed values before t=0.
    true_future : 1D array or None, length = F
        The actual observed values after t=0 (if available).
    median_forecast : 1D array, length = F
        The median forecast values.
    quantile_forecasts : 2D array, shape = (F, Q)
        The quantile forecasts (each column is one quantile level).
    encoder_attention_map : 2D array or None, shape = (F, H)
        If provided, shown as a heatmap below the time series.
    decoder_attention_map : 2D array or None, shape = (F, F)
        If provided, shown as a heatmap below the time series.
    meal_flags : 1D bool array or None, length = H+F
        If provided, draws vertical tick at each True.
    loss_value : float or None
        If provided, appended to the title.
    show_observed_future : bool
        Whether to plot `true_future` (if given).
    """
    # — Convert torch tensors to numpy arrays if needed —
    def convert_to_numpy(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x
    
    true_history = convert_to_numpy(true_history)
    true_future = convert_to_numpy(true_future)
    median_forecast = convert_to_numpy(median_forecast)
    quantile_forecasts = convert_to_numpy(quantile_forecasts)
    encoder_attention_map = convert_to_numpy(encoder_attention_map)
    decoder_attention_map = convert_to_numpy(decoder_attention_map)
    meal_flags = convert_to_numpy(meal_flags)
    
    # — style & palette —
    sns.set_style("white")
    PALETTE = {
        "historical":  "#2E4057",
        "observed":    "#CB4335",
        "forecast":    "#2471A3",
        "uncertainty":"#2471A3",
        "meal":        "#7D3C98",
        "attention": [
            "#1B1B2F",  # dark base
            "#3C2C54",  # plum
            "#722744",  # red-violet
            "#A7442A",  # copper rose
            "#DA7422",  # amber
            "#F6B78C",  # peach glow
            "#FBE3D1", # champagne
        ],        
    }
    
    attention_cmap = LinearSegmentedColormap.from_list(
        "attention_cmap", PALETTE["custom_attention"], N=512
    )
    mpl.rcParams.update({
        'font.family':'serif',
        'font.serif':['DejaVu Serif','Liberation Serif','serif'],
        'font.size':10,
        'axes.titlesize':12,
        'axes.labelsize':11,
        'xtick.labelsize':10,
        'ytick.labelsize':10,
        'legend.fontsize':10,
        'axes.linewidth':1.2,
        'xtick.direction':'in',
        'ytick.direction':'in',
        'xtick.major.width':1.0,
        'ytick.major.width':1.0,
    })

    # — unwrap shapes —
    H = true_history.size
    F = median_forecast.size
    total_len = H + F

    # — build index vectors —
    hist_idx = np.arange(H)
    fut_idx  = H + np.arange(F)
    combined_idx = np.arange(total_len)

    # — relative time labels —
    rel_steps = np.concatenate([np.arange(-H+1, 1), np.arange(1, F+1)])
    zero_pos  = int(np.where(rel_steps == 0)[0][0])
    
    # Generate ticks from -H to F in steps of 4, ensuring 0 is included
    neg_ticks = np.arange(0, -H, -4)[::-1]  # From 0 to -H in steps of -4, then reverse
    if neg_ticks[0] != 0:  # Ensure 0 is included
        neg_ticks = np.concatenate([[0], neg_ticks])
    pos_ticks = np.arange(4, F+1, 4)  # From 4 to F in steps of 4
    tick_rel = np.concatenate([neg_ticks, pos_ticks])
    
    tick_pos  = zero_pos + tick_rel
    valid     = (tick_pos >= 0) & (tick_pos < total_len)
    tick_pos  = tick_pos[valid]
    tick_lbl  = tick_rel[valid]
    

    # — start figure —
    fig = plt.figure(figsize=(12, 8), constrained_layout=False)
    nrows = 2 if encoder_attention_map is not None else 1
    ncols = 2 if encoder_attention_map is not None else 1  # Add an extra column for the colorbar
    height_ratios = [3, 1] if encoder_attention_map is not None else [1]
    width_ratios = [60, 1] if encoder_attention_map is not None else [1]  # Main plot gets 20x width of colorbar

    if encoder_attention_map is not None:
        # Create grid with extra column for colorbar
        gs = gridspec.GridSpec(nrows, ncols, height_ratios=height_ratios, width_ratios=width_ratios,
                               hspace=0.3, wspace=0.02, figure=fig)
        ax_ts = fig.add_subplot(gs[0, 0])  # Time series in top-left
        ax_at = fig.add_subplot(gs[1, 0], sharex=ax_ts)  # Attention map in bottom-left
        cax = fig.add_subplot(gs[1, 1])  # Colorbar in bottom-right
    else:
        # Original layout when there's no attention map
        gs = gridspec.GridSpec(nrows, 1, height_ratios=height_ratios,
                               hspace=0.3, figure=fig)
        ax_ts = fig.add_subplot(gs[0])
        ax_at = None
        cax = None

    # — plot history —
    ax_ts.plot(hist_idx, true_history,
               color=PALETTE["historical"], lw=2, marker="o", ms=6,
               markerfacecolor="white", markeredgecolor=PALETTE["historical"],
               label="Historical")

    # — plot observed future —
    if show_observed_future and (true_future is not None) and true_future.size:
        ext_i = np.concatenate([[H-1], fut_idx])
        ext_v = np.concatenate([[true_history[-1]], true_future])
        ax_ts.plot(ext_i, ext_v,
                   color=PALETTE["observed"], lw=2, marker="o", ms=6,
                   markerfacecolor="white", markeredgecolor=PALETTE["observed"],
                   label="Observed")

    # — plot median forecast —
    ext_i = np.concatenate([[H-1], fut_idx])
    ext_med = np.concatenate([[true_history[-1]], median_forecast])
    ax_ts.plot(ext_i, ext_med,
               color=PALETTE["forecast"], lw=2, ls="--", label="Forecast")

    # — plot uncertainty bands —
    # Flag to ensure we only add the quantile explanation to the legend once
    quantile_legend_added = False
    
    for q in range(quantile_forecasts.shape[1] - 1):
        low  = np.minimum(quantile_forecasts[:, q],   quantile_forecasts[:, q+1])
        high = np.maximum(quantile_forecasts[:, q], quantile_forecasts[:, q+1])
        alpha = 0.05 + abs(q - (quantile_forecasts.shape[1]//2))*0.02
        
        # Only add the legend entry for the middle quantile (most visible)
        label = None
        if q == quantile_forecasts.shape[1]//2 and not quantile_legend_added:
            label = "Forecast Quantiles"
            quantile_legend_added = True
        
        ax_ts.fill_between(
            ext_i,
            np.concatenate([[true_history[-1]], low]),
            np.concatenate([[true_history[-1]], high]),
            color=PALETTE["uncertainty"], alpha=alpha,
            label=label
        )

    # — meal event lines —
    if meal_flags is not None:
        for pos, flag in enumerate(meal_flags):
            if flag:
                ax_ts.axvline(pos, ls=":", lw=1.2, color=PALETTE["meal"],
                              label="Meal"
                                if "Meal" not in ax_ts.get_legend_handles_labels()[1]
                                else None)

    # — axes formatting —
    ax_ts.set_xticks(tick_pos)
    ax_ts.set_xticklabels(tick_lbl)
    ax_ts.axvline(zero_pos, color="black", lw=1.2, label="t=0")
    ax_ts.set_ylabel("Glucose (mmol/L)")
    ax_ts.legend(loc="upper left", frameon=True, framealpha=0.6, facecolor='white', edgecolor='none', 
                handlelength=1.5, handletextpad=0.5)
    for spine in ["top", "right"]:
        ax_ts.spines[spine].set_visible(False)

    # — title with optional loss —
    title = "Glucose Forecast"
    if loss_value is not None:
        title += f" (Loss: {loss_value:.3f})"
    ax_ts.set_title(title)

    # — attention heatmap —
    if encoder_attention_map is not None and decoder_attention_map is not None and ax_at is not None:
        attention_map = np.concatenate([encoder_attention_map, decoder_attention_map], axis=1)
        # Plot the attention map
        im = ax_at.imshow(attention_map, aspect="auto", origin="lower", cmap=attention_cmap)
        ax_at.set_ylabel("Forecast Step")
        ax_at.set_xlabel("Relative Timestep\n(15 min intervals)")
        ax_at.set_xticks(tick_pos)
        ax_at.set_xticklabels(tick_lbl)
        y_ticks = np.arange(0, attention_map.shape[0], 4)
        ax_at.set_yticks(y_ticks)
        ax_at.set_yticklabels(y_ticks + 1)
        for spine in ["top", "right"]:
            ax_at.spines[spine].set_visible(False)
        ax_at.spines["left"].set_edgecolor("grey")
        ax_at.spines["bottom"].set_visible(False)  # Hide the bottom (x-axis) line
        ax_at.text(
            0.98, 0.02, "Attention Map", transform=ax_at.transAxes,
            ha="right", va="bottom",
            fontsize=mpl.rcParams["legend.fontsize"],
            family=mpl.rcParams["font.family"],
            color="white", alpha=0.7
        )
        
        # Use the dedicated colorbar axis created in the GridSpec
        cbar = plt.colorbar(im, cax=cax)
        cbar.outline.set_visible(False)  # Remove the boundary/outline of the colorbar
        cax.set_ylabel("Attention Weight", fontsize=mpl.rcParams["axes.labelsize"]-1)
        cax.tick_params(labelsize=mpl.rcParams["xtick.labelsize"]-1)
        # Keep the tick labels but remove the tick markers
        cbar.ax.tick_params(size=0)  # This removes just the tick marks while keeping labels
    else:
        # When we only have the time series plot
        fig.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.15)

    return [fig]



def plot_forecast_examples(
    forecasts, 
    attn_weights_past, 
    attn_weights_future, 
    quantiles, 
    logger, 
    global_step, 
    fixed_indices=None
):
    """
    Plot historical glucose (past), predicted forecast (pred) with
    quantiles, and ground truth (truth). Optionally overlay vertical lines
    showing meal consumption times, and display up to two attention
    weight heatmaps (past vs future).

    Args:
        forecasts (dict): Must contain:
            "past":  shape (B, T_past)
            "pred":  shape (B, T_forecast, Q)
            "truth": shape (B, T_forecast)
            "future_meal_ids": shape (B, T_futureMeal, M)
            "past_meal_ids":   shape (B, T_pastMeal, M)
        attn_weights_past (torch.Tensor or None):
            shape (B, T_context, T_meals) or similar
            If None, the "past" attention heatmap is skipped.
        attn_weights_future (torch.Tensor or None):
            shape (B, T_context, T_meals) or similar
            If None, the "future" attention heatmap is skipped.
        quantiles (torch.Tensor or np.array): The list of quantiles used,
            e.g. [0.05, 0.1, ..., 0.95].
        logger: A logger object that supports logger.experiment.log().
        global_step (int): The current global step for logging.
        fixed_indices (list[int], optional): If provided, plot these exact
            sample indices from the batch. Otherwise, randomly sample.
    
    Returns:
        fixed_indices (list[int]): The indices actually plotted.
        fig (matplotlib.figure.Figure): The created figure.
    """
    past = forecasts["past"]
    pred = forecasts["pred"]
    truth = forecasts["truth"]
    meal_ids_future = forecasts["future_meal_ids"]
    meal_ids_past = forecasts["past_meal_ids"]

    # Decide how many examples to show
    num_examples = min(4, past.size(0))
    if fixed_indices is None:
        fixed_indices = random.sample(list(range(past.size(0))), num_examples)
    sampled_indices = fixed_indices


    figs = []
    for sampled_idx in sampled_indices:
        
        median_index = pred[sampled_idx].shape[1] // 2
        median_forecast = pred[sampled_idx][:, median_index]
        
        meal_flags = torch.concat(
            (meal_ids_past[sampled_idx].sum(axis=1) > 0, 
                meal_ids_future[sampled_idx].sum(axis=1) > 0), 
            axis=0
        )
        
        figs += plot_forecast(
                true_history=past[sampled_idx], 
                true_future=truth[sampled_idx], 
                median_forecast=median_forecast,
                quantile_forecasts=pred[sampled_idx],
                encoder_attention_map=attn_weights_past[sampled_idx],
                decoder_attention_map=attn_weights_future[sampled_idx],
                meal_flags=meal_flags,
                loss_value=None
            )
        
    return fixed_indices, figs


def plot_iAUC_scatter(all_pred_iAUC, all_true_iAUC, disable_plots=False):
    """
    Plot a scatter of predicted iAUC vs. true iAUC, plus compute their correlation.
    """

    mean_pred = torch.mean(all_pred_iAUC)
    mean_true = torch.mean(all_true_iAUC)
    cov = torch.mean((all_true_iAUC - mean_true) * (all_pred_iAUC - mean_pred))
    std_true = torch.std(all_true_iAUC, unbiased=False)
    std_pred = torch.std(all_pred_iAUC, unbiased=False)
    corr = cov / (std_true * std_pred + 1e-9)  # small epsilon for safety

    if disable_plots:
        fig = None
        return fig, corr

    fig_scatter, ax_scatter = plt.subplots(figsize=(6, 6))
    ax_scatter.scatter(
        all_true_iAUC.cpu().numpy(),
        all_pred_iAUC.cpu().numpy(),
        alpha=0.5, s=4
    )
    ax_scatter.set_xlabel("True iAUC")
    ax_scatter.set_ylabel("Predicted iAUC")
    ax_scatter.set_title("iAUC Scatter Plot")
    ax_scatter.grid(True)
    ax_scatter.text(
        0.05, 0.95,
        f"Corr: {corr.item():.2f}",
        transform=ax_scatter.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.7)
    )
    return fig_scatter, corr


def plot_meal_self_attention(
    attn_weights_past: torch.Tensor,
    meal_ids_past: torch.Tensor,
    attn_weights_future: torch.Tensor,
    meal_ids_future: torch.Tensor,
    logger,
    global_step: int,
    fixed_indices: list = None,
    max_examples: int = 4,  # Match default in plot_forecast_examples
    random_samples: bool = True
):
    """
    Visualize the meal self-attention across all timesteps.
    Creates optimized individual figures for each example.
    
    Args:
        attn_weights_past: Shape [B, T, M, M] - Attention weights for past meals
        meal_ids_past: Shape [B, T, M] - Meal IDs for past meals
        attn_weights_future: Shape [B, T, M, M] - Attention weights for future meals
        meal_ids_future: Shape [B, T, M] - Meal IDs for future meals
        logger: Logger for W&B
        global_step: Current training step
        fixed_indices: Specific batch indices to plot (default: None)
        max_examples: Maximum number of examples to plot (default: 4)
        random_samples: Whether to randomly sample examples (default: True)
        
    Returns:
        list[int]: The indices actually plotted
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import random
    import wandb
    import torch
    import io
    from PIL import Image
    
    # Determine batch size and select sample indices
    if attn_weights_past is not None:
        batch_size = attn_weights_past.size(0)
    elif attn_weights_future is not None:
        batch_size = attn_weights_future.size(0)
    else:
        # Nothing to plot
        fig = plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "No meal attention data available", 
                ha='center', va='center', fontsize=14)
        plt.axis('off')
        logger.experiment.log({"meal_self_attention_grid": wandb.Image(fig),
                              "global_step": global_step})
        plt.close(fig)
        return fixed_indices

    # Use EXACTLY the same index logic as plot_forecast_examples for consistency
    num_examples = min(max_examples, batch_size)
    if fixed_indices is None:
        indices = random.sample(list(range(batch_size)), num_examples)
    else:
        indices = fixed_indices[:min(len(fixed_indices), num_examples)]
    
    # Create individual images for each example
    valid_indices = []
    
    for batch_idx in indices:
        # Skip invalid indices silently
        if batch_idx >= batch_size:
            continue
            
        # First, identify which timesteps have valid meals and get max tokens
        valid_past_timesteps = []
        valid_future_timesteps = []
        
        # Store actual token counts for each valid timestep
        past_token_counts = {}
        future_token_counts = {}
        
        # Check past meals
        if attn_weights_past is not None and meal_ids_past is not None:
            for t in range(attn_weights_past.shape[1]):
                n_tokens = (meal_ids_past[batch_idx, t] != 0).sum().item()
                if n_tokens > 0:
                    valid_past_timesteps.append(t)
                    past_token_counts[t] = n_tokens
        
        # Check future meals
        if attn_weights_future is not None and meal_ids_future is not None:
            for t in range(attn_weights_future.shape[1]):
                n_tokens = (meal_ids_future[batch_idx, t] != 0).sum().item()
                if n_tokens > 0:
                    valid_future_timesteps.append(t)
                    future_token_counts[t] = n_tokens
        
        # If no valid timesteps, skip this example
        if not valid_past_timesteps and not valid_future_timesteps:
            continue
        
        valid_indices.append(batch_idx)
        
        # Determine the optimal layout for multiple attention matrices
        total_matrices = len(valid_past_timesteps) + len(valid_future_timesteps)
        
        # Calculate appropriate figure dimensions based on number of matrices
        # We want to create a row for past meals and a row for future meals, with each 
        # matrix taking appropriate width based on its token count
        
        # Calculate total width needed (1 unit per token for each matrix)
        total_width = sum(past_token_counts.values()) + sum(future_token_counts.values())
        max_past_height = max(past_token_counts.values()) if past_token_counts else 0
        max_future_height = max(future_token_counts.values()) if future_token_counts else 0
        total_height = max_past_height + max_future_height
        
        # Scale figure size based on total content, with minimum sizes
        base_size = 6
        width_scale = max(1.0, min(2.0, total_width / 20))  # Limit scaling
        height_scale = max(1.0, min(1.5, total_height / 10))  # Limit scaling
        
        fig_width = base_size * width_scale
        fig_height = base_size * height_scale * 0.6  # Slightly smaller height ratio
        
        # Create figure with optimized dimensions
        fig = plt.figure(figsize=(fig_width, fig_height))
        
        # Create subplots with appropriate space for labels and title
        ax = fig.add_subplot(111)
        
        # Now create compact matrices by placing them side by side without empty space
        
        # Initialize positions
        curr_x = 0
        # Track matrix positions
        matrix_positions = []  # Store (x_start, x_end, timestep, is_past)
        
        # Create a combined attention display matrix
        display_matrix = np.zeros((total_height, total_width))
        display_mask = np.zeros((total_height, total_width), dtype=bool)  # Track valid cells
        
        # Add past meal attention matrices
        for t in valid_past_timesteps:
            n_tokens = past_token_counts[t]
            attn = attn_weights_past[batch_idx, t]
            
            # Handle case with start token
            if attn.shape[0] > n_tokens:
                attn = attn[1:n_tokens+1, 1:n_tokens+1]
            else:
                attn = attn[:n_tokens, :n_tokens]
            
            # Place in the display matrix
            display_matrix[:n_tokens, curr_x:curr_x+n_tokens] = attn.cpu().numpy()
            display_mask[:n_tokens, curr_x:curr_x+n_tokens] = True
            
            # Store matrix position
            matrix_positions.append((curr_x, curr_x+n_tokens, t, True))
            
            # Add separator line
            if curr_x > 0:
                ax.axvline(x=curr_x-0.5, color='white', linestyle='--', linewidth=0.5)
                
            # Move to next position
            curr_x += n_tokens
        
        # Add future meal attention matrices
        for t in valid_future_timesteps:
            n_tokens = future_token_counts[t]
            attn = attn_weights_future[batch_idx, t]
            
            # Handle case with start token
            if attn.shape[0] > n_tokens:
                attn = attn[1:n_tokens+1, 1:n_tokens+1]
            else:
                attn = attn[:n_tokens, :n_tokens]
            
            # Place in the display matrix (below past meals)
            row_start = max_past_height
            display_matrix[row_start:row_start+n_tokens, curr_x:curr_x+n_tokens] = attn.cpu().numpy()
            display_mask[row_start:row_start+n_tokens, curr_x:curr_x+n_tokens] = True
            
            # Store matrix position
            matrix_positions.append((curr_x, curr_x+n_tokens, t, False))
            
            # Add separator line
            if curr_x > 0:
                ax.axvline(x=curr_x-0.5, color='white', linestyle='--', linewidth=0.5)
                
            # Move to next position
            curr_x += n_tokens
        
        # Create masked array for proper display
        masked_display = np.ma.array(display_matrix, mask=~display_mask)
        
        # Plot with custom colormap
        cmap = plt.cm.viridis
        im = ax.imshow(masked_display, cmap=cmap, vmin=0, vmax=1, aspect='equal')
        
        # Add horizontal line separating past and future meals
        if max_past_height > 0 and max_future_height > 0:
            ax.axhline(y=max_past_height-0.5, color='red', linestyle='-', linewidth=2)
        
        # Intelligently add timestep labels to avoid overlap
        # First, sort matrices by position
        matrix_positions.sort(key=lambda x: x[0])
        
        # Group timesteps that are close together to avoid label overlap
        MIN_LABEL_WIDTH = 5  # Minimum width in data units to display individual labels
        
        # Process past and future matrices separately
        for is_past in [True, False]:
            # Filter matrices by type
            filtered_positions = [p for p in matrix_positions if p[3] == is_past]
            
            # If no matrices of this type, skip
            if not filtered_positions:
                continue
            
            # Determine label groups (combine adjacent timesteps if too narrow)
            label_groups = []
            current_group = [filtered_positions[0]]
            
            for i in range(1, len(filtered_positions)):
                prev_x_end = current_group[-1][1]
                curr_x_start = filtered_positions[i][0]
                curr_width = filtered_positions[i][1] - filtered_positions[i][0]
                
                # If this matrix is close to the previous one or is very narrow
                if curr_x_start - prev_x_end < MIN_LABEL_WIDTH or curr_width < MIN_LABEL_WIDTH:
                    # Add to current group
                    current_group.append(filtered_positions[i])
                else:
                    # Start a new group
                    label_groups.append(current_group)
                    current_group = [filtered_positions[i]]
            
            # Add the last group
            if current_group:
                label_groups.append(current_group)
            
            # Create label for each group
            for group in label_groups:
                # Calculate group boundaries
                x_start = group[0][0]
                x_end = group[-1][1]
                
                # Create label text based on timesteps in the group
                if len(group) == 1:
                    # Single timestep
                    t = group[0][2]
                    label = f"P{t}" if is_past else f"F{t}"
                else:
                    # Multiple timesteps
                    # For large groups, just show count and range
                    if len(group) > 5:
                        first_t = group[0][2]
                        last_t = group[-1][2]
                        label = f"P{first_t}-{last_t}" if is_past else f"F{first_t}-{last_t}"
                    else:
                        # For small groups, show individual timesteps
                        times = [str(p[2]) for p in group]
                        label = f"P{','.join(times)}" if is_past else f"F{','.join(times)}"
                
                # Position label at center of group
                x_center = (x_start + x_end) / 2
                
                # Determine y position based on past/future
                # Move labels further from the plot to avoid overlap
                y_pos = -0.75 if is_past else max_past_height - 0.5
                
                # Add label with background for visibility
                # Reduce padding and make box more compact
                ax.text(x_center, y_pos, label, ha='center', va='bottom', fontsize=8, 
                       bbox=dict(facecolor='white', alpha=0.9, pad=1, boxstyle='round,pad=0.3'))
        
        # Add section labels with better positioning
        if max_past_height > 0:
            ax.text(-2, max_past_height/2-0.5, "Past\nMeals", ha='right', va='center', 
                    fontsize=10, fontweight='bold')
        if max_future_height > 0:
            ax.text(-2, max_past_height + max_future_height/2-0.5, "Future\nMeals", 
                    ha='right', va='center', fontsize=10, fontweight='bold')
        
        # Clean up ticks to show only valid positions
        ax.set_xticks([])  # Hide x-ticks for cleaner look
        
        # Set proper y-ticks
        y_ticks = []
        y_tick_labels = []
        
        # Past meal tokens
        if max_past_height > 0:
            y_ticks.extend(range(max_past_height))
            y_tick_labels.extend(range(1, max_past_height+1))
        
        # Future meal tokens
        if max_future_height > 0:
            y_ticks.extend(range(max_past_height, max_past_height + max_future_height))
            y_tick_labels.extend(range(1, max_future_height+1))
            
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels)
        
        # Set title with significantly more padding to avoid overlap with labels
        ax.set_title(f"Sample {batch_idx} - Meal Self-Attention Grid", 
                    fontsize=12, pad=25)
        
        # Add colorbar with better sizing
        cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
        cbar.set_label('Attention Strength', fontsize=8)
        cbar.outline.set_visible(False)  # Remove the boundary/outline
        
        # Adjust layout with extra padding at the top for title and labels
        plt.subplots_adjust(top=0.80, bottom=0.20, left=0.1, right=0.95)
        
        # Log to W&B
        logger.experiment.log({
            f"meal_attention_sample_{batch_idx}": wandb.Image(fig),
            "global_step": global_step
        })
        plt.close(fig)
    
    # If no valid examples were found
    if not valid_indices:
        fig = plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "No valid meal attention data in selected examples", 
                ha='center', va='center', fontsize=14)
        plt.axis('off')
        logger.experiment.log({"meal_self_attention_grid": wandb.Image(fig),
                              "global_step": global_step})
        plt.close(fig)
    
    return fixed_indices

def generate_mock_attention(batch_size: int, 
                           context_len: int, 
                           past_len: int, 
                           forecast_len: int,
                           meal_embedding_dim: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate mock attention weights for visualization.
    
    Args:
        batch_size: Number of samples
        context_len: Length of context window (not used in this implementation)
        past_len: Number of past timesteps
        forecast_len: Number of forecast timesteps
        meal_embedding_dim: Dimension of meal embeddings (not used for attention maps)
        
    Returns:
        Tuple of:
            encoder_attention: Encoder attention weights [batch, forecast_len, past_len]
            decoder_attention: Decoder attention weights [batch, forecast_len, forecast_len]
    """
    # Create encoder attention [batch_size, forecast_len, past_len]
    # Simply use random values for clear visibility
    encoder_attn = torch.rand(batch_size, forecast_len, past_len)
    
    # Normalize rows to sum to 1
    row_sums = encoder_attn.sum(dim=2, keepdim=True)
    encoder_attn = encoder_attn / row_sums
    
    # Create decoder attention [batch_size, forecast_len, forecast_len]
    # Use random values but maintain causal mask (empty bottom half)
    decoder_attn = torch.rand(batch_size, forecast_len, forecast_len)
    
    # Apply causal mask - zero out the upper triangular part (j > i)
    for b in range(batch_size):
        for i in range(forecast_len):
            for j in range(forecast_len):
                if j > i:  # Zero out future positions
                    decoder_attn[b, i, j] = 0.0
    
    # Normalize each row to sum to 1
    for b in range(batch_size):
        for i in range(forecast_len):
            row_sum = decoder_attn[b, i].sum()
            if row_sum > 0:
                decoder_attn[b, i] = decoder_attn[b, i] / row_sum
    
    return encoder_attn, decoder_attn

def test_plot_forecast_examples():
    print("\n===== Testing plot_forecast_examples =====")
    
    # Create mock data
    batch_size = 8
    past_len = 32  # 8 hours with 4 readings per hour
    forecast_len = 16  # 4 hours with 4 readings per hour
    quantiles = torch.tensor([0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95])
    num_quantiles = len(quantiles)
    past_meal_len = 8  # Number of past meal timepoints
    future_meal_len = 4  # Number of future meal timepoints
    meal_embedding_dim = 5  # Dimension of meal embeddings
    
    # Generate mock glucose and meal data
    forecasts = generate_mock_glucose_data(
        batch_size=batch_size, 
        past_len=past_len,
        forecast_len=forecast_len,
        num_quantiles=num_quantiles,
        past_meal_len=past_meal_len,
        future_meal_len=future_meal_len
    )
    
    # Generate mock attention weights with proper dimensions for plot_forecast
    # The attention maps should have shapes:
    # - encoder_attn: [batch_size, forecast_len, past_len]
    # - decoder_attn: [batch_size, forecast_len, forecast_len]
    encoder_attn, decoder_attn = generate_mock_attention(
        batch_size=batch_size,
        context_len=past_len,  # Not used in the updated function
        past_len=past_len,
        forecast_len=forecast_len
    )
    
    # Create a mock logger
    logger = MockLogger()
    global_step = 1000
    
    print(f"Generated mock data:")
    print(f"- past glucose shape: {forecasts['past'].shape}")
    print(f"- predicted glucose shape: {forecasts['pred'].shape}")
    print(f"- true future glucose shape: {forecasts['truth'].shape}")
    print(f"- past meal IDs shape: {forecasts['past_meal_ids'].shape}")
    print(f"- future meal IDs shape: {forecasts['future_meal_ids'].shape}")
    print(f"- encoder attention weights shape: {encoder_attn.shape}")
    print(f"- decoder attention weights shape: {decoder_attn.shape}")
    
    # Call the plotting function with updated axis labels
    fixed_indices, figs = plot_forecast_examples(
        forecasts=forecasts,
        attn_weights_past=encoder_attn,
        attn_weights_future=decoder_attn,
        quantiles=quantiles,
        logger=logger,
        global_step=global_step,
        fixed_indices=None  # Let the function randomly select examples
    )
            
    # Save figure instead of showing it
    for idx, fig in enumerate(figs):
        output_path = f"forecast_examples_test_{idx}.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Plot saved to {output_path}")
    
    return fixed_indices, figs

if __name__ == "__main__":
    import random
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import argparse
    from typing import Optional, Union, Dict, List, Tuple
    
    # Mock logger for Weights & Biases logging
    class MockLogger:
        class MockExperiment:
            def log(self, data):
                print(f"[MOCK W&B LOG] Logged data with keys: {list(data.keys())}")
        
        def __init__(self):
            self.experiment = self.MockExperiment()
    
    # Create command-line arguments to select which function to test
    parser = argparse.ArgumentParser(description="Test plot helper functions with mock data")
    parser.add_argument("--function", type=str, choices=["forecast", "iauc", "attention", "all"], 
                        default="all", help="Which plotting function to test")
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # ===== Mock data generation utilities =====
    
    def generate_mock_glucose_data(batch_size: int = 8, 
                                  past_len: int = 32, 
                                  forecast_len: int = 16,
                                  num_quantiles: int = 9,
                                  past_meal_len: int = 8,
                                  future_meal_len: int = 4,
                                  meal_embedding_dim: int = 5) -> Dict[str, torch.Tensor]:
        """
        Generate mock data for glucose forecasting visualization.
        
        Args:
            batch_size: Number of samples in batch
            past_len: Length of historical glucose values
            forecast_len: Length of forecast horizon
            num_quantiles: Number of quantiles in prediction
            past_meal_len: Number of past meal timepoints
            future_meal_len: Number of future meal timepoints
            meal_embedding_dim: Dimension of meal embeddings
            
        Returns:
            Dictionary with mock forecast data
        """
        # Generate realistic glucose values in mmol/L (normal range ~4-10)
        base_glucose = 5.5 + 1.0 * torch.randn(batch_size, 1)  # Different baseline for each patient
        
        # Historical glucose with some trends and noise
        past_time = torch.linspace(0, 1, past_len).unsqueeze(0).repeat(batch_size, 1)
        past_glucose = base_glucose + 1.5 * torch.sin(6 * past_time) + 0.5 * torch.randn(batch_size, past_len)
        
        # Future true glucose continues the pattern (with slight increase for meal effect)
        future_time = torch.linspace(1, 1.5, forecast_len).unsqueeze(0).repeat(batch_size, 1)
        future_glucose = base_glucose + 1.5 * torch.sin(6 * future_time) + 0.7 * torch.randn(batch_size, forecast_len)
        
        # Add meal effects to some samples (create realistic postprandial glucose rise)
        for b in range(batch_size):
            if random.random() < 0.7:  # 70% chance of meal effect
                # Add rising glucose after a meal - peak at ~2 mmol/L higher
                peak_time = random.randint(4, forecast_len-2)  # Random peak position
                peak_height = 1.5 + 0.8 * random.random()  # Peak height 1.5-2.3 mmol/L
                
                # Create a bell curve for the meal response
                for t in range(forecast_len):
                    dist_from_peak = abs(t - peak_time)
                    meal_effect = peak_height * np.exp(-0.3 * dist_from_peak)
                    future_glucose[b, t] += meal_effect
        
        # Predicted glucose - make it less accurate and vary by example
        # Shape: [batch_size, forecast_len, num_quantiles]
        median_idx = num_quantiles // 2
        
        # Make predictions somewhat shifted and with increasing error over time
        # This creates more separation between the forecast and ground truth
        future_pred_median = torch.zeros_like(future_glucose)
        for b in range(batch_size):
            forecast_error = 0.3 + 0.6 * random.random()  # Varying forecast error
            time_factor = torch.linspace(0.5, 1.5, forecast_len)  # Error grows with time
            
            # Add systematic error (trend direction bias) for realism
            trend_bias = 0.4 * (2 * random.random() - 1)  # Between -0.4 and 0.4
            
            # Combine various error components into a realistic forecast
            error = (forecast_error * torch.randn(forecast_len) + 
                    trend_bias * time_factor + 
                    0.15 * torch.sin(3 * torch.tensor(range(forecast_len))))
            
            future_pred_median[b] = future_glucose[b] + error
        
        # Create quantiles with appropriate spread
        # The spread should increase with forecast horizon
        quantile_offsets = torch.zeros(forecast_len, num_quantiles)
        for t in range(forecast_len):
            # Wider spreads for farther horizons
            width_factor = 0.5 + 0.8 * (t / forecast_len)
            quantile_offsets[t] = torch.linspace(-width_factor*2.0, width_factor*2.0, num_quantiles)
        
        # Expand dimensions for broadcasting
        future_pred = future_pred_median.unsqueeze(-1) + quantile_offsets.unsqueeze(0)
        
        # Create meal data - sparse binary indicators (1 = meal present, 0 = no meal)
        past_meals = torch.zeros(batch_size, past_meal_len, meal_embedding_dim)
        future_meals = torch.zeros(batch_size, future_meal_len, meal_embedding_dim)
        
        # Add some random meals (10-20% of timesteps will have meals)
        for b in range(batch_size):
            # Add 1-3 past meals
            for _ in range(random.randint(1, 3)):
                t = random.randint(0, past_meal_len - 1)
                meal_type = random.randint(1, meal_embedding_dim-1)  # 0 is reserved for no meal
                past_meals[b, t, 0] = 1  # Meal presence indicator
                past_meals[b, t, meal_type] = 1  # Meal type
            
            # Add 0-2 future meals
            for _ in range(random.randint(0, 2)):
                t = random.randint(0, future_meal_len - 1)
                meal_type = random.randint(1, meal_embedding_dim-1)
                future_meals[b, t, 0] = 1  # Meal presence indicator
                future_meals[b, t, meal_type] = 1  # Meal type
        
        return {
            "past": past_glucose,
            "pred": future_pred,
            "truth": future_glucose,
            "past_meal_ids": past_meals,
            "future_meal_ids": future_meals
        }
    
    def generate_mock_iauc_data(num_samples: int = 200) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate mock iAUC (incremental Area Under Curve) data.
        
        Args:
            num_samples: Number of data points
            
        Returns:
            Tuple of predicted and true iAUC values
        """
        # True iAUC values in the range of 0-30
        true_iauc = 15 + 7 * torch.randn(num_samples)
        
        # Add constraints to keep values in a realistic range
        true_iauc = torch.clamp(true_iauc, min=0, max=30)
        
        # Predicted iAUC - correlated with true but with noise
        # Create correlation of about 0.7
        correlation = 0.7
        noise = torch.randn(num_samples)
        pred_iauc = correlation * true_iauc + (1 - correlation) * 5 * noise
        
        # Ensure predictions are in the same range
        pred_iauc = torch.clamp(pred_iauc, min=0, max=30)
        
        return pred_iauc, true_iauc
    
    # ===== Test plot_forecast_examples =====
    
    def test_plot_forecast_examples():
        print("\n===== Testing plot_forecast_examples =====")
        
        # Create mock data
        batch_size = 8
        past_len = 32  # 8 hours with 4 readings per hour
        forecast_len = 16  # 4 hours with 4 readings per hour
        quantiles = torch.tensor([0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95])
        num_quantiles = len(quantiles)
        past_meal_len = 8  # Number of past meal timepoints
        future_meal_len = 4  # Number of future meal timepoints
        meal_embedding_dim = 5  # Dimension of meal embeddings
        
        # Generate mock glucose and meal data
        forecasts = generate_mock_glucose_data(
            batch_size=batch_size, 
            past_len=past_len,
            forecast_len=forecast_len,
            num_quantiles=num_quantiles,
            past_meal_len=past_meal_len,
            future_meal_len=future_meal_len
        )
        
        # Generate mock attention weights with proper dimensions for plot_forecast
        # The attention maps should have shapes:
        # - encoder_attn: [batch_size, forecast_len, past_len]
        # - decoder_attn: [batch_size, forecast_len, forecast_len]
        encoder_attn, decoder_attn = generate_mock_attention(
            batch_size=batch_size,
            context_len=past_len,  # Not used in the updated function
            past_len=past_len,
            forecast_len=forecast_len
        )
        
        # Create a mock logger
        logger = MockLogger()
        global_step = 1000
        
        print(f"Generated mock data:")
        print(f"- past glucose shape: {forecasts['past'].shape}")
        print(f"- predicted glucose shape: {forecasts['pred'].shape}")
        print(f"- true future glucose shape: {forecasts['truth'].shape}")
        print(f"- past meal IDs shape: {forecasts['past_meal_ids'].shape}")
        print(f"- future meal IDs shape: {forecasts['future_meal_ids'].shape}")
        print(f"- encoder attention weights shape: {encoder_attn.shape}")
        print(f"- decoder attention weights shape: {decoder_attn.shape}")
        
        # Call the plotting function with updated axis labels
        fixed_indices, figs = plot_forecast_examples(
            forecasts=forecasts,
            attn_weights_past=encoder_attn,
            attn_weights_future=decoder_attn,
            quantiles=quantiles,
            logger=logger,
            global_step=global_step,
            fixed_indices=None  # Let the function randomly select examples
        )
                
        # Save figure instead of showing it
        for idx, fig in enumerate(figs):
            output_path = f"forecast_examples_test_{idx}.png"
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Plot saved to {output_path}")
        
        return fixed_indices, figs
    
    # ===== Test plot_iAUC_scatter =====
    
    def test_plot_iauc_scatter():
        print("\n===== Testing plot_iAUC_scatter =====")
        
        # Generate mock iAUC data
        pred_iauc, true_iauc = generate_mock_iauc_data(num_samples=200)
        
        print(f"Generated mock iAUC data:")
        print(f"- predicted iAUC shape: {pred_iauc.shape}")
        print(f"- true iAUC shape: {true_iauc.shape}")
        
        # Call the function
        fig, corr = plot_iAUC_scatter(
            all_pred_iAUC=pred_iauc,
            all_true_iAUC=true_iauc,
            disable_plots=False
        )
        
        # Save figure instead of showing it
        output_path = "iauc_scatter_test.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"iAUC scatter plot created with correlation: {corr.item():.4f}")
        print(f"Plot saved to {output_path}")
        
        return fig, corr
    
    # ===== Test plot_meal_self_attention =====
    
    def test_plot_meal_self_attention():
        print("\n===== Testing plot_meal_self_attention =====")
        
        # Create mock data
        batch_size = 8
        past_len = 8  # Number of past meal timepoints
        future_len = 4  # Number of future meal timepoints
        meal_embedding_dim = 5  # Dimension of meal embeddings
        
        # Generate mock meal data
        mock_data = generate_mock_glucose_data(
            batch_size=batch_size,
            past_len=32,  # Not directly used in this test
            forecast_len=16,  # Not directly used in this test
            past_meal_len=past_len,
            future_meal_len=future_len,
            meal_embedding_dim=meal_embedding_dim
        )
        
        # Extract meal IDs
        meal_ids_past = mock_data["past_meal_ids"]
        meal_ids_future = mock_data["future_meal_ids"]
        
        # Generate mock self-attention matrices
        # Shape: [batch_size, T, M, M]
        # Where T is the number of timesteps and M is the meal embedding dimension
        # We need attention across meal types for each timestep
        
        # For past meals: [batch_size, past_len, meal_embedding_dim, meal_embedding_dim]
        attn_weights_past = torch.zeros(batch_size, past_len, meal_embedding_dim, meal_embedding_dim)
        
        # For future meals: [batch_size, future_len, meal_embedding_dim, meal_embedding_dim]
        attn_weights_future = torch.zeros(batch_size, future_len, meal_embedding_dim, meal_embedding_dim)
        
        # Fill with some plausible attention patterns for each timestep with a meal
        for b in range(batch_size):
            for t in range(past_len):
                if meal_ids_past[b, t, 0] > 0:  # If there's a meal at this timestep
                    # Create a random attention matrix
                    attn_matrix = torch.rand(meal_embedding_dim, meal_embedding_dim)
                    # Make diagonal stronger (fixed version)
                    for i in range(meal_embedding_dim):
                        attn_matrix[i, i] += 0.5
                    # Normalize
                    attn_matrix = attn_matrix / attn_matrix.sum(dim=1, keepdim=True)
                    attn_weights_past[b, t] = attn_matrix
            
            for t in range(future_len):
                if meal_ids_future[b, t, 0] > 0:  # If there's a meal at this timestep
                    # Create a random attention matrix
                    attn_matrix = torch.rand(meal_embedding_dim, meal_embedding_dim)
                    # Make diagonal stronger (fixed version)
                    for i in range(meal_embedding_dim):
                        attn_matrix[i, i] += 0.5
                    # Normalize
                    attn_matrix = attn_matrix / attn_matrix.sum(dim=1, keepdim=True)
                    attn_weights_future[b, t] = attn_matrix
        
        # Create a mock logger
        logger = MockLogger()
        global_step = 1000
        
        print(f"Generated mock data:")
        print(f"- past meal IDs shape: {meal_ids_past.shape}")
        print(f"- future meal IDs shape: {meal_ids_future.shape}")
        print(f"- past attention weights shape: {attn_weights_past.shape}")
        print(f"- future attention weights shape: {attn_weights_future.shape}")
        
        # Call the function
        fixed_indices = plot_meal_self_attention(
            attn_weights_past=attn_weights_past,
            meal_ids_past=meal_ids_past,
            attn_weights_future=attn_weights_future,
            meal_ids_future=meal_ids_future,
            logger=logger,
            global_step=global_step,
            fixed_indices=None,  # Let the function randomly select examples
            max_examples=4,
            random_samples=True
        )
        
        print(f"Meal self-attention plots created and logged with indices: {fixed_indices}")
        
        return fixed_indices
    
    # Run the selected test(s)
    if args.function == "all" or args.function == "forecast":
        fixed_indices, forecast_fig = test_plot_forecast_examples()
    
    if args.function == "all" or args.function == "iauc":
        iauc_fig, corr = test_plot_iauc_scatter()
    
    if args.function == "all" or args.function == "attention":
        attention_indices = test_plot_meal_self_attention()
    
    # Remove the plt.show() call since we're saving files instead
    # if args.function != "attention":  # Attention plots are saved to W&B, not displayed
    #     plt.show()
    