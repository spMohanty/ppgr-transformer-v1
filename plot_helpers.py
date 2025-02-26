import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

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

    # Create subplots: 
    # - For each example, we have 1 row and up to 3 columns:
    #     [0] time-series plot
    #     [1] past attention heatmap (if available)
    #     [2] future attention heatmap (if available)
    #
    # But if attn_weights_past or attn_weights_future is None, we skip that column
    # and reduce the subplot columns as needed.
    n_cols = 1
    if attn_weights_past is not None:
        n_cols += 1
    if attn_weights_future is not None:
        n_cols += 1
    
    fig, axs = plt.subplots(num_examples, n_cols, figsize=(6 * n_cols, 4 * num_examples))
    
    # If there's only 1 example, axs is 1D if n_cols>1, or just a single Axes if n_cols=1
    if num_examples == 1:
        axs = np.array([axs])  # make it 2D for consistency
    if n_cols == 1:
        # One column only => rework into 2D array
        axs = axs[:, None]

    for i, idx in enumerate(sampled_indices):
        col_idx = 0
        # Column 0: Time-series + forecast
        ax_ts = axs[i][col_idx]
        col_idx += 1

        past_i = past[idx].cpu().numpy()
        pred_i = pred[idx].cpu().numpy()   # shape (T_forecast, Q)
        truth_i = truth[idx].cpu().numpy() # shape (T_forecast, )
        
        T_context = past_i.shape[0]
        T_forecast = pred_i.shape[0]
        x_hist = list(range(-T_context + 1, 1))
        x_forecast = list(range(1, T_forecast + 1))

        # Historical glucose
        ax_ts.plot(x_hist, past_i, marker="o", markersize=2, label="Historical Glucose")

        # Ground-truth future
        ax_ts.plot(x_forecast, truth_i, marker="o", markersize=2, label="Ground Truth")

        # Plot quantile forecasts (shaded)
        median_index = pred_i.shape[1] // 2
        base_color = "blue"
        # Fill intervals between consecutive quantiles
        # e.g. pred_i[:,0]..pred_i[:,1], pred_i[:,1]..pred_i[:,2], etc.
        for qi in range(pred_i.shape[1] - 1):
            alpha_val = 0.1 + (abs(qi - median_index)) * 0.05
            q_lower = np.minimum(pred_i[:, qi], pred_i[:, qi+1])
            q_upper = np.maximum(pred_i[:, qi], pred_i[:, qi+1])
            ax_ts.fill_between(
                x_forecast, q_lower, q_upper,
                color=base_color, alpha=alpha_val
            )
        # The "median" line
        ax_ts.plot(
            x_forecast, pred_i[:, median_index],
            marker="o", markersize=2, color="darkblue", label="Median Forecast"
        )

        # Overplot vertical lines where "non‐padding" meals are present
        meal_label_added = False
        # Past meal lines
        meals_past = meal_ids_past[idx].cpu().numpy()
        T_past_meals = meals_past.shape[0]
        for t, meal in enumerate(meals_past):
            # If any non‐zero ID => there's a meal
            if (meal != 0).any():
                relative_time = t - T_past_meals + 1
                if not meal_label_added:
                    ax_ts.axvline(x=relative_time, color="purple", linestyle="--", alpha=0.7, label="Meal Consumption")
                    meal_label_added = True
                else:
                    ax_ts.axvline(x=relative_time, color="purple", linestyle="--", alpha=0.7)
        # Future meal lines
        meals_future = meal_ids_future[idx].cpu().numpy()
        for t, meal in enumerate(meals_future):
            if (meal != 0).any():
                relative_time = t + 1
                ax_ts.axvline(x=relative_time, color="purple", linestyle="--", alpha=0.7)

        ax_ts.set_xlabel("Relative Timestep")
        ax_ts.set_ylabel("Glucose Level")
        ax_ts.set_title(f"Forecast Example {i} (Idx: {idx})")
        ax_ts.legend(fontsize="small")
        
        # Column 1 (if attn_weights_past is not None): Past attention
        if attn_weights_past is not None:
            ax_attn_past = axs[i][col_idx]
            col_idx += 1
            attn_past_i = attn_weights_past[idx].cpu().numpy()
            im_past = ax_attn_past.imshow(attn_past_i, aspect="auto", cmap="viridis")
            ax_attn_past.set_title("Past Meals Attention")
            ax_attn_past.set_xlabel("Past Meal Timestep")
            ax_attn_past.set_ylabel("Glucose Timestep")
            cbar_past = fig.colorbar(im_past, ax=ax_attn_past, fraction=0.046, pad=0.04)
            cbar_past.set_label("Attention Weight", fontsize=8)

        # Column 2 (if attn_weights_future is not None): Future attention
        if attn_weights_future is not None:
            ax_attn_future = axs[i][col_idx]
            col_idx += 1
            attn_future_i = attn_weights_future[idx].cpu().numpy()
            im_future = ax_attn_future.imshow(attn_future_i, aspect="auto", cmap="viridis")
            ax_attn_future.set_title("Future Meals Attention")
            ax_attn_future.set_xlabel("Future Meal Timestep")
            ax_attn_future.set_ylabel("Glucose Timestep")
            cbar_future = fig.colorbar(im_future, ax=ax_attn_future, fraction=0.046, pad=0.04)
            cbar_future.set_label("Attention Weight", fontsize=8)

    fig.tight_layout()

    # Log to W&B
    logger.experiment.log({"forecast_samples": wandb.Image(fig), "global_step": global_step})
    return fixed_indices, fig


def plot_iAUC_scatter(all_pred_iAUC, all_true_iAUC):
    """
    Plot a scatter of predicted iAUC vs. true iAUC, plus compute their correlation.
    """
    mean_pred = torch.mean(all_pred_iAUC)
    mean_true = torch.mean(all_true_iAUC)
    cov = torch.mean((all_true_iAUC - mean_true) * (all_pred_iAUC - mean_pred))
    std_true = torch.std(all_true_iAUC, unbiased=False)
    std_pred = torch.std(all_pred_iAUC, unbiased=False)
    corr = cov / (std_true * std_pred + 1e-9)  # small epsilon for safety

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
    max_examples: int = 1,  # Default to 1 since grid is already large
    random_samples: bool = True
):
    """
    Visualize the meal self-attention across all timesteps in a single large grid.
    
    The grid is organized as:
    - 22 rows (11 for past foods, 11 for future foods)
    - 11 * (T_past + T_future) columns, with each timestep getting an 11×11 block
    - Past meals shown in top half, future meals in bottom half
    
    Args:
        attn_weights_past: Shape [B, T, M, M] - Attention weights for past meals
        meal_ids_past: Shape [B, T, M] - Meal IDs for past meals
        attn_weights_future: Shape [B, T, M, M] - Attention weights for future meals
        meal_ids_future: Shape [B, T, M] - Meal IDs for future meals
        logger: Logger for W&B
        global_step: Current training step
        fixed_indices: Specific batch indices to plot (default: None)
        max_examples: Maximum number of examples to plot (default: 1)
        random_samples: Whether to randomly sample examples (default: True)
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import random
    import wandb
    import torch
    
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
        return fig

    if fixed_indices is not None:
        indices = fixed_indices[:min(max_examples, len(fixed_indices))]
    elif random_samples:
        indices = random.sample(range(batch_size), k=min(max_examples, batch_size))
    else:
        indices = list(range(min(max_examples, batch_size)))
    
    # Process each sample
    for batch_idx in indices:
        # Figure out the total number of timesteps we have
        T_past = attn_weights_past.shape[1] if attn_weights_past is not None else 0
        T_future = attn_weights_future.shape[1] if attn_weights_future is not None else 0
        
        # Create a large empty grid to hold all timesteps
        # Shape: [22, 11 * (T_past + T_future)]
        # Top 11 rows for past meals, bottom 11 rows for future meals
        MAX_TOKENS = 11  # Maximum number of food tokens per meal
        grid_height = 2 * MAX_TOKENS
        grid_width = MAX_TOKENS * (T_past + T_future)
        
        # Initialize with NaN to distinguish from zero attention
        attention_grid = np.full((grid_height, grid_width), np.nan)
        
        # Create a mask to track valid meal positions (for better visualization)
        valid_mask = np.zeros((grid_height, grid_width), dtype=bool)
        
        # Helper function to insert an attention matrix into the grid
        def insert_attention_matrix(grid, mask, attn_matrix, row_start, col_start, max_size=MAX_TOKENS):
            actual_size = min(attn_matrix.shape[0], max_size)
            
            # Insert the attention matrix (or a subset if it's larger than max_size)
            grid[row_start:row_start+actual_size, col_start:col_start+actual_size] = \
                attn_matrix[:actual_size, :actual_size]
            
            # Update the mask
            mask[row_start:row_start+actual_size, col_start:col_start+actual_size] = True
        
        # Fill in past meals attention
        if attn_weights_past is not None:
            for t in range(T_past):
                # Skip if this timestep has no valid meals
                if meal_ids_past is None or not (meal_ids_past[batch_idx, t] != 0).any():
                    continue
                
                # Get the number of valid tokens
                n_tokens = (meal_ids_past[batch_idx, t] != 0).sum().item()
                if n_tokens == 0:
                    continue
                
                # Extract attention matrix for this timestep
                attn = attn_weights_past[batch_idx, t]
                
                # Handle case with start token (if shape is larger than valid tokens)
                if attn.shape[0] > n_tokens:
                    attn = attn[1:n_tokens+1, 1:n_tokens+1]  # Remove start token
                else:
                    attn = attn[:n_tokens, :n_tokens]
                
                # Place in the top half of the grid
                col_start = t * MAX_TOKENS
                insert_attention_matrix(attention_grid, valid_mask, attn.cpu().numpy(), 0, col_start)
        
        # Fill in future meals attention
        if attn_weights_future is not None:
            for t in range(T_future):
                # Skip if this timestep has no valid meals
                if meal_ids_future is None or not (meal_ids_future[batch_idx, t] != 0).any():
                    continue
                
                # Get the number of valid tokens
                n_tokens = (meal_ids_future[batch_idx, t] != 0).sum().item()
                if n_tokens == 0:
                    continue
                
                # Extract attention matrix for this timestep
                attn = attn_weights_future[batch_idx, t]
                
                # Handle case with start token (if shape is larger than valid tokens)
                if attn.shape[0] > n_tokens:
                    attn = attn[1:n_tokens+1, 1:n_tokens+1]  # Remove start token
                else:
                    attn = attn[:n_tokens, :n_tokens]
                
                # Place in the bottom half of the grid
                col_start = (T_past + t) * MAX_TOKENS
                insert_attention_matrix(attention_grid, valid_mask, attn.cpu().numpy(), MAX_TOKENS, col_start)
        
        # Create the figure - set size based on grid dimensions
        # Scaling factors for better visibility
        w_scale = max(0.2, min(1.0, 20 / grid_width))
        h_scale = max(0.4, min(1.0, 10 / grid_height))
        
        fig = plt.figure(figsize=(grid_width * w_scale, grid_height * h_scale))
        fig.suptitle(f"Sample {batch_idx} - Meal Self-Attention Grid", fontsize=14)
        
        # Create a custom colormap that masks NaN values
        cmap = plt.cm.viridis.copy()
        cmap.set_bad('lightgray')  # NaN values will be light gray
        
        # Plot the grid
        ax = plt.gca()
        im = ax.imshow(attention_grid, cmap=cmap, vmin=0, vmax=1)
        
        # Add a horizontal line separating past and future meals
        ax.axhline(y=MAX_TOKENS-0.5, color='red', linestyle='-', linewidth=2)
        
        # Add timestep labels
        # Top x-axis for past meal timesteps
        past_timesteps = []
        future_timesteps = []
        
        if attn_weights_past is not None:
            for t in range(T_past):
                if meal_ids_past is not None and (meal_ids_past[batch_idx, t] != 0).any():
                    past_timesteps.append(t)
        
        if attn_weights_future is not None:
            for t in range(T_future):
                if meal_ids_future is not None and (meal_ids_future[batch_idx, t] != 0).any():
                    future_timesteps.append(t)
        
        # Add vertical lines separating timesteps
        for t in range(1, T_past + T_future):
            ax.axvline(x=t*MAX_TOKENS-0.5, color='white', linestyle='--', linewidth=0.5, alpha=0.7)
        
        # Add timestep labels at the top of each 11x11 block
        for t in past_timesteps:
            col_center = t * MAX_TOKENS + MAX_TOKENS // 2
            ax.text(col_center, -0.5, f"P{t}", ha='center', va='bottom', fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.7))
        
        for t in future_timesteps:
            col_center = (T_past + t) * MAX_TOKENS + MAX_TOKENS // 2
            ax.text(col_center, -0.5, f"F{t}", ha='center', va='bottom', fontsize=10,
                   bbox=dict(facecolor='white', alpha=0.7))
        
        # Add labels for sections
        ax.text(-1, MAX_TOKENS//2, "Past\nMeals", ha='right', va='center', fontsize=12, fontweight='bold')
        ax.text(-1, MAX_TOKENS + MAX_TOKENS//2, "Future\nMeals", ha='right', va='center', fontsize=12, fontweight='bold')
        
        # Clean up ticks - only show token indices
        token_positions = np.arange(MAX_TOKENS)
        
        # Y-ticks for both past and future sections
        ax.set_yticks(np.concatenate([token_positions, token_positions + MAX_TOKENS]))
        ax.set_yticklabels(np.concatenate([token_positions + 1, token_positions + 1]))
        
        # X-ticks would be too crowded - skip them
        ax.set_xticks([])
        
        # Add a colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
        cbar.set_label('Attention Strength')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the title
        
        # Log to W&B
        logger.experiment.log({
            f"meal_self_attention_grid_{batch_idx}": wandb.Image(fig),
            "global_step": global_step
        })
        plt.close(fig)
    
    # Create a summary figure
    summary_fig = plt.figure(figsize=(8, 6))
    ax = summary_fig.add_subplot(111)
    ax.text(0.5, 0.5, f"Generated attention grids for {len(indices)} samples\nSee individual plots for details", 
           ha='center', va='center', fontsize=14)
    ax.axis('off')
    
    logger.experiment.log({
        "meal_self_attention_grid_summary": wandb.Image(summary_fig),
        "global_step": global_step
    })
    plt.close(summary_fig)
    
    return summary_fig
