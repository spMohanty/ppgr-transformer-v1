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
    