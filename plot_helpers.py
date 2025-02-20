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
            If None, the “past” attention heatmap is skipped.
        attn_weights_future (torch.Tensor or None):
            shape (B, T_context, T_meals) or similar
            If None, the “future” attention heatmap is skipped.
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
        alpha=0.5, s=5
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
    max_examples: int = 3,
    random_samples: bool = True
):
    """
    Visualize the meal self-attention for 'past' meals vs 'future' meals.
    We stack sub-blocks (one per meal-timestep) into big 2D arrays so that
    we see how each meal token attends to others.

    Args:
        attn_weights_past (torch.Tensor): shape [B, T, M, M]
            The self-attention for past meals. If None or empty, we skip.
        meal_ids_past (torch.Tensor): shape [B, T, M]
        attn_weights_future (torch.Tensor): shape [B, T, M, M]
            The self-attention for future meals. If None or empty, we skip.
        meal_ids_future (torch.Tensor): shape [B, T, M]
        logger: A logger for WandB
        global_step (int): for logging
        fixed_indices (list[int], optional): which samples in the batch to plot
        max_examples (int): how many samples to show
        random_samples (bool): if True, sample from the batch if no fixed_indices.

    Returns:
        fig (matplotlib.figure.Figure): the created figure
    """
    # Decide which batch indices we'll plot
    if attn_weights_past is not None:
        batch_size = attn_weights_past.size(0)
    elif attn_weights_future is not None:
        batch_size = attn_weights_future.size(0)
    else:
        # Nothing to plot
        fig = plt.figure()
        plt.title("No Past or Future Self-Attention Provided.")
        logger.experiment.log({"meal_self_attention_samples": wandb.Image(fig),
                               "global_step": global_step})
        plt.close(fig)
        return fig

    if fixed_indices is not None:
        indices = fixed_indices[:min(max_examples, len(fixed_indices))]
    elif random_samples:
        indices = random.sample(range(batch_size), k=min(max_examples, batch_size))
    else:
        indices = list(range(min(max_examples, batch_size)))

    # We'll have 2 columns: [Past Self-Attn, Future Self-Attn]
    # If either one is None, we skip that column.
    n_cols = 0
    if attn_weights_past is not None:
        n_cols += 1
    if attn_weights_future is not None:
        n_cols += 1

    if n_cols == 0:
        fig = plt.figure()
        plt.title("No Past or Future Self-Attention Provided.")
        logger.experiment.log({"meal_self_attention_samples": wandb.Image(fig),
                               "global_step": global_step})
        plt.close(fig)
        return fig

    fig, axes = plt.subplots(
        nrows=len(indices),
        ncols=n_cols,
        figsize=(7*n_cols, 4.5 * len(indices)),
        constrained_layout=True
    )
    if len(indices) == 1 and n_cols == 1:
        axes = [[axes]]
    elif len(indices) == 1:
        axes = [axes]
    elif n_cols == 1:
        # Then axes has shape (rows, ), turn it into (rows,1)
        axes = axes[:, None]

    # Reusable function to pack sub-blocks from a single sample
    def pack_valid_attn_subblocks(attn_4d, meal_ids_3d):
        """
        Convert [B, T, M, M] attention into a big 2D array by stacking only
        the valid sub-blocks (where meal_ids != 0).
        We'll remove the first row/col if it's a "start token".
        Returns: (big_array, boundary_rows), or (None, None) if no valid blocks.
        """
        # If None, return None
        if attn_4d is None:
            return None, None

        attn_4d = attn_4d[:, :, 1:, 1:]   # remove start token dimension
        meal_ids_3d = meal_ids_3d[:, :, 1:]
        # We only handle one sample => drop B dimension
        attn_4d = attn_4d[0]    # shape (T, M-1, M-1)
        meal_ids_3d = meal_ids_3d[0]  # shape (T, M-1)

        subblocks = []
        sizes = []
        T_meal = attn_4d.shape[0]
        for t in range(T_meal):
            meal_ids_slice = meal_ids_3d[t]
            valid_count = (meal_ids_slice != 0).sum().item()
            if valid_count > 0:
                block = attn_4d[t, :valid_count, :valid_count]
                subblocks.append(block.cpu().numpy())
                sizes.append(valid_count)
        if not subblocks:
            return None, None

        total_height = sum(sizes)
        max_width = max(sizes)
        big_array = np.full((total_height, max_width), np.nan, dtype=np.float32)
        boundaries = []
        row_offset = 0
        for sb, sz in zip(subblocks, sizes):
            big_array[row_offset:row_offset+sz, 0:sz] = sb
            row_offset += sz
            boundaries.append(row_offset)
        return big_array, boundaries

    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=0, vmax=1)

    for row_i, idx in enumerate(indices):
        col_i = 0
        if attn_weights_past is not None:
            # Past column
            ax_past = axes[row_i][col_i]
            col_i += 1

            attn_past_1 = attn_weights_past[idx:idx+1]   # shape [1, T, M, M]
            meals_past_1 = meal_ids_past[idx:idx+1]      # shape [1, T, M]

            packed_past, boundaries_past = pack_valid_attn_subblocks(attn_past_1, meals_past_1)
            if packed_past is None:
                # no valid meal tokens
                ax_past.text(0.5, 0.5, "No valid Past Meals", ha="center", va="center")
            else:
                im_past = ax_past.imshow(packed_past, aspect="auto", cmap=cmap, norm=norm)
                ax_past.set_title(f"Sample {idx} - Past Self-Attention")
                ax_past.set_xlabel("Token Position")
                ax_past.set_ylabel("Stacked Timesteps")
                if boundaries_past:
                    for b in boundaries_past[:-1]:
                        ax_past.axhline(y=b-0.5, color="white", linestyle="--", linewidth=0.8, alpha=0.6)
            
        if attn_weights_future is not None:
            # Future column
            ax_fut = axes[row_i][col_i]
            col_i += 1

            attn_fut_1 = attn_weights_future[idx:idx+1]  # shape [1, T, M, M]
            meals_fut_1 = meal_ids_future[idx:idx+1]     # shape [1, T, M]

            packed_fut, boundaries_future = pack_valid_attn_subblocks(attn_fut_1, meals_fut_1)
            if packed_fut is None:
                # no valid future meals
                ax_fut.text(0.5, 0.5, "No valid Future Meals", ha="center", va="center")
            else:
                im_fut = ax_fut.imshow(packed_fut, aspect="auto", cmap=cmap, norm=norm)
                ax_fut.set_title(f"Sample {idx} - Future Self-Attention")
                ax_fut.set_xlabel("Token Position")
                ax_fut.set_ylabel("Stacked Timesteps")
                if boundaries_future:
                    for b in boundaries_future[:-1]:
                        ax_fut.axhline(y=b-0.5, color="white", linestyle="--", linewidth=0.8, alpha=0.6)

    fig.suptitle("Meal Self-Attention for Selected Samples", fontsize=14, fontweight='bold')
    # We'll just add one colorbar for the entire figure (on the last axes we used).
    # If you want separate colorbars for each subplot, you can do so, but that can get busy.
    # Here, we'll do a single cbar spanning all:
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, fraction=0.02, pad=0.02, aspect=30)
    
    # Log to W&B
    logger.experiment.log({"meal_self_attention_samples": wandb.Image(fig), "global_step": global_step})
    plt.close(fig)
    return fig
