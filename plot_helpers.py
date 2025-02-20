import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

# -----------------------------------------------------------------------------
# Plotting Functions
# -----------------------------------------------------------------------------
def plot_forecast_examples(forecasts, attn_weights_past, attn_weights_future, quantiles, logger, global_step, fixed_indices=None):
    past = forecasts["past"]
    pred = forecasts["pred"]
    truth = forecasts["truth"]
    meal_ids_future = forecasts["future_meal_ids"]
    meal_ids_past = forecasts["past_meal_ids"]
    num_examples = min(4, past.size(0))
    if fixed_indices is None:
        fixed_indices = random.sample(list(range(past.size(0))), num_examples)
    sampled_indices = fixed_indices
    fig, axs = plt.subplots(num_examples, 3, figsize=(18, 4 * num_examples))
    if num_examples == 1:
        axs = [axs]
    for i, idx in enumerate(sampled_indices):
        ax_ts = axs[i][0]
        ax_attn_past = axs[i][1]
        ax_attn_future = axs[i][2]
        past_i = past[idx].cpu().numpy()
        pred_i = pred[idx].cpu().numpy()
        truth_i = truth[idx].cpu().numpy()
        attn_past_i = attn_weights_past[idx].cpu().numpy()
        attn_future_i = attn_weights_future[idx].cpu().numpy()
        T_context = past_i.shape[0]
        T_forecast = pred_i.shape[0]
        x_hist = list(range(-T_context + 1, 1))
        x_forecast = list(range(1, T_forecast + 1))
        ax_ts.plot(x_hist, past_i, marker="o", markersize=2, label="Historical Glucose")
        ax_ts.plot(x_forecast, truth_i, marker="o", markersize=2, label="Ground Truth Forecast")
        num_q = pred_i.shape[1]
        base_color = "blue"
        median_index = num_q // 2
        for qi in range(num_q - 1):
            alpha_val = 0.1 + (abs(qi - median_index)) * 0.05
            ax_ts.fill_between(x_forecast, pred_i[:, qi], pred_i[:, qi + 1], color=base_color, alpha=alpha_val / 2)
        ax_ts.plot(x_forecast, pred_i[:, median_index], marker="o", markersize=2, color="darkblue", label="Median Forecast")
        meal_label_added = False
        meals_past = meal_ids_past[idx].cpu().numpy()
        T_past = meals_past.shape[0]
        for t, meal in enumerate(meals_past):
            if (meal != 0).any():
                relative_time = t - T_past + 1
                if not meal_label_added:
                    ax_ts.axvline(x=relative_time, color="purple", linestyle="--", alpha=0.7, label="Meal Consumption")
                    meal_label_added = True
                else:
                    ax_ts.axvline(x=relative_time, color="purple", linestyle="--", alpha=0.7)
        meals_future = meal_ids_future[idx].cpu().numpy()
        for t, meal in enumerate(meals_future):
            if (meal != 0).any():
                relative_time = t + 1
                ax_ts.axvline(x=relative_time, color="purple", linestyle="--", alpha=0.7)
        ax_ts.set_xlabel("Relative Timestep")
        ax_ts.set_ylabel("Glucose Level")
        ax_ts.set_title(f"Forecast Example {i} (Idx: {idx})")
        ax_ts.legend(fontsize="small")
        
        im_past = ax_attn_past.imshow(attn_past_i, aspect="auto", cmap="viridis")
        
        
        num_past_timesteps = attn_past_i.shape[0]
        ax_attn_past.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax_attn_past.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{int(x - (num_past_timesteps - 1))}"))
        plt.setp(ax_attn_past.get_xticklabels(), rotation=90)
        
        ax_attn_past.set_title("Past Meals Attention")
        ax_attn_past.set_xlabel("Past Meal Timestep")
        ax_attn_past.set_ylabel("Glucose Timestep")
        
        
        fig.colorbar(im_past, ax=ax_attn_past, fraction=0.046, pad=0.04)
        im_future = ax_attn_future.imshow(attn_future_i, aspect="auto", cmap="viridis")
        ax_attn_future.set_title("Future Meals Attention")
        ax_attn_future.set_xlabel("Future Meal Timestep")
        ax_attn_future.set_ylabel("Glucose Timestep")
        fig.colorbar(im_future, ax=ax_attn_future, fraction=0.046, pad=0.04)
    fig.tight_layout()
    logger.experiment.log({"forecast_samples": wandb.Image(fig), "global_step": global_step})
    return fixed_indices, fig

def plot_iAUC_scatter(all_pred_iAUC, all_true_iAUC):
    mean_pred = torch.mean(all_pred_iAUC)
    mean_true = torch.mean(all_true_iAUC)
    cov = torch.mean((all_true_iAUC - mean_true) * (all_pred_iAUC - mean_pred))
    std_true = torch.std(all_true_iAUC, unbiased=False)
    std_pred = torch.std(all_pred_iAUC, unbiased=False)
    corr = cov / (std_true * std_pred)
    fig_scatter, ax_scatter = plt.subplots(figsize=(6, 6))
    ax_scatter.scatter(all_true_iAUC.cpu().numpy(), all_pred_iAUC.cpu().numpy(), alpha=0.5, s=0.5)
    ax_scatter.set_xlabel("True iAUC")
    ax_scatter.set_ylabel("Predicted iAUC")
    ax_scatter.set_title("iAUC Scatter Plot")
    ax_scatter.grid(True)
    ax_scatter.text(0.05, 0.95, f'Corr: {corr.item():.2f}', transform=ax_scatter.transAxes,
                    fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    return fig_scatter, corr

def pack_valid_attn_subblocks(attn_4d: torch.Tensor, meal_ids_3d: torch.Tensor):
    """
    Extract valid sub-blocks from attention [B,T,M,M] by removing padding around
    meal tokens.  Returns one "big" 2D numpy array, plus the total height (sum
    of valid M_i) and the max width (max(M_i)).
    
    attn_4d: [B, T, M, M] attention.  We'll ignore the start token dimension, so
             effectively we treat it as attn_4d[:, :, 1:, 1:].
    meal_ids_3d: [B, T, M] corresponding meal IDs.  A value of 0 means "padded".
    """
    # 1) Strip off the first row/col if you're ignoring the "start token"
    attn_4d = attn_4d[:, :, 1:, 1:]  # shape [B, T, M-1, M-1]
    meal_ids_3d = meal_ids_3d[:, :, 1:]  # shape [B, T, M-1]

    B, T, M, _ = attn_4d.shape
    # For each (b, t), figure out how many valid tokens there actually are:
    # (i.e. the number of non‐zero meal_ids).
    subblocks = []
    sizes = []
    for b in range(B):
        for t in range(T):
            meal_ids_slice = meal_ids_3d[b, t]  # shape [M]
            valid_count = (meal_ids_slice != 0).sum().item()
            if valid_count > 0:
                # slice out the valid sub-block [valid_count x valid_count]
                block = attn_4d[b, t, :valid_count, :valid_count]
                subblocks.append(block.cpu().numpy())
                sizes.append(valid_count)

    if not subblocks:
        # No valid sub‐blocks at all
        return np.zeros((1,1)), 1, 1

    # 2) Figure out how large an array we need:
    # Height is sum of all valid_counts, width is max of valid_counts
    total_height = sum(sizes)
    max_width = max(sizes)

    # 3) Create the big 2D array (we will fill it with NaN so that any "unfilled"
    # region is just blank).
    big_array = np.full((total_height, max_width), np.nan, dtype=np.float32)

    # 4) Copy each sub‐block in, one under the other
    row_offset = 0
    for block, size in zip(subblocks, sizes):
        big_array[row_offset:row_offset+size, 0:size] = block
        row_offset += size

    return big_array, total_height, max_width


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
    Show up to `max_examples` samples from the batch, each in its own row of a
    2-column figure: left = self-attn for the 'past' meals, right = self-attn
    for the 'future' meals.
    """
    # Decide which batch indices we will plot:
    batch_size = attn_weights_past.size(0)
    if fixed_indices is not None:
        # Use the fixed indices provided (limit to at most max_examples)
        indices = fixed_indices[:min(max_examples, len(fixed_indices))]
    elif random_samples:
        indices = random.sample(range(batch_size), k=min(max_examples, batch_size))
    else:
        indices = list(range(min(max_examples, batch_size)))

    # Increase figure size and adjust spacing
    fig, axes = plt.subplots(
        nrows=len(indices), 
        ncols=2, 
        figsize=(16, 5 * len(indices)),  # Wider figure, more height per row
        constrained_layout=True  # Better automatic layout handling
    )
    if len(indices) == 1:
        axes = [axes]

    # A helper that packs sub-blocks and returns boundaries to draw horizontal lines.
    def pack_with_boundaries(attn_4d, meal_ids_3d):
        """
        Returns (big_array, subblock_row_boundaries).

        big_array is the result of pack_valid_attn_subblocks.
        subblock_row_boundaries is the cumulative row index after each sub-block.
        """
        # Strip off "start token" dimension:
        attn_4d = attn_4d[:, :, 1:, 1:]    # shape [B, T, M-1, M-1]
        meal_ids_3d = meal_ids_3d[:, :, 1:]  # shape [B, T, M-1]

        # We only have a single sample in [B], so drop that dimension:
        attn_4d = attn_4d[0]      # [T, M-1, M-1]
        meal_ids_3d = meal_ids_3d[0]  # [T, M-1]

        subblocks = []
        sizes = []
        for t in range(attn_4d.size(0)):
            meal_ids_slice = meal_ids_3d[t]
            valid_count = (meal_ids_slice != 0).sum().item()
            if valid_count > 0:
                block = attn_4d[t, :valid_count, :valid_count]
                subblocks.append(block.cpu().numpy())
                sizes.append(valid_count)
        
        if not subblocks:
            return np.zeros((1, 1), dtype=np.float32), []

        # Create big packed array:
        total_height = sum(sizes)
        max_width = max(sizes)
        big_array = np.full((total_height, max_width), np.nan, dtype=np.float32)

        boundaries = []
        row_offset = 0
        for sz, sb in zip(sizes, subblocks):
            big_array[row_offset:row_offset+sz, 0:sz] = sb
            row_offset += sz
            boundaries.append(row_offset)  # The boundary after this sub-block

        return big_array, boundaries

    # Define a consistent colormap and normalization
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=0, vmax=1)

    # Loop over the chosen examples:
    for row_i, idx in enumerate(indices):
        ax_past = axes[row_i][0]
        ax_fut = axes[row_i][1]

        # Extract the single sample's T×M×M from the batch:
        attn_past_1 = attn_weights_past[idx:idx+1]   # shape [1,T,M,M]
        meals_past_1 = meal_ids_past[idx:idx+1]        # shape [1,T,M]
        attn_fut_1  = attn_weights_future[idx:idx+1]   # shape [1,T,M,M]
        meals_fut_1 = meal_ids_future[idx:idx+1]        # shape [1,T,M]

        # Pack and plot with improved formatting
        packed_past, boundaries_past = pack_with_boundaries(attn_past_1, meals_past_1)
        packed_fut, boundaries_future = pack_with_boundaries(attn_fut_1, meals_fut_1)

        # Past attention plot
        im_past = ax_past.imshow(packed_past, aspect="auto", cmap=cmap, norm=norm)
        ax_past.set_title(f"Sample {idx} - Past Self-Attention", pad=10, fontsize=12)
        ax_past.set_xlabel("Token Position", fontsize=10)
        ax_past.set_ylabel("Stacked Timesteps", fontsize=10)
        
        # Add boundaries with improved visibility
        for b in boundaries_past:
            ax_past.axhline(y=b-0.5, color="white", linestyle="--", linewidth=0.8, alpha=0.6)
        
        # Customize ticks
        ax_past.tick_params(axis='both', which='major', labelsize=9)

        # Future attention plot
        im_fut = ax_fut.imshow(packed_fut, aspect="auto", cmap=cmap, norm=norm)
        ax_fut.set_title(f"Sample {idx} - Future Self-Attention", pad=10, fontsize=12)
        ax_fut.set_xlabel("Token Position", fontsize=10)
        ax_fut.set_ylabel("Stacked Timesteps", fontsize=10)
        
        # Add boundaries with improved visibility
        for b in boundaries_future:
            ax_fut.axhline(y=b-0.5, color="white", linestyle="--", linewidth=0.8, alpha=0.6)
        
        # Customize ticks
        ax_fut.tick_params(axis='both', which='major', labelsize=9)

        # Add gridlines for better readability
        ax_past.grid(False)
        ax_fut.grid(False)

    # Main title with improved positioning
    fig.suptitle("Meal Self-Attention for Selected Samples", 
                 fontsize=14, 
                 y=1.01,  # Slightly adjusted for constrained_layout
                 fontweight='bold')

    # Add a single colorbar with better positioning and formatting
    cbar = fig.colorbar(im_fut, ax=axes, 
                       aspect=30)
    cbar.ax.set_ylabel("Attention Weight", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # Log to WandB
    logger.experiment.log({"meal_self_attention_samples": wandb.Image(fig), 
                          "global_step": global_step})
    plt.close(fig)
    return fig
