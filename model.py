import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
import random
import pandas as pd
import logging
from pytorch_lightning.loggers import WandbLogger
import wandb
import matplotlib.pyplot as plt
from utils import unscale_tensor
import click  # <-- click for CLI
import os
import pickle
import hashlib
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="The PyTorch API of nested tensors is in prototype stage.*"
)

# -----------------------------------------------------------------------------
# Global logging configuration
# -----------------------------------------------------------------------------
VERBOSE_LOGGING = False

logging.basicConfig(
    level=logging.DEBUG if VERBOSE_LOGGING else logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def log_tensor_stats(name, tensor, step=None):
    """
    Logs basic statistics of a tensor—including a histogram—to WandB.
    Pass a training/logging step if available.
    """
    if not VERBOSE_LOGGING:
        return

    if torch.isnan(tensor).any():
        logging.error(f"{name} has NaNs! (shape={tensor.shape})")
        wandb.log({f"{name}_error": f"NaNs found in tensor {name}"}, step=step)
        return

    mean_val = tensor.mean().item()
    std_val = tensor.std().item()
    nan_count = torch.isnan(tensor).sum().item()
    logging.debug(
        f"{name}: shape={tensor.shape}, mean={mean_val:.6f}, std={std_val:.6f}, nans={nan_count}"
    )

    wandb.log(
        {
            f"{name}/histogram": wandb.Histogram(tensor.detach().cpu().numpy()),
            f"{name}/mean": mean_val,
            f"{name}/std": std_val,
            f"{name}/nan_count": nan_count,
        },
        step=step,
    )

# -----------------------------
# Model Components
# -----------------------------
class MealEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_foods: int,
        macro_dim: int,
        max_meals: int = 11,
        num_heads: int = 4,
        num_layers: int = 1,
    ):
        super(MealEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.max_meals = max_meals
        self.num_foods = num_foods
        self.macro_dim = macro_dim

        self.food_emb = nn.Embedding(num_foods, embed_dim, padding_idx=0)
        self.macro_proj = nn.Linear(macro_dim, embed_dim, bias=False)
        self.pos_emb = nn.Embedding(max_meals, embed_dim)

        self.start_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, meal_ids: torch.LongTensor, meal_macros: torch.Tensor) -> torch.Tensor:
        B, T, M = meal_ids.size()
        if VERBOSE_LOGGING:
            logging.debug(
                f"MealEncoder: Input meal_ids.shape: {meal_ids.shape}, "
                f"meal_macros.shape: {meal_macros.shape}"
            )

        meal_ids_flat = meal_ids.view(B * T, M)
        meal_macros_flat = meal_macros.view(B * T, M, -1)

        meal_id_emb = self.food_emb(meal_ids_flat)  # [B*T, M, embed_dim]
        meal_macro_emb = self.macro_proj(meal_macros_flat)  # [B*T, M, embed_dim]
        meal_token_emb = meal_id_emb + meal_macro_emb       # [B*T, M, embed_dim]

        if VERBOSE_LOGGING:
            logging.debug(
                f"MealEncoder: meal_token_emb after summing embeddings: {meal_token_emb.shape}"
            )

        pos_indices = torch.arange(self.max_meals, device=meal_ids.device)
        pos_enc = self.pos_emb(pos_indices).unsqueeze(0)  # [1, M, embed_dim]
        meal_token_emb = meal_token_emb + pos_enc         # [B*T, M, embed_dim]

        if VERBOSE_LOGGING:
            logging.debug("MealEncoder: Added positional encoding.")

        start_token_expanded = self.start_token.expand(B * T, -1, -1)
        meal_token_emb = torch.cat([start_token_expanded, meal_token_emb], dim=1)

        if VERBOSE_LOGGING:
            logging.debug(
                f"MealEncoder: After prepending start token, shape: {meal_token_emb.shape}"
            )

        pad_mask = meal_ids_flat == 0  # [B*T, M]
        pad_mask = torch.cat(
            [torch.zeros(B * T, 1, device=pad_mask.device, dtype=torch.bool), pad_mask],
            dim=1,
        )

        meal_attn_out = self.encoder(
            meal_token_emb, src_key_padding_mask=pad_mask
        )  # [B*T, 1+M, embed_dim]

        if VERBOSE_LOGGING:
            logging.debug(
                f"MealEncoder: Transformer encoder output shape: {meal_attn_out.shape}"
            )

        meal_timestep_emb = meal_attn_out[:, 0, :]  # [B*T, embed_dim]
        meal_timestep_emb = meal_timestep_emb.view(B, T, self.embed_dim)

        if VERBOSE_LOGGING:
            logging.debug(
                f"MealEncoder: Final meal timestep embedding shape: {meal_timestep_emb.shape}"
            )
        return meal_timestep_emb


class GlucoseEncoder(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int = 4, num_layers: int = 1, max_seq_len: int = 100
    ):
        super(GlucoseEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.glucose_proj = nn.Linear(1, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, glucose_seq: torch.Tensor) -> torch.Tensor:
        B, T = glucose_seq.size()
        if VERBOSE_LOGGING:
            logging.debug(f"GlucoseEncoder: Input glucose_seq shape: {glucose_seq.shape}")
        x = glucose_seq.unsqueeze(-1)  # [B, T, 1]
        x = self.glucose_proj(x)       # [B, T, embed_dim]
        log_tensor_stats("Glucose projection", x)

        pos_indices = torch.arange(T, device=glucose_seq.device)
        pos_enc = self.pos_emb(pos_indices).unsqueeze(0)  # [1, T, embed_dim]
        x = x + pos_enc
        log_tensor_stats("Glucose with pos enc", x)

        x = self.encoder(x)
        log_tensor_stats("Glucose encoder output", x)
        return x


class MealGlucoseForecastModel(pl.LightningModule):
    def __init__(
        self,
        embed_dim=32,
        num_foods=100,
        macro_dim=5,
        max_meals=11,
        glucose_seq_len=20,
        forecast_horizon=4,
        num_heads=4,
        enc_layers=1,
        residual_pred=True,
    ):
        super(MealGlucoseForecastModel, self).__init__()
        self.embed_dim = embed_dim
        self.num_foods = num_foods
        self.forecast_horizon = forecast_horizon
        self.residual_pred = residual_pred

        self.meal_encoder = MealEncoder(
            embed_dim, num_foods, macro_dim, max_meals, num_heads, num_layers=enc_layers
        )
        self.glucose_encoder = GlucoseEncoder(
            embed_dim, num_heads, num_layers=enc_layers, max_seq_len=glucose_seq_len
        )
        self.future_meal_encoder = MealEncoder(
            embed_dim, num_foods, macro_dim, max_meals, num_heads, num_layers=enc_layers
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )

        hidden_dim = embed_dim * 2
        self.forecast_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, forecast_horizon),
        )

        self.last_attn_weights = None
        self.example_forecasts = None

    def forward(
        self,
        past_glucose,
        past_meal_ids,
        past_meal_macros,
        future_meal_ids,
        future_meal_macros,
        target_scales,
        return_attn=False,
    ):
        logging.debug("MealGlucoseForecastModel: --- Forward Pass Start ---")

        glucose_enc = self.glucose_encoder(past_glucose)  # [B, T, embed_dim]
        past_meal_enc = self.meal_encoder(
            past_meal_ids, past_meal_macros
        )  # [B, T, embed_dim]
        future_meal_enc = self.future_meal_encoder(
            future_meal_ids, future_meal_macros
        )  # [B, forecast_horizon, embed_dim]

        attn_output, attn_weights = self.cross_attn(
            query=glucose_enc, key=past_meal_enc, value=past_meal_enc, need_weights=True
        )

        combined_glucose = attn_output + glucose_enc  # [B, T, embed_dim]
        self.last_attn_weights = attn_weights

        final_rep = torch.cat(
            [combined_glucose[:, -1, :], future_meal_enc[:, -1, :]], dim=-1
        )  # [B, embed_dim*2]

        pred_future = self.forecast_mlp(final_rep)  # [B, forecast_horizon]

        if self.residual_pred:
            last_val = past_glucose[:, -1].unsqueeze(1)
            pred_future = pred_future + last_val

        # Unscale the output predictions using the provided helper
        pred_future = unscale_tensor(pred_future, target_scales)

        logging.debug("MealGlucoseForecastModel: --- Forward Pass End ---")
        if return_attn:
            return pred_future, past_meal_enc, attn_weights
        return pred_future

    def validation_step(self, batch, batch_idx):
        logging.debug(f"Validation step: batch_idx {batch_idx}")
        (
            past_glucose,
            past_meal_ids,
            past_meal_macros,
            future_meal_ids,
            future_meal_macros,
            future_glucose,
            target_scales,
        ) = batch

        if target_scales.dim() > 2:
            target_scales = target_scales.view(target_scales.size(0), -1)

        preds = self(
            past_glucose,
            past_meal_ids,
            past_meal_macros,
            future_meal_ids,
            future_meal_macros,
            target_scales,
        )

        future_glucose_unscaled = (
            future_glucose * target_scales[:, 1].unsqueeze(1)
            + target_scales[:, 0].unsqueeze(1)
        )
        val_loss = F.mse_loss(preds, future_glucose_unscaled)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True)

        val_rmse = torch.sqrt(val_loss)
        self.log("val_rmse", val_rmse, on_step=False, on_epoch=True)
        logging.debug(f"Validation loss: {val_loss.item()}")

        # --- iAUC Calculation ---
        past_glucose_unscaled = unscale_tensor(past_glucose, target_scales)
        baseline = past_glucose_unscaled[:, -2:].mean(dim=1)  # shape: [B]

        pred_diff = preds - baseline.unsqueeze(1)  # [B, T_forecast]
        true_diff = future_glucose_unscaled - baseline.unsqueeze(1)  # [B, T_forecast]

        pred_iAUC = torch.trapz(torch.clamp(pred_diff, min=0), dx=1, dim=1)  # [B]
        true_iAUC = torch.trapz(torch.clamp(true_diff, min=0), dx=1, dim=1)  # [B]

        if not hasattr(self, "val_outputs"):
            self.val_outputs = []
        self.val_outputs.append(
            {
                "val_loss": val_loss,
                "pred_iAUC": pred_iAUC,
                "true_iAUC": true_iAUC,
            }
        )

        if batch_idx == 0:
            self.example_forecasts = {
                "past": past_glucose_unscaled.detach().cpu(),
                "pred": preds.detach().cpu(),
                "truth": future_glucose_unscaled.detach().cpu(),
                "future_meal_ids": future_meal_ids.detach().cpu(),
                "past_meal_ids": past_meal_ids.detach().cpu(),
            }

        return {"val_loss": val_loss, "pred_iAUC": pred_iAUC, "true_iAUC": true_iAUC}

    def on_validation_epoch_end(self):
        if not hasattr(self, "val_outputs") or len(self.val_outputs) == 0:
            return
        outputs = self.val_outputs

        # ----- Plot Forecast Examples -----
        if self.example_forecasts is not None:
            forecasts = self.example_forecasts
            past = forecasts["past"]
            pred = forecasts["pred"]
            truth = forecasts["truth"]
            future_meal_ids = forecasts["future_meal_ids"]
            past_meal_ids = forecasts.get("past_meal_ids")
            num_examples = min(4, past.size(0))

            fig, axs = plt.subplots(num_examples, 1, figsize=(12, 4 * num_examples))
            if num_examples == 1:
                axs = [axs]

            for i in range(num_examples):
                past_i = past[i].numpy()
                pred_i = pred[i].numpy()
                truth_i = truth[i].numpy()
                meals_i = future_meal_ids[i].numpy()
                T_context = past_i.shape[0]
                T_forecast = pred_i.shape[0]

                x_hist = list(range(-T_context + 1, 1))
                x_forecast = list(range(1, T_forecast + 1))

                ax = axs[i]
                ax.plot(x_hist, past_i, marker="o", label="Historical Glucose")

                if past_meal_ids is not None:
                    past_meals_i = past_meal_ids[i].numpy()
                    meal_label_added_hist = False
                    for j, x_coord in enumerate(x_hist):
                        if (past_meals_i[j] != 0).any():
                            if not meal_label_added_hist:
                                ax.axvline(
                                    x=x_coord,
                                    color="green",
                                    linestyle="--",
                                    alpha=0.7,
                                    label="Historical Meal Consumption",
                                )
                                meal_label_added_hist = True
                            else:
                                ax.axvline(
                                    x=x_coord, color="green", linestyle="--", alpha=0.7
                                )

                ax.plot(x_forecast, truth_i, marker="o", label="Ground Truth Forecast")
                ax.plot(x_forecast, pred_i, marker="o", label="Prediction Forecast")

                meal_label_added_forecast = False
                for t, x_coord in enumerate(x_forecast):
                    if (meals_i[t] != 0).any():
                        if not meal_label_added_forecast:
                            ax.axvline(
                                x=x_coord,
                                color="red",
                                linestyle="--",
                                alpha=0.7,
                                label="Forecast Meal Consumption",
                            )
                            meal_label_added_forecast = True
                        else:
                            ax.axvline(
                                x=x_coord, color="red", linestyle="--", alpha=0.7
                            )

                ax.set_xlabel("Relative Timestep")
                ax.set_ylabel("Glucose Level")
                ax.set_title(f"Forecast Example {i}")
                ax.legend(fontsize="small")
            fig.tight_layout()
            self.logger.experiment.log(
                {"forecast_examples": wandb.Image(fig), "global_step": self.global_step}
            )
            plt.close(fig)
            self.example_forecasts = None

        # ----- Plot the Attention Heatmap -----
        if self.last_attn_weights is not None:
            attn = self.last_attn_weights[0].detach().cpu().numpy()
            fig, ax = plt.subplots(figsize=(8, 6))
            cax = ax.imshow(attn, aspect="auto", cmap="viridis")
            fig.colorbar(cax, ax=ax)
            ax.set_xlabel("Past Meal Timestep")
            ax.set_ylabel("Glucose Timestep")
            ax.set_title("Cross-Attention Weights\n(Glucose Queries vs. Meal Keys)")
            fig.tight_layout()
            self.logger.experiment.log(
                {"attention_weights": wandb.Image(fig), "global_step": self.global_step}
            )
            plt.close(fig)

        # ----- Compute iAUC Correlation and Log Scatter Plot -----
        all_pred_iAUC = torch.cat([o["pred_iAUC"] for o in outputs], dim=0)
        all_true_iAUC = torch.cat([o["true_iAUC"] for o in outputs], dim=0)

        x_mean = all_true_iAUC.mean()
        y_mean = all_pred_iAUC.mean()
        cov = torch.sum((all_true_iAUC - x_mean) * (all_pred_iAUC - y_mean))
        std_x = torch.sqrt(torch.sum((all_true_iAUC - x_mean) ** 2))
        std_y = torch.sqrt(torch.sum((all_pred_iAUC - y_mean) ** 2))
        iAUC_corr = cov / (std_x * std_y + 1e-8)

        self.log("val_iAUC_corr", iAUC_corr, prog_bar=True)
        self.logger.experiment.log(
            {"val_iAUC_corr": iAUC_corr.item(), "global_step": self.global_step}
        )

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(
            all_true_iAUC.cpu().numpy(),
            all_pred_iAUC.cpu().numpy(),
            alpha=0.5,
            s=0.5,
        )
        min_val = min(all_true_iAUC.min().item(), all_pred_iAUC.min().item())
        max_val = max(all_true_iAUC.max().item(), all_pred_iAUC.max().item())
        ax.plot([min_val, max_val], [min_val, max_val], "r--")
        ax.set_xlabel("Real iAUC")
        ax.set_ylabel("Predicted iAUC")
        ax.set_title("Predicted vs. Real iAUC")
        fig.tight_layout()

        self.logger.experiment.log({"iAUC_scatter": wandb.Image(fig), "global_step": self.global_step})
        plt.close(fig)

        self.val_outputs.clear()

    def training_step(self, batch, batch_idx):
        logging.debug(f"Training step: batch_idx {batch_idx}")
        (
            past_glucose,
            past_meal_ids,
            past_meal_macros,
            future_meal_ids,
            future_meal_macros,
            future_glucose,
            target_scales,
        ) = batch

        if target_scales.dim() > 2:
            target_scales = target_scales.view(target_scales.size(0), -1)

        preds = self(
            past_glucose,
            past_meal_ids,
            past_meal_macros,
            future_meal_ids,
            future_meal_macros,
            target_scales,
        )

        future_glucose_unscaled = (
            future_glucose * target_scales[:, 1].unsqueeze(1)
            + target_scales[:, 0].unsqueeze(1)
        )

        loss = F.mse_loss(preds, future_glucose_unscaled)
        self.log("train_loss", loss, on_step=False, on_epoch=True)

        train_rmse = torch.sqrt(loss)
        self.log("train_rmse", train_rmse, on_step=False, on_epoch=True)

        logging.debug(f"Training loss: {loss.item()}")
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def on_train_epoch_end(self):
        for name, param in self.named_parameters():
            log_tensor_stats(f"Parameter/{name}", param, step=self.current_epoch)


# -----------------------------
# Dummy Example Dataset (only if needed)
# -----------------------------
class DummyMealGlucoseDataset(Dataset):
    def __init__(
        self,
        num_sequences=1000,
        past_seq_length=20,
        forecast_horizon=4,
        num_foods=100,
        max_meals=11,
        macro_dim=5,
    ):
        super().__init__()
        self.num_sequences = num_sequences
        self.past_seq_length = past_seq_length
        self.forecast_horizon = forecast_horizon
        self.max_meals = max_meals
        self.macro_dim = macro_dim
        self.num_foods = num_foods
        self.data = []
        for _ in range(num_sequences):
            past_glucose = torch.rand(past_seq_length)
            future_glucose = torch.rand(forecast_horizon)

            past_meal_ids = torch.zeros(past_seq_length, max_meals, dtype=torch.long)
            past_meal_macros = torch.zeros(
                past_seq_length, max_meals, macro_dim, dtype=torch.float
            )
            for t in range(past_seq_length):
                num_meals = random.randint(0, 3)
                num_meals = min(num_meals, max_meals)
                for k in range(num_meals):
                    food_id = random.randint(1, num_foods - 1)
                    macros = torch.rand(macro_dim)
                    past_meal_ids[t, k] = food_id
                    past_meal_macros[t, k] = macros

            future_meal_ids = torch.zeros(forecast_horizon, max_meals, dtype=torch.long)
            future_meal_macros = torch.zeros(
                forecast_horizon, max_meals, macro_dim, dtype=torch.float
            )
            for t in range(forecast_horizon):
                num_meals = random.randint(0, 3)
                num_meals = min(num_meals, max_meals)
                for k in range(num_meals):
                    food_id = random.randint(1, num_foods - 1)
                    macros = torch.rand(macro_dim)
                    future_meal_ids[t, k] = food_id
                    future_meal_macros[t, k] = macros

            self.data.append(
                (
                    past_glucose,
                    past_meal_ids,
                    past_meal_macros,
                    future_meal_ids,
                    future_meal_macros,
                    future_glucose,
                )
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# -----------------------------------------------------------------------------
# NEW create_cached_dataset function that caches all splits + encoders/scalers
# -----------------------------------------------------------------------------
def create_cached_dataset(
    dataset_version,
    debug_mode,
    validation_percentage,
    test_percentage,
    min_encoder_length,
    prediction_length,
    is_food_anchored,
    sliding_window_stride,
    use_meal_level_food_covariates,
    use_microbiome_embeddings,
    group_by_columns,
    temporal_categoricals,
    temporal_reals,
    user_static_categoricals,
    user_static_reals,
    food_categoricals,
    food_reals,
    targets,
    cache_dir,
    use_cache=True,
):
    """
    Create or load a cached dataset containing training, validation, and test splits,
    along with their scalers/encoders. If a matching cache file is found and use_cache
    is True, we use the cached result.

    This version uses torch.save and torch.load for dataset caching.
    """
    # Local imports for your pipeline
    from dataset import PPGRTimeSeriesDataset, PPGRToMealGlucoseWrapper, split_timeseries_df_based_on_food_intake_rows
    from utils import load_dataframe, enforce_column_types, setup_scalers_and_encoders

    # 1) Define config that influences final dataset creation
    config = {
        "dataset_version": dataset_version,
        "debug_mode": debug_mode,
        "validation_percentage": validation_percentage,
        "test_percentage": test_percentage,
        "min_encoder_length": min_encoder_length,
        "prediction_length": prediction_length,
        "is_food_anchored": is_food_anchored,
        "sliding_window_stride": sliding_window_stride,
        "use_meal_level_food_covariates": use_meal_level_food_covariates,
        "use_microbiome_embeddings": use_microbiome_embeddings,
        "group_by_columns": group_by_columns,
        "temporal_categoricals": temporal_categoricals,
        "temporal_reals": temporal_reals,
        "user_static_categoricals": user_static_categoricals,
        "user_static_reals": user_static_reals,
        "food_categoricals": food_categoricals,
        "food_reals": food_reals,
        "targets": targets,
    }

    # 2) Compute a unique hash from the config
    import pickle  # Only used for hashing the config; caching itself is now handled by torch.save
    config_bytes = pickle.dumps(config)
    config_hash = hashlib.md5(config_bytes).hexdigest()
    cache_file = os.path.join(cache_dir, f"{config_hash}_all_splits.pt")  # Updated extension to .pt

    # 3) If cache exists and caching is enabled, load & return immediately using torch.load
    if use_cache and os.path.exists(cache_file):
        logging.info(f"[CACHE-HIT] Loading pipeline from: {cache_file}")
        cached_data = torch.load(cache_file)
        return (
            cached_data["training_dataset"],
            cached_data["validation_dataset"],
            cached_data["test_dataset"],
            cached_data["categorical_encoders"],
            cached_data["continuous_scalers"],
        )

    # 4) Otherwise, proceed to load & build everything
    logging.info("[CACHE-MISS] Building dataset from scratch...")
    os.makedirs(cache_dir, exist_ok=True)

    # Load base dataframes
    ppgr_df, users_demographics_df, dishes_df, microbiome_embeddings_df = load_dataframe(
        dataset_version, debug_mode
    )

    # Split the dataframes
    (training_df, validation_df, test_df) = split_timeseries_df_based_on_food_intake_rows(
        ppgr_df,
        validation_percentage=validation_percentage,
        test_percentage=test_percentage,
    )

    # Columns to enforce
    main_df_scaled_all_categorical_columns = (
        user_static_categoricals + food_categoricals + temporal_categoricals
    )
    main_df_scaled_all_real_columns = (
        user_static_reals + food_reals + temporal_reals + targets
    )

    # Enforce types
    ppgr_df, users_demographics_df, dishes_df = enforce_column_types(
        ppgr_df,
        users_demographics_df,
        dishes_df,
        main_df_scaled_all_categorical_columns,
        main_df_scaled_all_real_columns,
    )

    # Fit encoders/scalers on training data
    categorical_encoders, continuous_scalers = setup_scalers_and_encoders(
        ppgr_df=ppgr_df,
        training_df=training_df,
        users_demographics_df=users_demographics_df,
        dishes_df=dishes_df,
        categorical_columns=main_df_scaled_all_categorical_columns,
        real_columns=main_df_scaled_all_real_columns,
        use_meal_level_food_covariates=use_meal_level_food_covariates,
    )

    # Build PPGRTimeSeriesDataset for each split
    training_dataset = PPGRTimeSeriesDataset(
        ppgr_df=training_df,
        user_demographics_df=users_demographics_df,
        dishes_df=dishes_df,
        time_idx="read_at",
        target_columns=targets,
        group_by_columns=group_by_columns,
        is_food_anchored=is_food_anchored,
        sliding_window_stride=sliding_window_stride,
        min_encoder_length=min_encoder_length,
        prediction_length=prediction_length,
        use_food_covariates_from_prediction_window=True,
        use_meal_level_food_covariates=use_meal_level_food_covariates,
        use_microbiome_embeddings=use_microbiome_embeddings,
        microbiome_embeddings_df=microbiome_embeddings_df,
        temporal_categoricals=temporal_categoricals,
        temporal_reals=temporal_reals,
        user_static_categoricals=user_static_categoricals,
        user_static_reals=user_static_reals,
        food_categoricals=food_categoricals,
        food_reals=food_reals,
        categorical_encoders=categorical_encoders,
        continuous_scalers=continuous_scalers,
    )

    validation_dataset = PPGRTimeSeriesDataset(
        ppgr_df=validation_df,
        user_demographics_df=users_demographics_df,
        dishes_df=dishes_df,
        time_idx="read_at",
        target_columns=targets,
        group_by_columns=group_by_columns,
        is_food_anchored=is_food_anchored,
        sliding_window_stride=sliding_window_stride,
        min_encoder_length=min_encoder_length,
        prediction_length=prediction_length,
        use_food_covariates_from_prediction_window=True,
        use_meal_level_food_covariates=use_meal_level_food_covariates,
        use_microbiome_embeddings=use_microbiome_embeddings,
        microbiome_embeddings_df=microbiome_embeddings_df,
        temporal_categoricals=temporal_categoricals,
        temporal_reals=temporal_reals,
        user_static_categoricals=user_static_categoricals,
        user_static_reals=user_static_reals,
        food_categoricals=food_categoricals,
        food_reals=food_reals,
        categorical_encoders=categorical_encoders,
        continuous_scalers=continuous_scalers,
    )

    test_dataset = PPGRTimeSeriesDataset(
        ppgr_df=test_df,
        user_demographics_df=users_demographics_df,
        dishes_df=dishes_df,
        time_idx="read_at",
        target_columns=targets,
        group_by_columns=group_by_columns,
        is_food_anchored=is_food_anchored,
        sliding_window_stride=sliding_window_stride,
        min_encoder_length=min_encoder_length,
        prediction_length=prediction_length,
        use_food_covariates_from_prediction_window=True,
        use_meal_level_food_covariates=use_meal_level_food_covariates,
        use_microbiome_embeddings=use_microbiome_embeddings,
        microbiome_embeddings_df=microbiome_embeddings_df,
        temporal_categoricals=temporal_categoricals,
        temporal_reals=temporal_reals,
        user_static_categoricals=user_static_categoricals,
        user_static_reals=user_static_reals,
        food_categoricals=food_categoricals,
        food_reals=food_reals,
        categorical_encoders=categorical_encoders,
        continuous_scalers=continuous_scalers,
    )

    wrapped_training_dataset = PPGRToMealGlucoseWrapper(training_dataset)
    wrapped_validation_dataset = PPGRToMealGlucoseWrapper(validation_dataset)
    wrapped_test_dataset = PPGRToMealGlucoseWrapper(test_dataset)

    # Pack everything to cache using torch.save
    cache_dict = {
        "training_dataset": wrapped_training_dataset,
        "validation_dataset": wrapped_validation_dataset,
        "test_dataset": wrapped_test_dataset,
        "categorical_encoders": categorical_encoders,
        "continuous_scalers": continuous_scalers,
    }
    torch.save(cache_dict, cache_file)
    logging.info(f"Dataset pipeline built and saved to cache: {cache_file}")
    return (
        wrapped_training_dataset,
        wrapped_validation_dataset,
        wrapped_test_dataset,
        categorical_encoders,
        continuous_scalers,
    )


@click.command()
@click.option("--debug/--no-debug", default=False, help="Enable debug logging.")
@click.option("--no-cache", is_flag=True, default=False, help="Ignore cached datasets and rebuild dataset from scratch.")
def main(debug, no_cache):
    """
    Main entry point of the script.
    Pass --debug to enable detailed logging.
    Pass --no-cache to ignore any cached datasets.
    """
    global VERBOSE_LOGGING
    VERBOSE_LOGGING = debug
    logging.getLogger().setLevel(logging.DEBUG if debug else logging.INFO)
    logging.info(f"Starting main block with debug={debug}, no_cache={no_cache}.")

    ############################################################################
    # Hyperparameters & Configurations
    ############################################################################
    dataset_version = "v0.4"
    debug_mode = debug
    min_encoder_length = 8 * 4
    prediction_length = 2 * 4
    validation_percentage = 0.2
    test_percentage = 0.2

    is_food_anchored = True
    sliding_window_stride = None
    use_meal_level_food_covariates = True
    use_microbiome_embeddings = True
    group_by_columns = ["timeseries_block_id"]

    user_static_categoricals = [
        "user_id",
        "user__edu_degree",
        "user__income",
        "user__household_desc",
        "user__job_status",
        "user__smoking",
        "user__health_state",
        "user__physical_activities_frequency",
    ]
    user_static_reals = [
        "user__age",
        "user__weight",
        "user__height",
        "user__bmi",
        "user__general_hunger_level",
        "user__morning_hunger_level",
        "user__mid_hunger_level",
        "user__evening_hunger_level",
    ]

    if use_meal_level_food_covariates:
        food_categoricals = ["food__food_group_cname", "food_id"]
    else:
        food_categoricals = [
            "food__vegetables_fruits",
            "food__grains_potatoes_pulses",
            "food__unclassified",
            "food__non_alcoholic_beverages",
            "food__dairy_products_meat_fish_eggs_tofu",
            "food__sweets_salty_snacks_alcohol",
            "food__oils_fats_nuts",
        ]
    food_reals = [
        "food__eaten_quantity_in_gram",
        "food__energy_kcal_eaten",
        "food__carb_eaten",
        "food__fat_eaten",
        "food__protein_eaten",
        "food__fiber_eaten",
        "food__alcohol_eaten",
    ]

    temporal_categoricals = ["loc_eaten_dow", "loc_eaten_dow_type", "loc_eaten_season"]
    temporal_reals = ["loc_eaten_hour"]
    targets = ["val"]

    glucose_seq_len = min_encoder_length
    forecast_horizon = prediction_length
    embedding_dim = 256
    num_heads = 4
    enc_layers = 2
    residual_pred = False
    batch_size = 1024
    max_epochs = 30
    optimizer_lr = 1e-4

    wandb_project = "meal-representations-learning-v0"
    wandb_run_name = "MealGlucoseForecastModel_Run"

    cache_dir = "/scratch/mohanty/food/ppgr-v1/datasets-cache"

    # -------------------------------------------------------------------------
    # Use the new caching function with the no_cache flag
    # -------------------------------------------------------------------------
    (
        training_dataset,
        validation_dataset,
        test_dataset,
        categorical_encoders,
        continuous_scalers,
    ) = create_cached_dataset(
        dataset_version=dataset_version,
        debug_mode=debug_mode,
        validation_percentage=validation_percentage,
        test_percentage=test_percentage,
        min_encoder_length=min_encoder_length,
        prediction_length=prediction_length,
        is_food_anchored=is_food_anchored,
        sliding_window_stride=sliding_window_stride,
        use_meal_level_food_covariates=use_meal_level_food_covariates,
        use_microbiome_embeddings=use_microbiome_embeddings,
        group_by_columns=group_by_columns,
        temporal_categoricals=temporal_categoricals,
        temporal_reals=temporal_reals,
        user_static_categoricals=user_static_categoricals,
        user_static_reals=user_static_reals,
        food_categoricals=food_categoricals,
        food_reals=food_reals,
        targets=targets,
        cache_dir=cache_dir,
        use_cache=not no_cache,
    )

    ############################################################################
    # Create DataLoaders
    ############################################################################
    from torch.utils.data import DataLoader

    train_loader = DataLoader(training_dataset, batch_size=batch_size, num_workers=7, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=7)

    ############################################################################
    # Model Initialization
    ############################################################################
    model = MealGlucoseForecastModel(
        embed_dim=embedding_dim,
        num_foods=training_dataset.num_foods,
        macro_dim=training_dataset.num_nutrients,
        max_meals=training_dataset.max_meals,
        glucose_seq_len=glucose_seq_len,
        forecast_horizon=forecast_horizon,
        num_heads=num_heads,
        enc_layers=enc_layers,
        residual_pred=residual_pred,
    )

    ############################################################################
    # WandB Logger Setup & Training
    ############################################################################
    wandb_logger = WandbLogger(
        project=wandb_project,
        name=wandb_run_name,
        config={
            "embed_dim": embedding_dim,
            "num_foods": training_dataset.num_foods,
            "macro_dim": training_dataset.num_nutrients,
            "max_meals": training_dataset.max_meals,
            "glucose_seq_len": glucose_seq_len,
            "forecast_horizon": forecast_horizon,
            "batch_size": batch_size,
            "optimizer_lr": optimizer_lr,
            "verbose_logging": VERBOSE_LOGGING,
        },
        log_model=True,
    )
    logging.info("WandB Logger initialized.")

    trainer = pl.Trainer(max_epochs=max_epochs, enable_checkpointing=False, logger=wandb_logger)
    logging.info("Starting training.")
    trainer.fit(model, train_loader, val_loader)
    logging.info("Training complete.")

    ############################################################################
    # Evaluation on a Validation Batch
    ############################################################################
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))
        (
            past_glucose,
            past_meal_ids,
            past_meal_macros,
            future_meal_ids,
            future_meal_macros,
            future_glucose,
            target_scales,
        ) = [x.to(device) for x in batch]

        preds, past_meal_embeds, attn_weights = model(
            past_glucose,
            past_meal_ids,
            past_meal_macros,
            future_meal_ids,
            future_meal_macros,
            target_scales,
            return_attn=True,
        )

    logging.info("Predicted future glucose (first sample): %s", preds[0].cpu().numpy())
    logging.info("Actual future glucose (first sample):   %s", future_glucose[0].cpu().numpy())
    logging.info("Past meal embedding shape: %s", past_meal_embeds.shape)
    logging.info("Cross-attention weight shape: %s", attn_weights.shape)
    logging.info("Attention weights (first sample): %s", attn_weights[0])


if __name__ == "__main__":
    main()
