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

# -----------------------------------------------------------------------------
# Global logging configuration
# -----------------------------------------------------------------------------
# Set this flag to False to disable detailed debug logging.
VERBOSE_LOGGING = True

logging.basicConfig(
    level=logging.DEBUG if VERBOSE_LOGGING else logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# -----------------------------------------------------------------------------
# Utility Logging Function
# -----------------------------------------------------------------------------
def log_tensor_stats(name, tensor):
    """Utility to log basic statistics of a tensor.
       Logs only when VERBOSE_LOGGING is True.
    """
    if not VERBOSE_LOGGING:
        return
    if torch.isnan(tensor).any():
        logging.error(f"{name} has NaNs! (shape={tensor.shape})")
    else:
        mean_val = tensor.mean().item()
        std_val = tensor.std().item()
        nan_count = torch.isnan(tensor).sum().item()
        logging.debug(f"{name}: shape={tensor.shape}, mean={mean_val:.6f}, std={std_val:.6f}, nans={nan_count}")

# -----------------------------
# Model Components with Debug Logs
# -----------------------------

class MealEncoder(nn.Module):
    def __init__(self, embed_dim: int, num_foods: int, macro_dim: int, max_meals: int = 11, num_heads: int = 4, num_layers: int = 1):
        super(MealEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.max_meals = max_meals
        
        self.food_emb = nn.Embedding(num_foods, embed_dim, padding_idx=0)
        self.macro_proj = nn.Linear(macro_dim, embed_dim, bias=False)
        self.pos_emb = nn.Embedding(max_meals, embed_dim)
        
        # Add a learnable start token embedding (one per timestep)
        self.start_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 2, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, meal_ids: torch.LongTensor, meal_macros: torch.Tensor) -> torch.Tensor:
        B, T, M = meal_ids.size()
        if VERBOSE_LOGGING:
            logging.debug(f"MealEncoder: Input meal_ids.shape: {meal_ids.shape}, meal_macros.shape: {meal_macros.shape}")
        # Flatten batch and time dims for processing: (B*T, M)
        meal_ids_flat = meal_ids.view(B * T, M)
        meal_macros_flat = meal_macros.view(B * T, M, -1)
        meal_id_emb = self.food_emb(meal_ids_flat)              # [B*T, M, embed_dim]
        meal_macro_emb = self.macro_proj(meal_macros_flat)       # [B*T, M, embed_dim]
        meal_token_emb = meal_id_emb + meal_macro_emb            # [B*T, M, embed_dim]
        if VERBOSE_LOGGING:
            logging.debug(f"MealEncoder: meal_token_emb after summing embeddings: {meal_token_emb.shape}")

        # Add positional encoding for each meal token
        pos_indices = torch.arange(self.max_meals, device=meal_ids.device)
        pos_enc = self.pos_emb(pos_indices).unsqueeze(0)         # [1, M, embed_dim]
        meal_token_emb = meal_token_emb + pos_enc               # [B*T, M, embed_dim]
        if VERBOSE_LOGGING:
            logging.debug("MealEncoder: Added positional encoding.")

        # Now prepend the start token to every sequence.
        start_token_expanded = self.start_token.expand(B * T, -1, -1)  # [B*T, 1, embed_dim]
        meal_token_emb = torch.cat([start_token_expanded, meal_token_emb], dim=1)  # [B*T, 1+M, embed_dim]
        if VERBOSE_LOGGING:
            logging.debug(f"MealEncoder: After prepending start token, shape: {meal_token_emb.shape}")

        # Adjust padding mask: create a new mask of shape [B*T, 1+M]
        pad_mask = (meal_ids_flat == 0)                         # [B*T, M]
        pad_mask = torch.cat([torch.zeros(B * T, 1, device=pad_mask.device, dtype=torch.bool), pad_mask], dim=1)
        
        meal_attn_out = self.encoder(meal_token_emb, src_key_padding_mask=pad_mask)  # [B*T, 1+M, embed_dim]
        if VERBOSE_LOGGING:
            logging.debug(f"MealEncoder: Transformer encoder output shape: {meal_attn_out.shape}")
        
        # Pool over the tokens (or you can simply take the start token's output).
        # Here we'll take the first token's output as the representation.
        meal_timestep_emb = meal_attn_out[:, 0, :]               # [B*T, embed_dim]
        meal_timestep_emb = meal_timestep_emb.view(B, T, self.embed_dim)
        if VERBOSE_LOGGING:
            logging.debug(f"MealEncoder: Final meal timestep embedding shape: {meal_timestep_emb.shape}")
        return meal_timestep_emb
    
class GlucoseEncoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4, num_layers: int = 1, max_seq_len: int = 100):
        super(GlucoseEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.glucose_proj = nn.Linear(1, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 2, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        
    def forward(self, glucose_seq: torch.Tensor) -> torch.Tensor:
        B, T = glucose_seq.size()
        if VERBOSE_LOGGING:
            logging.debug(f"GlucoseEncoder: Input glucose_seq shape: {glucose_seq.shape}")
        x = glucose_seq.unsqueeze(-1)            # [B, T, 1]
        x = self.glucose_proj(x)                  # [B, T, embed_dim]
        log_tensor_stats("Glucose projection", x)
        
        pos_indices = torch.arange(T, device=glucose_seq.device)
        pos_enc = self.pos_emb(pos_indices).unsqueeze(0)  # [1, T, embed_dim]
        x = x + pos_enc
        log_tensor_stats("Glucose with pos enc", x)
        
        x = self.encoder(x)
        log_tensor_stats("Glucose encoder output", x)
        return x

class MealGlucoseForecastModel(pl.LightningModule):
    def __init__(self, embed_dim=32, num_foods=100, macro_dim=3, max_meals=11, 
                 glucose_seq_len=20, forecast_horizon=4, num_heads=4, enc_layers=1, residual_pred=True):
        super(MealGlucoseForecastModel, self).__init__()
        self.embed_dim = embed_dim
        self.forecast_horizon = forecast_horizon
        self.residual_pred = residual_pred
        
        self.meal_encoder = MealEncoder(embed_dim, num_foods, macro_dim, max_meals, num_heads, num_layers=enc_layers)
        self.glucose_encoder = GlucoseEncoder(embed_dim, num_heads, num_layers=enc_layers, max_seq_len=glucose_seq_len)
        self.future_meal_encoder = MealEncoder(embed_dim, num_foods, macro_dim, max_meals, num_heads, num_layers=enc_layers)
        
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        
        hidden_dim = embed_dim * 2
        self.forecast_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, forecast_horizon)
        )
        
        self.last_attn_weights = None
        # The fixed examples for forecast visualization.
        self.example_forecasts = None

    def forward(self, past_glucose, past_meal_ids, past_meal_macros, future_meal_ids, future_meal_macros, target_scales, return_attn=False):
        logging.debug("MealGlucoseForecastModel: --- Forward Pass Start ---")
        log_tensor_stats("Input past_glucose", past_glucose)
        log_tensor_stats("Input past_meal_ids", past_meal_ids.float())
        log_tensor_stats("Input past_meal_macros", past_meal_macros)
        
        glucose_enc = self.glucose_encoder(past_glucose)  # [B, T, embed_dim]
        past_meal_enc = self.meal_encoder(past_meal_ids, past_meal_macros)  # [B, T, embed_dim]
        
        log_tensor_stats("Glucose encoder final", glucose_enc)
        log_tensor_stats("Past meal encoder final", past_meal_enc)
        
        future_meal_enc = self.future_meal_encoder(future_meal_ids, future_meal_macros)  # [B, forecast_horizon, embed_dim]
        log_tensor_stats("Future meal encoder", future_meal_enc)
        
        # Cross-attention: let glucose queries attend to past meal embeddings
        attn_output, attn_weights = self.cross_attn(query=glucose_enc, key=past_meal_enc, value=past_meal_enc, need_weights=True)
        log_tensor_stats("Attention output", attn_output)
        log_tensor_stats("Attention weights", attn_weights)
        
        combined_glucose = attn_output + glucose_enc  # [B, T, embed_dim]
        log_tensor_stats("Combined glucose", combined_glucose)
        self.last_attn_weights = attn_weights
        
        # Use last timestep of combined past glucose and last timestep of future meal encoding
        final_rep = torch.cat([combined_glucose[:, -1, :], future_meal_enc[:, -1, :]], dim=-1)  # [B, embed_dim*2]
        log_tensor_stats("Final representation", final_rep)
        
        pred_future = self.forecast_mlp(final_rep)  # [B, forecast_horizon]
        log_tensor_stats("Forecast MLP output", pred_future)
        
        # Apply residual addition if necessary
        if self.residual_pred:
            last_val = past_glucose[:, -1].unsqueeze(1)
            pred_future = pred_future + last_val
            log_tensor_stats("Residual forecast", pred_future)
        
        # Un-scale the output predictions using the standard helper
        pred_future = unscale_tensor(pred_future, target_scales)
        log_tensor_stats("Unscaled Forecast", pred_future)
        
        logging.debug("MealGlucoseForecastModel: --- Forward Pass End ---")
        if return_attn:
            return pred_future, past_meal_enc, attn_weights
        return pred_future

    def validation_step(self, batch, batch_idx):
        logging.debug(f"Validation step: batch_idx {batch_idx}")
        past_glucose, past_meal_ids, past_meal_macros, future_meal_ids, future_meal_macros, future_glucose, target_scales = batch
        
        # Adjust target_scales shape if needed.
        if target_scales.dim() > 2:
            target_scales = target_scales.view(target_scales.size(0), -1)
        
        preds = self(past_glucose, past_meal_ids, past_meal_macros, future_meal_ids, future_meal_macros, target_scales)
        
        # -- UN-SCALE THE GROUND-TRUTH TARGET --
        future_glucose_unscaled = future_glucose * target_scales[:, 1].unsqueeze(1) + target_scales[:, 0].unsqueeze(1)
        
        val_loss = F.mse_loss(preds, future_glucose_unscaled)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True)
        logging.debug(f"Validation loss: {val_loss.item()}")
        
        # <-- New addition: Unscale the past glucose values for plotting -->
        past_glucose_unscaled = unscale_tensor(past_glucose, target_scales)

        # Store forecast examples from the first batch to visualize fixed datapoints.
        if batch_idx == 0:
            self.example_forecasts = {
                "past": past_glucose_unscaled.detach().cpu(),  # Now using unscaled past glucose
                "pred": preds.detach().cpu(),     # already unscaled from forward()
                "truth": future_glucose_unscaled.detach().cpu(),
                "future_meal_ids": future_meal_ids.detach().cpu(),
            }
        return val_loss

    def on_validation_epoch_end(self):
        """
        At the end of every validation epoch, log two plots using WandB:
          1. A forecast plot for a few fixed examples, showing:
             - The historical (context) glucose values.
             - The ground truth forecast and predictions.
             - Vertical dashed markers where a meal was consumed.
          2. The cross-attention heatmap.
        """
        import matplotlib.pyplot as plt

        # ----- Plot Forecast Examples -----
        if self.example_forecasts is not None:
            forecasts = self.example_forecasts
            past = forecasts["past"]                # shape: [B, T_context]
            pred = forecasts["pred"]                # shape: [B, forecast_horizon]
            truth = forecasts["truth"]              # shape: [B, forecast_horizon]
            future_meal_ids = forecasts["future_meal_ids"]  # shape: [B, forecast_horizon, max_meals]
            num_examples = min(4, past.size(0))
            fig, axs = plt.subplots(num_examples, 1, figsize=(12, 4 * num_examples))
            if num_examples == 1:
                axs = [axs]
            for i in range(num_examples):
                past_i = past[i].numpy()   # Historical context [T_context,]
                pred_i = pred[i].numpy()   # Forecast prediction [T_forecast,]
                truth_i = truth[i].numpy() # Ground truth for forecast [T_forecast,]
                meals_i = future_meal_ids[i].numpy()  # [T_forecast, max_meals]
                T_context = past_i.shape[0]
                T_forecast = pred_i.shape[0]
                total_t = T_context + T_forecast

                ax = axs[i]
                # Plot historical glucose values.
                ax.plot(range(T_context), past_i, marker='o', label='Historical')
                # Plot ground truth forecast and predictions on the forecast region.
                forecast_timesteps = list(range(T_context, total_t))
                ax.plot(forecast_timesteps, truth_i, marker='o', label='Ground Truth Forecast')
                ax.plot(forecast_timesteps, pred_i, marker='o', label='Prediction Forecast')
                
                # For forecast timesteps, mark meal consumption events.
                meal_label_added = False
                for t in range(T_forecast):
                    if (meals_i[t] != 0).any():
                        x_coord = T_context + t
                        if not meal_label_added:
                            ax.axvline(x=x_coord, color='red', linestyle='--', alpha=0.7, label='Meal Consumption')
                            meal_label_added = True
                        else:
                            ax.axvline(x=x_coord, color='red', linestyle='--', alpha=0.7)
                
                ax.set_xlabel("Timestep")
                ax.set_ylabel("Glucose Level")
                ax.set_title(f"Forecast Example {i}")
                ax.legend(fontsize='small')
            fig.tight_layout()
            self.logger.experiment.log({"forecast_examples": wandb.Image(fig)})
            plt.close(fig)
            # Clear stored examples for the next epoch.
            self.example_forecasts = None

        # ----- Plot the Attention Heatmap -----
        if self.last_attn_weights is not None:
            attn = self.last_attn_weights[0].detach().cpu().numpy()
            fig, ax = plt.subplots(figsize=(8, 6))
            cax = ax.imshow(attn, aspect='auto', cmap='viridis')
            fig.colorbar(cax, ax=ax)
            ax.set_xlabel("Past Meal Timestep")
            ax.set_ylabel("Glucose Timestep")
            ax.set_title("Cross-Attention Weights\n(Glucose Queries vs. Meal Keys)")
            fig.tight_layout()
            self.logger.experiment.log({"attention_weights": wandb.Image(fig)})
            plt.close(fig)

    def training_step(self, batch, batch_idx):
        logging.debug(f"Training step: batch_idx {batch_idx}")
        past_glucose, past_meal_ids, past_meal_macros, future_meal_ids, future_meal_macros, future_glucose, target_scales = batch
        
        # Adjust target_scales shape if needed.
        if target_scales.dim() > 2:
            target_scales = target_scales.view(target_scales.size(0), -1)
        
        preds = self(past_glucose, past_meal_ids, past_meal_macros, future_meal_ids, future_meal_macros, target_scales)
        
        # -- UN-SCALE THE GROUND-TRUTH TARGET --
        future_glucose_unscaled = future_glucose * target_scales[:, 1].unsqueeze(1) + target_scales[:, 0].unsqueeze(1)
        
        loss = F.mse_loss(preds, future_glucose_unscaled)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        logging.debug(f"Training loss: {loss.item()}")
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

# -----------------------------
# Dummy Dataset
# -----------------------------

class DummyMealGlucoseDataset(Dataset):
    def __init__(self, num_sequences=1000, past_seq_length=20, forecast_horizon=4, num_foods=100, max_meals=11, macro_dim=3):
        super().__init__()
        self.num_sequences = num_sequences
        self.past_seq_length = past_seq_length
        self.forecast_horizon = forecast_horizon
        self.max_meals = max_meals
        self.macro_dim = macro_dim
        self.num_foods = num_foods
        self.data = []
        for _ in range(num_sequences):
            # Past glucose sequence (e.g., 20 timesteps)
            past_glucose = torch.rand(past_seq_length)
            # Future glucose sequence (forecast horizon, e.g., 4 timesteps)
            future_glucose = torch.rand(forecast_horizon)
            
            # Past meals for each timestep in past sequence
            past_meal_ids = torch.zeros(past_seq_length, max_meals, dtype=torch.long)
            past_meal_macros = torch.zeros(past_seq_length, max_meals, macro_dim, dtype=torch.float)
            for t in range(past_seq_length):
                num_meals = random.randint(0, 3)
                num_meals = min(num_meals, max_meals)
                for k in range(num_meals):
                    food_id = random.randint(1, num_foods - 1)  # 0 is pad
                    macros = torch.rand(macro_dim)
                    past_meal_ids[t, k] = food_id
                    past_meal_macros[t, k] = macros
            
            # Future meals for each timestep in forecast horizon
            future_meal_ids = torch.zeros(forecast_horizon, max_meals, dtype=torch.long)
            future_meal_macros = torch.zeros(forecast_horizon, max_meals, macro_dim, dtype=torch.float)
            for t in range(forecast_horizon):
                num_meals = random.randint(0, 3)
                num_meals = min(num_meals, max_meals)
                for k in range(num_meals):
                    food_id = random.randint(1, num_foods - 1)
                    macros = torch.rand(macro_dim)
                    future_meal_ids[t, k] = food_id
                    future_meal_macros[t, k] = macros
            
            self.data.append((past_glucose, past_meal_ids, past_meal_macros, future_meal_ids, future_meal_macros, future_glucose))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# -----------------------------
# Main Block
# -----------------------------

if __name__ == '__main__':
    
    
    from dataset import PPGRTimeSeriesDataset, PPGRToMealGlucoseWrapper, split_timeseries_df_based_on_food_intake_rows
    from utils import load_dataframe, enforce_column_types, setup_scalers_and_encoders
    
    logging.info("Starting main block.")
    dataset_version = "v0.4"
    debug_mode = True

    min_encoder_length = 8 * 4 # 8 hours with 4 timepoints per hour
    prediction_length = 2 * 4 # 2 hours with 4 timepoints per hour
    
    validation_percentage = 0.2
    test_percentage = 0.2


    is_food_anchored = True # When False, its the standard sliding window timeseries, but when True, every timeseries will have a food intake row at the last item of the context window
    sliding_window_stride = None # This has to change everytime is_food_anchored is changed
    
    use_meal_level_food_covariates = True
    
    use_microbiome_embeddings = True
    
    # Unique Grouping Column
    group_by_columns = ["timeseries_block_id"]

    # User 
    user_static_categoricals = ["user_id", "user__edu_degree", "user__income", "user__household_desc", "user__job_status", "user__smoking", "user__health_state", "user__physical_activities_frequency"]
    user_static_reals = ["user__age", "user__weight", "user__height", "user__bmi", "user__general_hunger_level", "user__morning_hunger_level", "user__mid_hunger_level", "user__evening_hunger_level"]

    # Food Covariates
    food_categoricals = []
    if use_meal_level_food_covariates:
        food_categoricals = ['food__food_group_cname', 'food_id']
    else:
        food_categoricals = [   'food__vegetables_fruits', 'food__grains_potatoes_pulses', 'food__unclassified',
                                'food__non_alcoholic_beverages', 'food__dairy_products_meat_fish_eggs_tofu',
                                'food__sweets_salty_snacks_alcohol', 'food__oils_fats_nuts'] 
    
    food_reals = ['food__eaten_quantity_in_gram', 'food__energy_kcal_eaten',
        'food__carb_eaten', 'food__fat_eaten', 'food__protein_eaten',
        'food__fiber_eaten', 'food__alcohol_eaten']

    # Temporal Covariates
    temporal_categoricals = ["loc_eaten_dow", "loc_eaten_dow_type", "loc_eaten_season"]
    temporal_reals = ["loc_eaten_hour"]

    # Targets
    targets = ["val"]

    main_df_scaled_all_categorical_columns = user_static_categoricals + food_categoricals + temporal_categoricals
    main_df_scaled_all_real_columns = user_static_reals + food_reals + temporal_reals + targets
    
    
    # Load the data frames
    ppgr_df, users_demographics_df, dishes_df, microbiome_embeddings_df = load_dataframe(dataset_version, debug_mode)

    # Split the data frames into training, validation and test sets
    training_df, validation_df, test_df = split_timeseries_df_based_on_food_intake_rows(ppgr_df, validation_percentage=validation_percentage, test_percentage=test_percentage)
    
    # Validate the data frames
    ppgr_df, users_demographics_df, dishes_df = enforce_column_types(  ppgr_df, 
                                                            users_demographics_df, 
                                                            dishes_df,
                                                            main_df_scaled_all_categorical_columns,
                                                            main_df_scaled_all_real_columns)    
    
    
    # Setup the scalers and encoders
    categorical_encoders, continuous_scalers = setup_scalers_and_encoders(
        ppgr_df = ppgr_df,
        training_df = training_df,
        users_demographics_df = users_demographics_df,
        dishes_df=dishes_df,
        categorical_columns = main_df_scaled_all_categorical_columns,
        real_columns = main_df_scaled_all_real_columns,
        use_meal_level_food_covariates = use_meal_level_food_covariates # This determines which data to fit the encoders on
    ) # Note: the encoders are fit on the full ppgr_df, and the scalers are fit on the training_df

    # Create the training dataset
    
    def _create_dataset(df: pd.DataFrame):
        dataset = PPGRTimeSeriesDataset(ppgr_df = df, 
                                            user_demographics_df = users_demographics_df,
                                            dishes_df = dishes_df,
                                            time_idx = "read_at",
                                            target_columns = ["val"],                                                                                    
                                            group_by_columns = ["timeseries_block_id"],

                                            is_food_anchored = is_food_anchored, # When False, its the standard sliding window timeseries, but when True, every timeseries will have a food intake row at the last item of the context window
                                            sliding_window_stride = sliding_window_stride, # This has to change everytime is_food_anchored is changed

                                            min_encoder_length = min_encoder_length, # 8 hours with 4 timepoints per hour
                                            prediction_length = prediction_length, # 2 hours with 4 timepoints per hour
                                            
                                            use_food_covariates_from_prediction_window = True,
                                            
                                            use_meal_level_food_covariates = use_meal_level_food_covariates, # This uses the granular meal level food covariates instead of the food item level covariates
                                            
                                            use_microbiome_embeddings = use_microbiome_embeddings,
                                            microbiome_embeddings_df = microbiome_embeddings_df,
                                            
                                            temporal_categoricals = temporal_categoricals,
                                            temporal_reals = temporal_reals,

                                            user_static_categoricals = user_static_categoricals,
                                            user_static_reals = user_static_reals,
                                            
                                            food_categoricals = food_categoricals,
                                            food_reals = food_reals,
                                            
                                            categorical_encoders = categorical_encoders,
                                            continuous_scalers = continuous_scalers,
                                            )
        return dataset

    training_dataset = _create_dataset(training_df)
    validation_dataset = _create_dataset(validation_df)
    test_dataset = _create_dataset(test_df)
    
    wrapped_training_dataset = PPGRToMealGlucoseWrapper(training_dataset)
    wrapped_validation_dataset = PPGRToMealGlucoseWrapper(validation_dataset)
    wrapped_test_dataset = PPGRToMealGlucoseWrapper(test_dataset)
    
    # -----------------------------------------------------------------------------
    # Create DataLoaders
    # -----------------------------------------------------------------------------
    batch_size = 32
    train_loader = DataLoader(wrapped_training_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(wrapped_validation_dataset, batch_size=batch_size)

    # -----------------------------------------------------------------------------
    # Initialize the model using parameters from the wrapped dataset.
    # We'll assume:
    #   - The past sequence length equals min_encoder_length.
    #   - The forecast horizon equals prediction_length.
    #   - macro_dim is 3 (since we are extracting columns 2:5 from the food real features).
    # -----------------------------------------------------------------------------
    glucose_seq_len = min_encoder_length  # e.g. 32
    forecast_horizon = prediction_length    # e.g. 8


    embedding_dim = 32
    macro_dim = 3
    max_meals = wrapped_training_dataset.max_meals
    forecast_horizon = prediction_length
    glucose_seq_len = min_encoder_length

    num_heads = 4
    enc_layers = 1
    residual_pred = False

    model = MealGlucoseForecastModel(
        embed_dim=embedding_dim,
        num_foods=wrapped_training_dataset.num_foods,  # estimated from the PPGR categorical encoder
        macro_dim=macro_dim,                                    # as defined in our wrapper extraction
        max_meals=max_meals,   # from the PPGR dataset attribute
        glucose_seq_len=glucose_seq_len,
        forecast_horizon=forecast_horizon,
        num_heads=num_heads,
        enc_layers=enc_layers,
        residual_pred=residual_pred
    )

    # -----------------------------------------------------------------------------
    # Set up WandB Logger (with hard-coded project values) and Train the model
    # -----------------------------------------------------------------------------
    wandb_logger = WandbLogger(
        project="meal-representations-learning-v0",
        name="MealGlucoseForecastModel_Run",
        config={
            "embed_dim": embedding_dim,
            "num_foods": wrapped_training_dataset.num_foods,
            "macro_dim": macro_dim,
            "max_meals": max_meals,
            "glucose_seq_len": glucose_seq_len,
            "forecast_horizon": forecast_horizon,
            "batch_size": batch_size,
            "verbose_logging": VERBOSE_LOGGING,
        },
        log_model=True
    )
    logging.info("WandB Logger initialized with project 'HardCodedPPGRProject'.")
    
    trainer = pl.Trainer(max_epochs=10, enable_checkpointing=False, logger=wandb_logger)
    logging.info("Starting training.")
    trainer.fit(model, train_loader, val_loader)
    logging.info("Training complete.")

    # -----------------------------------------------------------------------------
    # Evaluate on one batch from the validation DataLoader
    # -----------------------------------------------------------------------------
    device = torch.device('cpu')
    model.to(device)
    model.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))
        # Each batch is a 6-tuple:
        past_glucose, past_meal_ids, past_meal_macros, future_meal_ids, future_meal_macros, future_glucose, target_scales = [
            x.to(device) for x in batch
        ]
        
        # Shapes (example):
        # past_glucose: [B, T_enc]
        # past_meal_ids: [B, T_enc, M]
        # past_meal_macros: [B, T_enc, M, D]
        # future_meal_ids: [B, T_pred, M]
        # future_meal_macros: [B, T_pred, M, D]
        # future_glucose: [B, T_pred]
        
        preds, past_meal_embeds, attn_weights = model(
            past_glucose,
            past_meal_ids,
            past_meal_macros,
            future_meal_ids,
            future_meal_macros,
            target_scales,
            return_attn=True
        )

    logging.info("Predicted future glucose (first sample): %s", preds[0].cpu().numpy())
    logging.info("Actual future glucose (first sample):   %s", future_glucose[0].cpu().numpy())
    logging.info("Past meal embedding shape: %s", past_meal_embeds.shape)  # expected: [B, T_enc, embed_dim]
    logging.info("Cross-attention weight shape: %s", attn_weights.shape)    # expected: [B, T_enc, T_enc]
    logging.info("Attention weights (first sample): %s", attn_weights[0])

    # Suppose past_glucose is of shape [B, T] (normalized)
    past_glucose_unscaled = unscale_tensor(past_glucose, target_scales)
    # Now past_glucose_unscaled is in the original (physical) scale
