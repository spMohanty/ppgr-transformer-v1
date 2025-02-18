import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
import random

# -----------------------------
# Model Components
# -----------------------------

class MealEncoder(nn.Module):
    def __init__(self, embed_dim: int, num_foods: int, macro_dim: int, max_meals: int = 11, num_heads: int = 4, num_layers: int = 1):
        """
        Transformer-based encoder for meals at each timestep.
        Each timestep contains up to max_meals items, each with a food ID and macronutrient values.
        """
        super(MealEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.max_meals = max_meals
        
        self.food_emb = nn.Embedding(num_foods, embed_dim, padding_idx=0)
        self.macro_proj = nn.Linear(macro_dim, embed_dim, bias=False)
        self.pos_emb = nn.Embedding(max_meals, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 2, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, meal_ids: torch.LongTensor, meal_macros: torch.Tensor) -> torch.Tensor:
        """
        Args:
            meal_ids: Tensor of shape [B, T, max_meals] (food IDs, 0 = pad)
            meal_macros: Tensor of shape [B, T, max_meals, macro_dim]
        Returns:
            meal_timestep_emb: Tensor of shape [B, T, embed_dim], one embedding per timestep.
        """
        B, T, M = meal_ids.size()
        meal_ids_flat = meal_ids.view(B * T, M)
        meal_macros_flat = meal_macros.view(B * T, M, -1)
        meal_id_emb = self.food_emb(meal_ids_flat)              # [B*T, M, embed_dim]
        meal_macro_emb = self.macro_proj(meal_macros_flat)       # [B*T, M, embed_dim]
        meal_token_emb = meal_id_emb + meal_macro_emb            # [B*T, M, embed_dim]
        
        pos_indices = torch.arange(self.max_meals, device=meal_ids.device)
        pos_enc = self.pos_emb(pos_indices).unsqueeze(0)         # [1, M, embed_dim]
        meal_token_emb = meal_token_emb + pos_enc
        
        pad_mask = (meal_ids_flat == 0)                          # [B*T, M]
        meal_attn_out = self.encoder(meal_token_emb, src_key_padding_mask=pad_mask)  # [B*T, M, embed_dim]
        
        # Average pooling (ignoring pads) to get a single vector per timestep
        mask_inv = (~pad_mask).unsqueeze(-1).float()            # [B*T, M, 1]
        summed = (meal_attn_out * mask_inv).sum(dim=1)           # [B*T, embed_dim]
        count = mask_inv.sum(dim=1).clamp(min=1)                 # [B*T, 1]
        meal_timestep_emb = summed / count                     # [B*T, embed_dim]
        meal_timestep_emb = meal_timestep_emb.view(B, T, self.embed_dim)
        return meal_timestep_emb

class GlucoseEncoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4, num_layers: int = 1, max_seq_len: int = 100):
        """
        Transformer-based encoder for past glucose readings.
        """
        super(GlucoseEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.glucose_proj = nn.Linear(1, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 2, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        
    def forward(self, glucose_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            glucose_seq: Tensor of shape [B, T] (past glucose values)
        Returns:
            x: Tensor of shape [B, T, embed_dim]
        """
        B, T = glucose_seq.size()
        x = glucose_seq.unsqueeze(-1)            # [B, T, 1]
        x = self.glucose_proj(x)                 # [B, T, embed_dim]
        pos_indices = torch.arange(T, device=glucose_seq.device)
        pos_enc = self.pos_emb(pos_indices).unsqueeze(0)  # [1, T, embed_dim]
        x = x + pos_enc
        x = self.encoder(x)
        return x

class MealGlucoseForecastModel(pl.LightningModule):
    def __init__(self, embed_dim=32, num_foods=100, macro_dim=3, max_meals=11, 
                 glucose_seq_len=20, forecast_horizon=4, num_heads=4, enc_layers=1, residual_pred=True):
        """
        PyTorch Lightning Module that forecasts future glucose values.
        It uses:
         - A meal encoder to encode past meals,
         - A glucose encoder to encode past glucose,
         - A separate meal encoder for future meals as a covariate,
         - Cross-attention to fuse past meals and glucose,
         - And an MLP that predicts the future glucose sequence.
        """
        super(MealGlucoseForecastModel, self).__init__()
        self.embed_dim = embed_dim
        self.forecast_horizon = forecast_horizon
        self.residual_pred = residual_pred
        
        # Encoders for past meals, past glucose, and future meals
        self.meal_encoder = MealEncoder(embed_dim, num_foods, macro_dim, max_meals, num_heads, num_layers=enc_layers)
        self.glucose_encoder = GlucoseEncoder(embed_dim, num_heads, num_layers=enc_layers, max_seq_len=glucose_seq_len)
        self.future_meal_encoder = MealEncoder(embed_dim, num_foods, macro_dim, max_meals, num_heads, num_layers=enc_layers)
        
        # Cross-attention: let past glucose attend to past meal embeddings
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        
        # Forecasting MLP: takes concatenated representation of the last past glucose and future meal embedding
        hidden_dim = embed_dim * 2
        self.forecast_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, forecast_horizon)
        )
        
        # Placeholder for inspection of attention weights
        self.last_attn_weights = None

    def forward(self, past_glucose, past_meal_ids, past_meal_macros, future_meal_ids, future_meal_macros, return_attn=False):
        # Encode past glucose and meals
        glucose_enc = self.glucose_encoder(past_glucose)  # [B, T, embed_dim]
        past_meal_enc = self.meal_encoder(past_meal_ids, past_meal_macros)  # [B, T, embed_dim]
        # Encode future meals (for the prediction window)
        future_meal_enc = self.future_meal_encoder(future_meal_ids, future_meal_macros)  # [B, forecast_horizon, embed_dim]
        
        # Cross-attention: let glucose query attend to past meal embeddings
        attn_output, attn_weights = self.cross_attn(query=glucose_enc, key=past_meal_enc, value=past_meal_enc, need_weights=True)
        combined_glucose = attn_output + glucose_enc  # [B, T, embed_dim]
        self.last_attn_weights = attn_weights
        
        # Use the last timestep from combined past glucose and the last timestep from future meals
        final_rep = torch.cat([combined_glucose[:, -1, :], future_meal_enc[:, -1, :]], dim=-1)  # [B, embed_dim*2]
        pred_future = self.forecast_mlp(final_rep)  # [B, forecast_horizon]
        if self.residual_pred:
            last_val = past_glucose[:, -1].unsqueeze(1)  # [B, 1]
            pred_future = pred_future + last_val
        if return_attn:
            return pred_future, past_meal_enc, attn_weights
        return pred_future

    def training_step(self, batch, batch_idx):
        past_glucose, past_meal_ids, past_meal_macros, future_meal_ids, future_meal_macros, future_glucose = batch
        preds = self(past_glucose, past_meal_ids, past_meal_macros, future_meal_ids, future_meal_macros)
        loss = F.mse_loss(preds, future_glucose)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        past_glucose, past_meal_ids, past_meal_macros, future_meal_ids, future_meal_macros, future_glucose = batch
        preds = self(past_glucose, past_meal_ids, past_meal_macros, future_meal_ids, future_meal_macros)
        val_loss = F.mse_loss(preds, future_glucose)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True)
        return val_loss

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
    # Parameters
    past_seq_length = 20
    forecast_horizon = 4
    batch_size = 32
    num_sequences = 200

    # Create dataset and dataloaders
    dataset = DummyMealGlucoseDataset(
        num_sequences=num_sequences,
        past_seq_length=past_seq_length,
        forecast_horizon=forecast_horizon,
        num_foods=100,
        max_meals=11,
        macro_dim=3
    )
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Initialize model
    model = MealGlucoseForecastModel(
        embed_dim=32,
        num_foods=100,
        macro_dim=3,
        max_meals=11,
        glucose_seq_len=past_seq_length,
        forecast_horizon=forecast_horizon,
        num_heads=4,
        enc_layers=1,
        residual_pred=True
    )

    # Train the model
    trainer = pl.Trainer(max_epochs=5, enable_checkpointing=False, logger=False)
    trainer.fit(model, train_loader, val_loader)

    # Evaluate on one batch from validation
    device = torch.device('cpu')
    model.to(device)
    model.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))
        past_glucose, past_meal_ids, past_meal_macros, future_meal_ids, future_meal_macros, future_glucose = [x.to(device) for x in batch]
        
        """
        past_glucose: [B, T]
        past_meal_ids: [B, T, M]
        past_meal_macros: [B, T, M, D]
        future_meal_ids: [B, F, M]
        future_meal_macros: [B, F, M, D]
        future_glucose: [B, F]
        """
        
        breakpoint()
        
        preds, past_meal_embeds, attn_weights = model(
            past_glucose,
            past_meal_ids,
            past_meal_macros,
            future_meal_ids,
            future_meal_macros,
            return_attn=True
        )

    print("Predicted future glucose (first sample):", preds[0].cpu().numpy())
    print("Actual future glucose (first sample):   ", future_glucose[0].cpu().numpy())
    print("Past meal embedding shape:", past_meal_embeds.shape)  # [B, past_seq_length, embed_dim]
    print("Cross-attention weight shape:", attn_weights.shape)    # [B, past_seq_length, past_seq_length]
    print("Attention weights (first sample, first 5x5 block):")
    # print(attn_weights[0][:5, :5].cpu().numpy())
    print(attn_weights[0])
    breakpoint()
