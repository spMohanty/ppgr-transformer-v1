import pandas as pd
import uuid
from loguru import logger
from sklearn.preprocessing import StandardScaler
from pytorch_forecasting.data.encoders import (
    EncoderNormalizer,
    GroupNormalizer,
    MultiNormalizer,
    NaNLabelEncoder,
    TorchNormalizer,
)

import torch
from torch.nn.utils.rnn import pad_sequence
def create_slice(df: pd.DataFrame, slice_start: int, slice_end: int) -> pd.DataFrame:
    """
    Create a time-series slice with a unique cluster id and a time index.
    """
    df_slice = df.iloc[slice_start:slice_end].copy()
    df_slice["time_series_cluster_id"] = str(uuid.uuid4())
    df_slice["time_idx"] = list(range(len(df_slice)))
    return df_slice


def load_dataframe(dataset_version: str, debug_mode: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load processed CSV files for the PPGR, user demographics data, microbiome embeddings, granular food intake data.
    User IDs are converted to strings, and ppgr_df is sorted by user_id, timeseries_block_id, and read_at.
    """
    logger.info(f"Loading dataframe for dataset version {dataset_version}")
    prefix = "debug-" if debug_mode else ""
    subdir = "debug/" if debug_mode else ""

    ppgr_path = (
        f"data/processed/{dataset_version}/{subdir}"
        f"{prefix}fay-ppgr-processed-and-aggregated-{dataset_version}.csv"
    )
    demographics_path = (
        f"data/processed/{dataset_version}/{subdir}"
        f"{prefix}users-demographics-data-{dataset_version}.csv"
    )
    microbiome_embeddings_path = (
        f"data/processed/{dataset_version}/{subdir}"
        f"{prefix}microbiome-data-{dataset_version}.csv"
    )
    dishes_path = (
        f"data/processed/{dataset_version}/{subdir}"
        f"{prefix}dishes-data-{dataset_version}.csv"
    )
    
    ppgr_df = pd.read_csv(ppgr_path)
    users_demographics_df = pd.read_csv(demographics_path)
    microbiome_embeddings_df = pd.read_csv(microbiome_embeddings_path).set_index("user_id")
    dishes_df = pd.read_csv(dishes_path).reset_index(drop=True)
    
    
    # Convert user_id to string in both dataframes
    ppgr_df["user_id"] = ppgr_df["user_id"].astype(str)
    users_demographics_df["user_id"] = users_demographics_df["user_id"].astype(str)
    
    
    ppgr_df.sort_values(by=["user_id", "timeseries_block_id", "read_at"], inplace=True)
    logger.info(f"Loaded dataframe with {len(ppgr_df)} rows from {ppgr_path}")
    
    return ppgr_df, users_demographics_df, dishes_df, microbiome_embeddings_df


def enforce_column_types(
    ppgr_df: pd.DataFrame,
    users_df: pd.DataFrame,
    dishes_df: pd.DataFrame,
    categorical_columns: list[str],
    real_columns: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ensure that each column in categorical_columns is cast to str and each column in real_columns to float.
    If a column exists in both dataframes, it will be cast in both.
    """
    logger.info("Enforcing column types")
    for col in categorical_columns:
        casted = False
        if col in ppgr_df.columns:
            ppgr_df[col] = ppgr_df[col].astype(str)
            casted = True
        if col in users_df.columns:
            users_df[col] = users_df[col].astype(str)
            casted = True
        if col in dishes_df.columns:
            dishes_df[col] = dishes_df[col].astype(str)
            casted = True
        if not casted:
            raise ValueError(f"Categorical column '{col}' not found in either dataframe.")

    for col in real_columns:
        casted = False
        if col in ppgr_df.columns:
            ppgr_df[col] = ppgr_df[col].astype(float)
            casted = True
        if col in users_df.columns:
            users_df[col] = users_df[col].astype(float)
            casted = True
        if col in dishes_df.columns:
            dishes_df[col] = dishes_df[col].astype(float)
            casted = True
        if not casted:
            raise ValueError(f"Real column '{col}' not found in either dataframe.")
    
    return ppgr_df, users_df, dishes_df

def get_all_dishes_in_df(df: pd.DataFrame) -> list[str]:
    """
    Get all the dishes in the dataframe.
    """
    logger.info("Getting all dishes in dataframe")
    dish_candidates = df["dish_id"].dropna().unique()
    dish_ids = []
    for dish_candidate_row in dish_candidates:
        dish_candidate = dish_candidate_row.split("||")
        for dish_candidate_item in dish_candidate:
            dish_candidate_item = int(dish_candidate_item)
            if dish_candidate_item > 0: # we use 0.0 as a placeholder for missing values
                dish_ids.append(dish_candidate_item)
                
    return dish_ids

def setup_scalers_and_encoders(
    ppgr_df: pd.DataFrame,
    training_df: pd.DataFrame,
    users_demographics_df: pd.DataFrame,
    dishes_df: pd.DataFrame,
    categorical_columns: list[str],
    real_columns: list[str],
    use_meal_level_food_covariates: bool = False
) -> tuple[dict[str, NaNLabelEncoder], dict[str, StandardScaler]]:
    """
    Setup and fit categorical encoders and continuous scalers using the training data.
    The categorical encoders (NaNLabelEncoder) are fit on the full dataset while the continuous scalers 
    (StandardScaler) are fit only on the training set to avoid data leakage.
    """
    logger.info("Setting up scalers and encoders")
    # Initialize and fit categorical encoders
    categorical_encoders: dict[str, NaNLabelEncoder] = {}
        
    for col in categorical_columns:
        logger.info(f"Setting up categorical encoder for column {col}")
        encoder = NaNLabelEncoder(add_nan=True, warn=True)
        if col in ppgr_df.columns:
            # In case of food columns, check if we want to use the granular food data or the aggregated food data
            if col.startswith("food__") and use_meal_level_food_covariates:
                encoder.fit(dishes_df[col])
            else:
                encoder.fit(ppgr_df[col])
                
        elif col in users_demographics_df.columns:
            encoder.fit(users_demographics_df[col])
        elif col in dishes_df.columns: # fallback to the dishes dataframe if also not present in ppgr_df
            encoder.fit(dishes_df[col])
        else:
            raise ValueError(f"Categorical column '{col}' not found in either dataframe.")
        categorical_encoders[col] = encoder
        
    # Initialize and fit continuous scalers on the training set only
    
    ## Calculate all the dishes in the training set (for fitting scalers on the granular food data)
    if use_meal_level_food_covariates:
        dish_ids_from_training_set = get_all_dishes_in_df(training_df)
        dishes_training_df = dishes_df[dishes_df["dish_id"].isin(dish_ids_from_training_set)]
    
    continuous_scalers: dict[str, StandardScaler] = {}
    for col in real_columns:
        logger.info(f"Setting up continuous scaler for column {col}")
        if col in training_df.columns:
            # In case of food columns, check if we want to use the granular food data or the aggregated food data
            if col.startswith("food__") and use_meal_level_food_covariates:
                values = dishes_training_df[col].to_numpy().reshape(-1, 1)
            else:
                values = training_df[col].to_numpy().reshape(-1, 1)
        elif col in users_demographics_df.columns:
            values = users_demographics_df[col].to_numpy().reshape(-1, 1)
        else:
            raise ValueError(f"Real column '{col}' not found in training or demographics dataframe.")
        
        scaler = StandardScaler().fit(values)
        logger.info(f"Fitted scaler for '{col}': mean={scaler.mean_}, scale={scaler.scale_}")
        continuous_scalers[col] = scaler

    return categorical_encoders, continuous_scalers


def ppgr_collate_fn(batch):
    """
    Custom collate function to pad variable-length sequences in the batch.
    
    This function assumes that each item in the batch is a dictionary with keys
    like 'x_temporal_cat', 'y_temporal_cat', etc. It pads those tensors along
    the time dimension if their first dimension (sequence length) is variable.
    Other keys (like metadata or static tensors) are simply collated into a list
    or stacked if they have consistent shapes.
    """
    collated = {}
            
            
    batch_size = len(batch)
    
    # Get lengths of the individual timeseries in the batch
    encoder_lengths = torch.tensor([item["encoder_length"] for item in batch])
    prediction_lengths = torch.tensor([item["prediction_length"] for item in batch])
            
    # Calculate the maximum length of the timeseries in the batch
    max_encoder_len = max(encoder_lengths)
    max_prediction_len = max(prediction_lengths)

    if batch[0]["metadata"]["use_meal_level_food_covariates"]:
        # Get the number of dishes recorded for each timestep in the batch element
        encoder_dish_tensors_recorded = torch.stack([item["x_dish_tensors_recorded"] for item in batch])
        prediction_dish_tensors_recorded = torch.stack([item["y_dish_tensors_recorded"] for item in batch])
        
        max_encoder_dish_tensors_recorded = torch.max(encoder_dish_tensors_recorded)
        max_prediction_dish_tensors_recorded = torch.max(prediction_dish_tensors_recorded)
    
    MEAL_PADDING_VALUE = 0 # Padding the meals with 0 for consistency
    TEMPORAL_PADDING_VALUE = 1991 # as we are not supporting variable encoder and prediction lengths, we should not see this value anywhere
    
    # Create encoder mask (left padding)
    # Shape: [batch_size, max_encoder_len]
    encoder_mask = torch.arange(max_encoder_len).unsqueeze(0) >= (max_encoder_len - encoder_lengths.unsqueeze(1))
    
    # Create prediction mask (right padding)
    # Shape: [batch_size, max_prediction_len]
    prediction_mask = torch.arange(max_prediction_len).unsqueeze(0) < prediction_lengths.unsqueeze(1)
    
    
    for key in batch[0]:
        if key == "metadata":
            collated[key] = batch[0][key] # the metadata is the same for all items in the batch
            continue
        
        if isinstance(batch[0][key], torch.Tensor):            
            if key.startswith("x_"):
                
                # If meal level data is provided, first pad them to the max number of dishes recorded for this timestep
                if batch[0][key].is_nested:
                    # Get the max number of dishes recorded for this timestep
                    # All this shenenigans only to handle the granual meal level data                    
                    if key == "x_food_cat":
                        num_features = len(batch[0]["metadata"]["food_categoricals"])
                    elif key == "x_food_real":
                        num_features = len(batch[0]["metadata"]["food_reals"])
                    else:
                        raise ValueError(f"Key {key} not supported for meal level padding.")
                    
                    batch_wise_padded_tensors = []
                    for _idx in range(batch_size):
                        # Padd the meal data for this batch (while considering the batch wise max number of dishes recorded)
                        
                        # NOTE: Variable encoder lengths are not supported yet !!
                        padded_tensor = torch.nested.to_padded_tensor(
                            batch[_idx][key],
                            padding=MEAL_PADDING_VALUE,
                            output_size = (max_encoder_len, max_encoder_dish_tensors_recorded, num_features)
                        ) # T, num_dishes, num_features
                        batch_wise_padded_tensors.append(padded_tensor)
                        
                    collated[key] = torch.stack(batch_wise_padded_tensors)
                else:
                    # These are the encoder variables (by naming convention)    
                    collated[key] = pad_sequence(
                        [batch[_idx][key] for _idx in range(len(batch))], batch_first=True, padding_side="left", padding_value=TEMPORAL_PADDING_VALUE)                    
            elif key.startswith("y_"):
                if batch[0][key].is_nested:
                    # Get the max number of dishes recorded for this timestep
                    # All this shenenigans only to handle the granual meal level data                    
                    if key == "y_food_cat":
                        num_features = len(batch[0]["metadata"]["food_categoricals"])
                    elif key == "y_food_real":
                        num_features = len(batch[0]["metadata"]["food_reals"])
                    else:
                        raise ValueError(f"Key {key} not supported for meal level padding.")
                    
                    batch_wise_padded_tensors = []
                    for _idx in range(batch_size):
                        # Padd the meal data for this batch (while considering the batch wise max number of dishes recorded)
                        
                        # NOTE: Variable encoder lengths are not supported yet !!
                        padded_tensor = torch.nested.to_padded_tensor(
                            batch[_idx][key],
                            padding=MEAL_PADDING_VALUE,
                            output_size = (max_prediction_len, max_prediction_dish_tensors_recorded, num_features)
                        ) # T, num_dishes, num_features
                        batch_wise_padded_tensors.append(padded_tensor)
                        
                    collated[key] = torch.stack(batch_wise_padded_tensors)
                else:
                    # These are the prediction window variables (by naming convention)
                    # NOTE: This next step is redundant, as we are currently not supporting variable encoder and prediction lengths
                    collated[key] = pad_sequence(
                        [batch[_idx][key] for _idx in range(len(batch))], batch_first=True, padding_side="right", padding_value=TEMPORAL_PADDING_VALUE)
            else:
                # These are the other variables, that should not need padding by default and a simple stacking should do the trick
                collated[key] = torch.stack([batch[_idx][key] for _idx in range(len(batch))])
        else:
            # raise Error unless y_food_cat or y_food_real, as they might be suppressed 
            # when future food information is not allowed
            if key not in ["y_food_cat", "y_food_real", "x_dish_tensors_recorded", "y_dish_tensors_recorded"]:
                raise ValueError(f"Key {key} is not a tensor")
    
    # Add masks to the collated dictionary
    collated["encoder_mask"] = encoder_mask
    collated["prediction_mask"] = prediction_mask
    
    # Add encoder length and prediction length to the collated dictionary
    collated["encoder_length"] = encoder_lengths
    collated["prediction_length"] = prediction_lengths
    
    return collated

def scale_tensor(tensor, scales):
    """
    Scale a tensor using provided scale factors.
    
    Args:
        tensor (torch.Tensor): The input tensor (e.g., in original scale).
        scales (torch.Tensor): Tensor of scale factors. Assumed to have shape 
            [B, 2] or extra dims that can be collapsed. scales[:, 0] is the mean,
            scales[:, 1] is the std.
            
    Returns:
        torch.Tensor: The normalized tensor.
    
    Normalization formula: normalized = (original - mean) / std
    """
    if scales.dim() > 2:
        scales = scales.view(scales.size(0), -1)
    mean = scales[:, 0].unsqueeze(1)
    std = scales[:, 1].unsqueeze(1)
    return (tensor - mean) / std


def unscale_tensor(tensor, scales):
    """
    Unscale a tensor using provided scale factors.
    
    Args:
        tensor (torch.Tensor): The normalized tensor.
        scales (torch.Tensor): Tensor of scale factors. Assumed to have shape 
            [B, 2] or extra dims that can be collapsed. scales[:, 0] is the mean,
            scales[:, 1] is the std.
            
    Returns:
        torch.Tensor: The tensor transformed back to the original scale.
    
    Unscaling formula: original = normalized * std + mean
    """
    if scales.dim() > 2:
        scales = scales.view(scales.size(0), -1)
    mean = scales[:, 0].unsqueeze(1)
    std = scales[:, 1].unsqueeze(1)
    return tensor * std + mean 