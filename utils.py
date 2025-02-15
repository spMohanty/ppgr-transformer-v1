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
    Load processed CSV files for the PPGR and user demographics data.
    User IDs are converted to strings, and ppgr_df is sorted by user_id, timeseries_block_id, and read_at.
    """
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
    
    ppgr_df = pd.read_csv(ppgr_path)
    users_demographics_df = pd.read_csv(demographics_path)
    
    # Convert user_id to string in both dataframes
    ppgr_df["user_id"] = ppgr_df["user_id"].astype(str)
    users_demographics_df["user_id"] = users_demographics_df["user_id"].astype(str)
    
    microbiome_embeddings_df = pd.read_csv(microbiome_embeddings_path).set_index("user_id")
    
    ppgr_df.sort_values(by=["user_id", "timeseries_block_id", "read_at"], inplace=True)
    logger.info(f"Loaded dataframe with {len(ppgr_df)} rows from {ppgr_path}")
    
    return ppgr_df, users_demographics_df, microbiome_embeddings_df


def enforce_column_types(
    ppgr_df: pd.DataFrame,
    users_df: pd.DataFrame,
    categorical_columns: list[str],
    real_columns: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ensure that each column in categorical_columns is cast to str and each column in real_columns to float.
    If a column exists in both dataframes, it will be cast in both.
    """
    for col in categorical_columns:
        casted = False
        if col in ppgr_df.columns:
            ppgr_df[col] = ppgr_df[col].astype(str)
            casted = True
        if col in users_df.columns:
            users_df[col] = users_df[col].astype(str)
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
        if not casted:
            raise ValueError(f"Real column '{col}' not found in either dataframe.")
    
    return ppgr_df, users_df


def setup_scalers_and_encoders(
    ppgr_df: pd.DataFrame,
    training_df: pd.DataFrame,
    users_demographics_df: pd.DataFrame,
    categorical_columns: list[str],
    real_columns: list[str]
) -> tuple[dict[str, NaNLabelEncoder], dict[str, StandardScaler]]:
    """
    Setup and fit categorical encoders and continuous scalers using the training data.
    The categorical encoders (NaNLabelEncoder) are fit on the full dataset while the continuous scalers 
    (StandardScaler) are fit only on the training set to avoid data leakage.
    """
    # Initialize and fit categorical encoders
    categorical_encoders: dict[str, NaNLabelEncoder] = {}
    for col in categorical_columns:
        encoder = NaNLabelEncoder(add_nan=True, warn=True)
        if col in ppgr_df.columns:
            encoder.fit(ppgr_df[col])
        elif col in users_demographics_df.columns:
            encoder.fit(users_demographics_df[col])
        else:
            raise ValueError(f"Categorical column '{col}' not found in either dataframe.")
        categorical_encoders[col] = encoder
        
    # Initialize and fit continuous scalers on the training set only
    continuous_scalers: dict[str, StandardScaler] = {}
    for col in real_columns:
        if col in training_df.columns:
            values = training_df[col].to_numpy().reshape(-1, 1)
        elif col in users_demographics_df.columns:
            values = users_demographics_df[col].to_numpy().reshape(-1, 1)
        else:
            raise ValueError(f"Real column '{col}' not found in training or demographics dataframe.")
        
        scaler = StandardScaler().fit(values)
        logger.info(f"Fitted scaler for '{col}': mean={scaler.mean_}, scale={scaler.scale_}")
        continuous_scalers[col] = scaler

    return categorical_encoders, continuous_scalers
