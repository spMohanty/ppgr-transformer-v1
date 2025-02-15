
import pandas as pd
from loguru import logger
import uuid

# =============================================================================
# Data Loading and Preprocessing Functions
# =============================================================================
def load_dataframe(dataset_version: str, debug_mode: bool) -> pd.DataFrame:
    """
    Load the processed CSV file into a DataFrame and sort by user, block, and time.
    """
    if debug_mode:
        PREFIX = "debug-"
        SUBDIR = "debug/"
    else:
        PREFIX = ""
        SUBDIR = ""

    ppgr_df_path = (
        f"data/processed/{dataset_version}/{SUBDIR}"
        f"{PREFIX}fay-ppgr-processed-and-aggregated-{dataset_version}.csv"
    )
    users_demographics_df_path = (
        f"data/processed/{dataset_version}/{SUBDIR}"
        f"{PREFIX}users-demographics-data-{dataset_version}.csv"
    )
    
    ppgr_df = pd.read_csv(ppgr_df_path)
    users_demographics_df = pd.read_csv(users_demographics_df_path)
    
    # set user_id as int in both dataframes
    ppgr_df["user_id"] = ppgr_df["user_id"].astype(str)
    users_demographics_df["user_id"] = users_demographics_df["user_id"].astype(str)
    
    # Merge user demographics into ppgr_df
    # ppgr_df = ppgr_df.merge(users_demographics_df, on="user_id", how="left")
    
    ppgr_df = ppgr_df.sort_values(by=["user_id", "timeseries_block_id", "read_at"])
    logger.info(f"Loaded dataframe with {len(ppgr_df)} rows from {ppgr_df_path}")
    return ppgr_df, users_demographics_df

def create_slice(
    df: pd.DataFrame, slice_start: int, slice_end: int
) -> pd.DataFrame:
    """Helper to create a time-series slice with a unique cluster id and time index."""
    df_slice = df.iloc[slice_start:slice_end].copy()
    df_slice["time_series_cluster_id"] = str(uuid.uuid4())
    df_slice["time_idx"] = list(range(len(df_slice)))
    return df_slice