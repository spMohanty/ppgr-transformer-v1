import pandas as pd
from pytorch_forecasting.data import TimeSeriesDataSet
from loguru import logger

from sklearn.preprocessing import StandardScaler

from typing import Tuple, Optional
from typing import Dict, List


from tqdm import tqdm

from utils import load_dataframe, enforce_column_types, setup_scalers_and_encoders, ppgr_collate_fn

from pytorch_forecasting.data.encoders import (
    NaNLabelEncoder,
)
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch.nn.functional as F

import p_tqdm

import numpy as np
import hashlib
import os

import random

import torch
import torch.nn as nn   

def split_timeseries_df_based_on_food_intake_rows(
    df: pd.DataFrame,
    validation_percentage: float,
    test_percentage: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create time-series slices anchored around food intake rows and split them
    into training, validation, and test sets.
    """
    training_data = []
    validation_data = []
    test_data = []
    
    training_percentage = 1 - validation_percentage - test_percentage
    
    df["read_at"] = pd.to_datetime(df["read_at"])
    
    # sort dataframe by user_id and read_at to ensure that the data is in the correct order
    df = df.sort_values(by=["user_id", "read_at"]).reset_index(drop=True)
    
    for user_id, group in tqdm(df.groupby("user_id"), total=len(df["user_id"].unique())):
        # Get indices where food intake occurred
        food_intake_indices = group[group["food_intake_row"] == 1].index.tolist()
        n_samples = len(food_intake_indices)
        
        if n_samples == 0:
            continue
        
        # Calculate split points (in terms of the food intake rows)
        train_end = int(training_percentage * n_samples)
        val_end = int((training_percentage + validation_percentage) * n_samples)
        
        # Get the boundary indices for each split
        train_boundary = food_intake_indices[train_end-1] if train_end > 0 else group.index[0]
        val_boundary = food_intake_indices[val_end-1] if val_end > train_end else train_boundary
        
        # Split all rows based on these boundaries
        training_data_slices = group.loc[:train_boundary]
        validation_data_slices = group.loc[train_boundary+1:val_boundary]
        test_data_slices = group.loc[val_boundary+1:]
        
        # Add checks to ensure that the data is in the correct order
        assert training_data_slices["read_at"].is_monotonic_increasing
        assert validation_data_slices["read_at"].is_monotonic_increasing
        assert test_data_slices["read_at"].is_monotonic_increasing
        
        # Add checks to ensure that the validation data is later than the training data
        # and test data is later than validation data
        try:
            assert validation_data_slices["read_at"].max() > training_data_slices["read_at"].max()
            assert test_data_slices["read_at"].max() > validation_data_slices["read_at"].max()
        except Exception as e:
            print(f"Error: {e}")
            breakpoint()
        
        # Add the splits to their respective lists
        if not training_data_slices.empty:
            training_data.append(training_data_slices)
        if not validation_data_slices.empty:
            validation_data.append(validation_data_slices)
        if not test_data_slices.empty:
            test_data.append(test_data_slices)
    
        
    # Combine all users' data
    training_df = pd.concat(training_data)
    validation_df = pd.concat(validation_data)
    test_df = pd.concat(test_data)
        
    return training_df, validation_df, test_df        
    

class PPGRTimeSeriesSliceMetadata:
    def __init__(self, 
                 slice_start: int,
                 slice_end: int,
                 anchor_row_idx: int, 
                 block_start: int,  # This is the start of the timeseries block
                 block_end: int,  # This is the end of the timeseries block
                 microbiome_idx: Optional[int] = None):  # This is the index of the microbiome embedding for this slice
        self.slice_start = slice_start
        self.slice_end = slice_end
        self.anchor_row_idx = anchor_row_idx
        self.block_start = block_start
        self.block_end = block_end
        self.microbiome_idx = microbiome_idx
    
    def __repr__(self):
        return f"PPGRTimeSeriesSliceMetadata(slice_start={self.slice_start}, slice_end={self.slice_end}, encoder_length={self.encoder_length}, prediction_length={self.prediction_length})"
        

class PPGRTimeSeriesDataset(Dataset):
    def __init__(self,  ppgr_df: pd.DataFrame,
                        user_demographics_df: pd.DataFrame,
                        dishes_df: pd.DataFrame,
                        
                        is_food_anchored: bool = False, # decides if the timeseries is food anchored or not
                        sliding_window_stride: int = 10, # decides the stride of the sliding window (Only used when is_food_anchored is False)
                        
                        time_idx: str = "time_idx", # Unique integral column in each timeseries for each datapoint (should be in the same order as read_at) 

                        target_columns: List[str] = ["val"], # These are the columns that are the targets (should also be provided in the time_varying_unknown_reals
                        group_by_columns: List[str] = ["timeseries_block_id"], # Decides what the timeseries grouping/boundaries are
                        
                        min_encoder_length: int = 8 * 4, # context window
                        max_encoder_length: int = 12 * 4, # context window
                        prediction_length: int = 3 * 4, # prediction window
                        disable_encoder_length_randomization: bool = False, # decides if the encoder length should be randomized betwen min and max encoder length. 
                        
                        use_food_covariates_from_prediction_window: bool = True, # decides if the food covariates are used from the prediction window as well
                        
                        use_meal_level_food_covariates: bool = True, # decides if the meal level food covariates are used (instead of the aggregated onces)
                        max_meals_per_timestep: int = 11, # maximum number of meals per timestep, the rest are discarded
                        
                        use_microbiome_embeddings: bool = True, # decides if the microbiome embeddings are used
                        microbiome_embeddings_df: Optional[pd.DataFrame] = None, # the microbiome embeddings dataframe
                        
                        use_bootstraped_food_embeddings: bool = True, # decides if the food embeddings are bootstrapped
                        food_embeddings_df: Optional[pd.DataFrame] = None, # the food embeddings dataframe
                        
                        temporal_categoricals: List[str] = [], # these also need to be provded in the static_categoricals list
                        temporal_reals: List[str] = [], # these also need to be provded in the static_reals list
                        
                        user_static_categoricals: List[str] = [], # these also need to be provded in the static_categoricals list
                        user_static_reals: List[str] = [], # these also need to be provded in the static_reals list
                        
                        food_categoricals: List[str] = [], # these also need to be provded in the static_categoricals list
                        food_reals: List[str] = [], # these also need to be provded in the static_reals list
                        
                        categorical_encoders: Dict[str, NaNLabelEncoder] = {},  # These have to be provided to avoid mistakes
                        continuous_scalers: Dict[str, StandardScaler] = {}, # These have to be provided to avoid mistakes
                        
                        device: torch.device = torch.device("cpu")
                    ):
        self.ppgr_df = ppgr_df
        self.user_demographics_df = user_demographics_df
        self.dishes_df = dishes_df
        
        self.is_food_anchored = is_food_anchored
        self.sliding_window_stride = sliding_window_stride
        
        self.time_idx = time_idx
        
        self.target_columns = target_columns
        
        self.min_encoder_length = min_encoder_length
        self.max_encoder_length = max_encoder_length
        self.prediction_length = prediction_length
        self.disable_encoder_length_randomization = disable_encoder_length_randomization
                
        self.use_food_covariates_from_prediction_window = use_food_covariates_from_prediction_window
        
        self.use_meal_level_food_covariates = use_meal_level_food_covariates
        self.max_meals_per_timestep = max_meals_per_timestep
        
        self.use_microbiome_embeddings = use_microbiome_embeddings
        self.microbiome_embeddings_df = microbiome_embeddings_df
        
        
        self.use_bootstraped_food_embeddings = use_bootstraped_food_embeddings
        self.food_embeddings_df = food_embeddings_df
        
        # Temporal Covariates
        self.temporal_categoricals = temporal_categoricals
        self.temporal_reals = temporal_reals
        
        # User Static Covariates
        self.user_static_categoricals = user_static_categoricals
        self.user_static_reals = user_static_reals
        
        # Food Static Covariates
        self.food_categoricals = food_categoricals
        self.food_reals = food_reals

        self.group_by_columns = group_by_columns
        self.categorical_encoders = categorical_encoders
        self.continuous_scalers = continuous_scalers
        
        self.device = device
        
        self.timeseries_slices_indices = [] # Will house a list of PPGRTimeSeriesSliceMetadata that will be referenced in __getitem__

        self.validate_parameters()
        self.prepare_data()
    
    def validate_parameters(self):
        assert self.min_encoder_length > 0, "Minimum encoder length must be greater than 0"
        assert self.max_encoder_length > self.min_encoder_length or self.max_encoder_length is None, "Maximum encoder length must be greater than minimum encoder length, or set to False"
        assert self.prediction_length > 0, "Prediction length must be greater than 0"
        assert self.sliding_window_stride is None or self.sliding_window_stride > 0, "Sliding window stride must be greater than 0"
        
        if self.is_food_anchored:
            assert self.sliding_window_stride is None, "Sliding window stride must be None when is_food_anchored is True"
        else:
            assert self.sliding_window_stride is not None, "Sliding window stride must be provided when is_food_anchored is False"

    def prepare_data(self):
        
        # 0. Merge the ppgr_df and user_demographics_df
        self.merge_ppgr_and_user_demographics()
        
        # 2. Prepare the all dataset columns
        self.prepare_all_dataset_columns()                
                
        # 1. Scale and CatEmbed the dataset df
        self.prepare_scaled_and_catembed_dataset()
        
        # 1.1. Prepare the food embeddings
        self.prepare_food_embeddings()
        
        # 1.2. Prepare the microbiome embeddings
        self.prepare_microbiome_embeddings()
                
        if self.is_food_anchored:
            # 3.1. Prepare the data for food anchored timeseries
            self.prepare_food_anchored_data()
        else:
            # 3.2 Prepare the data for the standard sliding window timeseries    
            self.prepare_sliding_window_data()
            
        # 4. Prepare the cross index of the ppgr_df and the dishes_df
        self.prepare_cross_index_of_ppgr_df_and_dishes_df()
    
    def merge_ppgr_and_user_demographics(self):
        logger.debug("Merging ppgr_df and user_demographics_df")
        self.ppgr_df = self.ppgr_df.merge(self.user_demographics_df, on="user_id", how="left")
        self.ppgr_df = self.ppgr_df.sort_values(by=self.group_by_columns + [self.time_idx]).reset_index(drop=True)
        logger.debug("ppgr_df and user_demographics_df merged")
    
    def prepare_all_dataset_columns(self):
        
        if not self.use_meal_level_food_covariates:
            # Gather all columns in a single place
            self.main_df_scaled_all_categorical_columns = sorted(self.user_static_categoricals + self.food_categoricals + self.temporal_categoricals)
            self.main_df_scaled_all_real_columns = sorted(self.user_static_reals + self.food_reals + self.temporal_reals + self.target_columns)
            self.main_df_all_columns = self.main_df_scaled_all_categorical_columns + self.main_df_scaled_all_real_columns
        else:
            # Case when using meal level food covariates
            
            # Main DF columns  
            self.main_df_scaled_all_categorical_columns = sorted(self.user_static_categoricals + self.temporal_categoricals)
            self.main_df_scaled_all_real_columns = sorted(self.user_static_reals + self.temporal_reals + self.target_columns)
            self.main_df_all_columns = self.main_df_scaled_all_categorical_columns + self.main_df_scaled_all_real_columns


        # Gather the indices of the individual columns for easy access using tensor indexing
        # Temporal Covariates   
        self.temporal_categorical_col_idx = [self.main_df_all_columns.index(col) for col in self.temporal_categoricals]
        self.temporal_real_col_idx = [self.main_df_all_columns.index(col) for col in self.temporal_reals]        
        
        # User Static Covariates
        self.user_static_categorical_col_idx = [self.main_df_all_columns.index(col) for col in self.user_static_categoricals]
        self.user_static_real_col_idx = [self.main_df_all_columns.index(col) for col in self.user_static_reals]
                            
        # Target Columns
        self.target_col_idx = [self.main_df_all_columns.index(col) for col in self.target_columns]

        if not self.use_meal_level_food_covariates:
            # Food Categorical Columns for the main DF
            self.food_categorical_col_idx = [self.main_df_all_columns.index(col) for col in self.food_categoricals]
            self.food_real_col_idx = [self.main_df_all_columns.index(col) for col in self.food_reals]          
        else: 
            # Food Categorical Columns for the dishes_df
            self.dish_df_scaled_all_columns = sorted(self.food_categoricals + self.food_reals)
            
            self.food_categorical_col_idx = [self.dish_df_scaled_all_columns.index(col) for col in self.food_categoricals] 
            self.food_real_col_idx = [self.dish_df_scaled_all_columns.index(col) for col in self.food_reals]

    
    def prepare_scaled_and_catembed_dataset(self):
        """
        Prepares a scaled + categorical embedded version of the dataset that can be readily accessed in __getitem__
        """
        ### 1. Handle the scaling of the main DF first

        self.df_scaled = self.ppgr_df.copy() # NOTE: This should be the merged dataframe by now
        
        assert (self.df_scaled.index == self.ppgr_df.index).all()
        
        # Scale the continuous variables
        for col in self.main_df_scaled_all_real_columns:
            logger.debug(f"Scaling {col}")            
            self.df_scaled[col] = self.continuous_scalers[col].transform(
                self.df_scaled[col].to_numpy().reshape(-1, 1)
            )
            
        ### 2. Encode the categorical variables
        for col in self.main_df_scaled_all_categorical_columns:            
            logger.debug(f"Encoding {col}")
            self.df_scaled[col] = self.categorical_encoders[col].transform(self.df_scaled[col])
            
        # Add target scales to conveniently access in __getitem__
        self.target_scales =[]
        for col in self.target_columns:
            self.target_scales.append([self.continuous_scalers[col].mean_, self.continuous_scalers[col].scale_])
        self.target_scales = torch.tensor(np.array(self.target_scales)).to(self.device).squeeze()
        
        # reset the index so we can start from index 0 and maintain a continuous index that aligns with the torch indexing system
        self.df_scaled = self.df_scaled.reset_index(drop=True)
        
        # Create a tensor optimized version of the dataframe with only the relevant columns
        self.df_scaled_tensor = torch.tensor(self.df_scaled[self.main_df_all_columns].to_numpy()).to(self.device)
        
        # NOTE: It is important that the indices of both df_scaled and df_scaled_tensor align, 
        # which is why we did the reset_index just before
        
        ### 3. Handle the scaling of the dishes_df if required
        if self.use_meal_level_food_covariates:
            # Sort dishes_df by food_eaten_quantity_in_gram
            # this sort ensures that when we are discarding meals > max_meals_per_timestep,
            # we discard the meals with the least quantity first
            self.dishes_df = self.dishes_df.sort_values(by="food__eaten_quantity_in_gram", ascending=False)
            self.dishes_df_scaled = self.dishes_df.copy()
            
            assert len(self.food_categoricals) > 0, "Food categoricals should not be empty when using meal level food covariates. We should atleast have food_id column"
            assert "food_id" in self.food_categoricals, "food_id should be present in the food_categoricals list when using meal level food covariates"
            
            assert len(self.food_reals) > 0, "Food reals should not be empty when using meal level food covariates"
            
            # In this mode, the food_id and food_group_cname are the categorical columns
            
            # Scale the continuous variables
            for col in self.food_reals:
                logger.debug(f"Scaling {col}")
                # Note: in this mode, the continuous scalers are already fit on the dishes data (instead of that of the ppgr_df)
                # check setup_scalers_and_encoders in utils.py for more details
                
                # temporary: some column values may be nan, so we fillna with 0 first (this is not the case for the ppgr_df)
                # ideally, we should move this pre-processing to the fay-data-aggregator. 
                self.dishes_df_scaled[col] = self.dishes_df_scaled[col].fillna(0)
                
                # scale !
                self.dishes_df_scaled[col] = self.continuous_scalers[col].transform(
                    self.dishes_df_scaled[col].to_numpy().reshape(-1, 1)
                )
            # Encode the categorical variables
            for col in self.food_categoricals:
                logger.debug(f"Encoding {col}")
                self.dishes_df_scaled[col] = self.categorical_encoders[col].transform(self.dishes_df_scaled[col])
            
            
            # Prepare the dishes df tensor
            self.dishes_df_scaled = self.dishes_df_scaled.reset_index(drop=True)
            self.dishes_df_scaled_tensor = torch.tensor(self.dishes_df_scaled[self.dish_df_scaled_all_columns].to_numpy()).to(self.device)        
    
    def prepare_food_embeddings(self):
        """
        Prepare the food embeddings layer if bootstraped food embeddings are enabled.

        This function creates a frozen PyTorch embedding layer from the embeddings provided
        in self.food_embeddings_df. The embeddings are arranged according to the internal food ID
        mapping found in self.categorical_encoders["food_id"]. It asserts that every encoded
        food ID (except for the 'nan' placeholder) is present in the dishes dataframe.

        Preconditions:
          - self.use_bootstraped_food_embeddings is True.
          - "food_id" must be present in self.categorical_encoders.
          - All encoded food_ids (apart from 'nan') in the ppgr_df are present in the dishes_df.
        """
        if not self.use_bootstraped_food_embeddings:
            return

        if "food_id" not in self.categorical_encoders:
            raise AssertionError(
                "food_id should be present in the categorical_encoders dictionary when using bootstraped food embeddings"
            )

        # Convert encoder keys and dish food_ids to sets of strings for robust comparison
        encoded_food_ids = set(self.categorical_encoders["food_id"].classes_.keys())
        dishes_food_ids = set(self.dishes_df["food_id"].astype(str).unique())

        # Ensure that all food_ids in the ppgr_df (except the 'nan' placeholder) are present in the dishes_df
        missing_food_ids = encoded_food_ids - dishes_food_ids
        expected_missing = {"nan"}
        if missing_food_ids != expected_missing:
            raise AssertionError(
                f"All encoded food_ids in the ppgr_df must be present in the food_embeddings_df. Missing: {missing_food_ids}"
            )

        # Compute the mapping from MFR food_id to internal food id.
        # Although this mapping is not used further for reordering, it confirms consistency.
        mfr_food_id_to_internal_food_id = self.categorical_encoders["food_id"].classes_
        
        # This is the ordering of the internal id vs the food_embeddings_df.index
        food_embeddings_df_internal_food_ids = [mfr_food_id_to_internal_food_id[str(mfr_food_id)] for mfr_food_id in self.food_embeddings_df.index]
        food_embeddings_df_internal_food_ids = torch.tensor(food_embeddings_df_internal_food_ids)


        # Define the vocabulary size and determine the embedding dimension from the first vector.
        vocabulary_size = len(self.categorical_encoders["food_id"].classes_)
        embedding_vector = self.food_embeddings_df["embedding"].iloc[0]
        embedding_dim = len(embedding_vector)

        # Create a frozen embedding layer and initialize its weights with the stacked embedding vectors.
        self.food_id_embeddings_layer = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_dim,
            _freeze=True
        )
        self.food_id_embeddings_layer.weight.data.zero_() # default weight of 0 for nan values
        embeddings_tensor = torch.stack(self.food_embeddings_df["embedding"].tolist())
        self.food_id_embeddings_layer.weight.data[food_embeddings_df_internal_food_ids] = embeddings_tensor
        
        # Gather human readable food names for easier debugging        
        # Reorder to match the ordering of the embeddings
        custom_order = list(self.categorical_encoders["food_id"].classes_.keys())[1:] # skip the nan value
        custom_order = [int(x) for x in custom_order]
        unique_foods_df = self.food_embeddings_df.loc[custom_order]
        
                
        # Add a human readable list of food names for the model plotters to readily use
        self.food_names = ["UNKNOWN"] + unique_foods_df["display_name"].tolist() # Add UNKNOWN to represent the nan value as 0 index 
        self.food_group_names = ["UNKNOWN"] + unique_foods_df["food_group_cname"].tolist()
            
    def get_food_id_embeddings_layer(self):
        """
        Get the embeddings for the given food ids
        """
        return self.food_id_embeddings_layer
        
        
    def prepare_microbiome_embeddings(self):
        """
        microbiome embeddings expect that all the user_ids are present in the microbiome embeddings dataframe
        and that the data is already normalized
        """
        self.microbiome_embeddings_df = self.microbiome_embeddings_df.sort_index() # Note: user_id is the index of the microbiome embeddings dataframe
        # assert set(self.ppgr_df["user_id"].astype(int).unique()) == set(self.microbiome_embeddings_df.index), "All user_ids in the ppgr_df must be present in the microbiome_embeddings_df, and vice versa"
        
        assert set(self.ppgr_df["user_id"].astype(int).unique()) - set(self.microbiome_embeddings_df.index) == set(), "All user_ids in the ppgr_df must have a microbiome embedding in the microbiome_embeddings_df, and vice versa"

        self.microbiome_tensor = torch.tensor(self.microbiome_embeddings_df.to_numpy()).to(self.device)
        self.user_id_to_microbiome_idx = {user_id: idx for idx, user_id in enumerate(self.microbiome_embeddings_df.index)}
        
    
    def prepare_sliding_window_data(self):
        # Iterate over each of the timeseries_block_ids 
        groups = self.df_scaled.groupby(self.group_by_columns)
        for _, group in tqdm(groups, total=len(groups)):
            block_start = group.index[0]
            block_end = group.index[-1]
            
            # 3. Keep a record of the microbiome index for this slice            
            if self.use_microbiome_embeddings:
                if "user_id" in self.categorical_encoders:
                    # When user_id is a provided covariate
                    user_id = int(self.categorical_encoders["user_id"].inverse_transform(group["user_id"].iloc[0]))
                else:
                    # When user_id is directly available from the dataframe
                    # CAREFUL: if the user_id is a categorical variable, this this value is an encoded version
                    # and doesnt not match the user_id in the whole dataset
                    user_id = int(group["user_id"].iloc[0])
                microbiome_idx = self.user_id_to_microbiome_idx[user_id]
            else:
                microbiome_idx = None
            
            # Iterate over the whole timeseries block            
            for row_idx in tqdm(range(block_start, block_end + 1, self.sliding_window_stride), total=len(range(block_start, block_end + 1, self.sliding_window_stride))):
                # 1. Identify the start and end of the context and prediction window
                slice_start = row_idx - self.min_encoder_length + 1
                slice_end = row_idx + self.prediction_length
                
                # 2. Ignore the fringe points where the slice is not fully within the block
                if slice_start < block_start or slice_end > block_end: continue 
                
                # print(f"slice_start: {slice_start}, slice_end: {slice_end}")
                # 3. Keep a record of the slice metadata to be used later in __getitem__
                self.timeseries_slices_indices.append(
                    PPGRTimeSeriesSliceMetadata(
                        slice_start = slice_start, # recommended slice start  (using the minimum encoder length)
                        slice_end = slice_end, # recommended slice end (using the minimum prediction length)
                        anchor_row_idx = row_idx, # anchor row for this slice (usually the food intake row, but not necessarily)
                        block_start = block_start, # start of the timeseries block
                        block_end = block_end, # end of the timeseries block
                        microbiome_idx = microbiome_idx # keep track of the microbiome index for this slice
                    ))
            
    
    def prepare_food_anchored_data(self):
        # 1. Iterate over each of the timeseries_block_ids 
        groups = self.df_scaled.groupby(self.group_by_columns)
        for _, group in tqdm(groups, total=len(groups)):            
            # 2 Keep a record of the block_start so that we can randomize the encoder lengths during the training
            block_start = group.index[0]
            block_end = group.index[-1]
            
            # 3. Keep a record of the microbiome index for this slice            
            if self.use_microbiome_embeddings:
                if "user_id" in self.categorical_encoders:
                    # When user_id is a provided covariate
                    user_id = int(self.categorical_encoders["user_id"].inverse_transform(group["user_id"].iloc[0]))
                else:
                    # When user_id is directly available from the dataframe
                    # CAREFUL: if the user_id is a categorical variable, this this value is an encoded version
                    # and doesnt not match the user_id in the whole dataset
                    user_id = int(group["user_id"].iloc[0])
                microbiome_idx = self.user_id_to_microbiome_idx[user_id]
            else:
                microbiome_idx = None

            # 4. For each timeseries_block_id, identify the food intake rows
            food_intake_mask = group["food_intake_row"] == 1
            
            # 5. Ensure that the there is enough past and future data for all potential slices
            food_intake_mask.iloc[:self.min_encoder_length] = False
            food_intake_mask.iloc[-self.prediction_length:] = False
            
            # 6. Get all the rows that have enough past and future data
            food_intake_rows = group[food_intake_mask]

            # 7. Iterate over all the food intake rows
            for row_idx, _ in food_intake_rows.iterrows():
                # 8. For each food intake row, identify the start and end of the context and prediction window
                slice_start = row_idx - self.min_encoder_length + 1 # row_idx is the food anchor, and we want the encoder slice to end with it
                slice_end = row_idx + self.prediction_length
                
                # 9. Store slice start, and slice end to be retrieved later in __getitem__
                self.timeseries_slices_indices.append(
                    PPGRTimeSeriesSliceMetadata(
                        slice_start = slice_start, # recommended slice start  (using the minimum encoder length)
                        slice_end = slice_end, # recommended slice end (using the minimum prediction length)
                        anchor_row_idx = row_idx, # anchor row for this slice (usually the food intake row, but not necessarily)
                        block_start = block_start, # start of the timeseries block
                        block_end = block_end, # end of the timeseries block
                        microbiome_idx = microbiome_idx # keep track of the microbiome index for this slice
                    ))
    
    def prepare_cross_index_of_ppgr_df_and_dishes_df(self):
        """
        This function prepares a cross index of the ppgr_df and the dishes_df
        """
    
        # For all the rows in ppgr_df, we want to identify the exact index number of the corresponding
        # dish items, so that it is easy to access them in __getitem__        

        # This assumes that ppgr_df, df_scaled, dishes_df will not change their indices
        # or be reset etc after
        
        
        if not self.use_meal_level_food_covariates:
            # This function is not needed when not using the meal level food covariates
            return 
                
        # rows with non-null dish ids
        ppgr_df_with_dish_ids = self.ppgr_df[~self.ppgr_df["dish_id"].isna()]
        
        
        self.ppgr_df_row_idx_to_dishes_df_idxs = {} # Dictionary which will store a mapping of ppgr_df index to dishes_df_scaled indices
        
        # Iterate over each of the rows
        for idx, row in tqdm(ppgr_df_with_dish_ids.iterrows(), total=len(ppgr_df_with_dish_ids)):
            row_index = row.name
            
            # currently multiple dish ids are stored as: "123 || 456"
            mfr_dish_ids = [int(x) for x in str(row["dish_id"]).split("||")]
            
            internal_dish_idxs = [] # These will store the indices of the dishes_df_scaled tensor
            for mfr_dish_id in mfr_dish_ids:
                internal_dish_idx = self.dishes_df_scaled[self.dishes_df_scaled["dish_id"] == mfr_dish_id].index.values
                internal_dish_idxs.append(internal_dish_idx)
        
            self.ppgr_df_row_idx_to_dishes_df_idxs[row_index] = torch.tensor(np.concatenate(internal_dish_idxs)).to(self.device)
        
    def __len__(self):
        return len(self.timeseries_slices_indices)
    
    def _get_slice_start_and_end(self, slice_metadata: PPGRTimeSeriesSliceMetadata):
        
        # Get block boundaries and anchor point
        block_start = slice_metadata.block_start
        slice_anchor_row_idx = slice_metadata.anchor_row_idx
        slice_end = slice_metadata.slice_end
        
        if self.max_encoder_length is False:
            encoder_length = self.min_encoder_length
        else:
            # Randomly sample encoder length between min and max
            max_possible_encoder_length = min(
                self.max_encoder_length,
                slice_anchor_row_idx - block_start  # Ensure we don't go before block start
            )
            
            if self.disable_encoder_length_randomization:
                # If we are not randomizing, we just use the max possible encoder length
                encoder_length = max_possible_encoder_length
            else:
                encoder_length = torch.randint(
                    low=self.min_encoder_length,
                    high=max_possible_encoder_length + 1,  # +1 because randint's high is exclusive
                    size=(1,)
                ).item()
        
        # Calculate slice start based on randomly sampled encoder length
        slice_start = slice_anchor_row_idx - encoder_length + 1
        slice_end = slice_metadata.slice_end
        
        return slice_start, slice_end, encoder_length

    def _get_dish_tensors_for_this_slice(self, slice_start: int, slice_end: int):
        """
        This function returns the dish tensors for the current slice, with a special start/empty token
        at the beginning of each timestep's sequence.
        
        Args:
            slice_start (int): Starting index of the slice (inclusive)
            slice_end (int): Ending index of the slice (inclusive)        
        Returns:
            Tuple of lists containing categorical and real tensors for dishes
        """
        
        # For all the rows in the current slice, gather the dish_idxs
        dish_tensors_cat_for_this_slice = []
        dish_tensors_real_for_this_slice = []
        
        # Get dimensions for start/empty token tensor creation
        num_cat_features = len(self.food_categorical_col_idx)
        num_real_features = len(self.food_real_col_idx)
            
        # TODO: Setup sensible values for the start token
        max_meals_per_timestep = self.max_meals_per_timestep
        
        dish_tensors_recorded = []
        # Use range(start, end + 1) to make end inclusive
        for slice_row_idx in range(slice_start, slice_end + 1):
            dish_idxs_for_this_row = self.ppgr_df_row_idx_to_dishes_df_idxs.get(slice_row_idx, None)
            
            if dish_idxs_for_this_row is None or len(dish_idxs_for_this_row) == 0:                                
                # register that no dish tensors created for this timestep        
                cat_tensor = torch.empty(0, num_cat_features, device=self.device)
                real_tensor = torch.empty(0, num_real_features, device=self.device)

                dish_tensors_recorded.append(0) # adding 1 as we have a start token
                dish_tensors_cat_for_this_slice.append(cat_tensor)
                dish_tensors_real_for_this_slice.append(real_tensor)
            else:
                # If dishes exist, concatenate start token with dish tensors
                dish_tensors_for_this_row = self.dishes_df_scaled_tensor[dish_idxs_for_this_row]
                                                
                # Only use up to max_dishes_per_row 
                cat_tensor = torch.cat([
                    dish_tensors_for_this_row[:max_meals_per_timestep, self.food_categorical_col_idx]
                ], dim=0)
                real_tensor = torch.cat([
                    dish_tensors_for_this_row[:max_meals_per_timestep, self.food_real_col_idx]
                ], dim=0)
                                
                dish_tensors_recorded.append(len(dish_idxs_for_this_row))         
                dish_tensors_cat_for_this_slice.append(cat_tensor)
                dish_tensors_real_for_this_slice.append(real_tensor)

        
        # We return everything as a nested tensor        
        dish_tensors_cat_for_this_slice = torch.nested.nested_tensor(dish_tensors_cat_for_this_slice)
        dish_tensors_real_for_this_slice = torch.nested.nested_tensor(dish_tensors_real_for_this_slice)
                
        # Padd the sequences to the same size for easier batching
        # Probably better to move the padding to the collate function
        # PADDING_VALUE = -1
        # dish_tensors_cat_for_this_slice = torch.nested.to_padded_tensor(dish_tensors_cat_for_this_slice, PADDING_VALUE, output_size=(max_meals_per_timestep, num_cat_features))
        # dish_tensors_real_for_this_slice = torch.nested.to_padded_tensor(dish_tensors_real_for_this_slice, PADDING_VALUE, output_size=(max_meals_per_timestep, num_real_features))

        ## Padding example: 
        ### 32 = batch size
        ### 8 = max number of dishes per row (can change this as we please)
        ### 2 = number of features per dish (cat and real)
        ### torch.nested.to_padded_tensor(ab, -1, output_size=(32, 8, 2))
        
        return dish_tensors_cat_for_this_slice, dish_tensors_real_for_this_slice, dish_tensors_recorded


    def get_item_from_df(self, idx: int):
        slice_metadata = self.timeseries_slices_indices[idx]
        
        slice_start, slice_end, encoder_length = self._get_slice_start_and_end(slice_metadata) # Takes care of randomizing the encoder length between min and max encoder lengths
        slice_anchor = slice_metadata.anchor_row_idx
        prediction_length = slice_end - slice_anchor 
        
        return self.ppgr_df.iloc[slice_start:slice_end+1][self.main_df_all_columns]
        
    
    def __getitem__(self, idx):
        slice_metadata = self.timeseries_slices_indices[idx]
        

        slice_start, slice_end, encoder_length = self._get_slice_start_and_end(slice_metadata) # Takes care of randomizing the encoder length between min and max encoder lengths
        slice_anchor = slice_metadata.anchor_row_idx
        prediction_length = slice_end - slice_anchor 
        
        # Get the slice from the dataframe
        data_slice_tensor = self.df_scaled_tensor[slice_start:slice_end+1, :].clone() # [T, C] where T is the number of rows in the slice and C is the number of all relevant columns
                
        # Calculate slices for encoder and prediction window
        data_slice_encoder_tensor = data_slice_tensor[:encoder_length, :]
        data_slice_prediction_window_tensor = data_slice_tensor[-prediction_length:, :]
        
        
        assert data_slice_encoder_tensor.shape[0] == encoder_length
        assert data_slice_prediction_window_tensor.shape[0] == prediction_length
        
        ############################################################
        # Gather Encoder Variables: 
        ############################################################        
        
        # Temporal Covariates
        x_temporal_cat = data_slice_encoder_tensor[:, self.temporal_categorical_col_idx]
        x_temporal_real = data_slice_encoder_tensor[:, self.temporal_real_col_idx]
        
        # User Static Covariates
        x_user_cat = data_slice_encoder_tensor[:, self.user_static_categorical_col_idx]
        x_user_real = data_slice_encoder_tensor[:, self.user_static_real_col_idx]
        
        # Food Static Covariates
        if self.use_meal_level_food_covariates:
            # Gather the food covariates for the current slice            
            x_food_cat, x_food_real, x_dish_tensors_recorded = self._get_dish_tensors_for_this_slice(slice_start, slice_anchor)
            
            # x_dish_tensors_recorded is a list of the number of dish tensors recorded for each row in the slice
            # we can use this to reconstruct the structure of the food covariates later
        else:    
            # Gather the dish_idxs for the current slice            
            x_food_cat = data_slice_encoder_tensor[:, self.food_categorical_col_idx]
            x_food_real = data_slice_encoder_tensor[:, self.food_real_col_idx]
        
        # Gather Historical Value of the Target(s)
        x_real_target = data_slice_encoder_tensor[:, self.target_col_idx]
        
        if self.use_microbiome_embeddings:
            x_microbiome_embedding = self.microbiome_tensor[slice_metadata.microbiome_idx, :]
        else:
            x_microbiome_embedding = None

        ############################################################
        # Gather Prediction Window Variables
        ############################################################
        
        # Temporal Covariates
        y_temporal_cat = data_slice_prediction_window_tensor[:, self.temporal_categorical_col_idx]
        y_temporal_real = data_slice_prediction_window_tensor[:, self.temporal_real_col_idx]

        # User Static Covariates
        y_user_cat = data_slice_prediction_window_tensor[:, self.user_static_categorical_col_idx]
        y_user_real = data_slice_prediction_window_tensor[:, self.user_static_real_col_idx]

        # Food Static Covariates
        if self.use_food_covariates_from_prediction_window:
            if self.use_meal_level_food_covariates:
                # When we want meal level data
                y_food_cat, y_food_real, y_dish_tensors_recorded = self._get_dish_tensors_for_this_slice(slice_anchor+1, slice_end)
                
                # Padd the sequences to the same length
                # y_food_cat = pad_sequence(y_food_cat, batch_first=True, padding_side="right", padding_value=-1)
                # y_food_real = pad_sequence(y_food_real, batch_first=True, padding_side="right", padding_value=-1)
            else:
                # standard aggregated food covariates
                y_food_cat = data_slice_prediction_window_tensor[:, self.food_categorical_col_idx]
                y_food_real = data_slice_prediction_window_tensor[:, self.food_real_col_idx]
        else:
            y_food_cat = None
            y_food_real = None

        # Target Variables
        y_real_target = data_slice_prediction_window_tensor[:, self.target_col_idx]
        target_scales = self.target_scales # mean, std : This can be varying later if we want to normalize by user_id or other columns

        if self.use_meal_level_food_covariates:
            x_dish_tensors_recorded = torch.tensor(x_dish_tensors_recorded, dtype=torch.int32).to(self.device) # [N]
            y_dish_tensors_recorded = torch.tensor(y_dish_tensors_recorded, dtype=torch.int32).to(self.device) # [N']
        else:
            x_dish_tensors_recorded = None
            y_dish_tensors_recorded = None

        
        _response = dict(
            # Encoder Temporal Variables (dynamic)
            x_temporal_cat = x_temporal_cat, # [N, T_c]
            x_temporal_real = x_temporal_real, # [N, T_r]
            
            # Encoder User Variables (static)
            x_user_cat = x_user_cat, # [N, U_c] wher N = T_enc 
            x_user_real = x_user_real, # [N, U_r]
            
            # Encoder Food Variables (dynamic)
            x_food_cat = x_food_cat, # [N, F_c]
            x_food_real = x_food_real, # [N, F_r]
            
            # Encoder Microbiome Embedding Variables (static)
            x_microbiome_embedding = x_microbiome_embedding, # [N, M]
            
            # Encoder Target Variables # Historical value of the overall target
            x_real_target = x_real_target, # [N, T_r]
            
            # Prediction Window Temporal Variables (dynamic)
            y_temporal_cat = y_temporal_cat, # [N', T_c]
            y_temporal_real = y_temporal_real, # [N', T_r]
            
            # Prediction Window User Variables (static)
            y_user_cat = y_user_cat, # [N', U_c] where N' = T_pred
            y_user_real = y_user_real, # [N', U_r]
            
            # Prediction Window Food Variables (dynamic)
            y_food_cat = y_food_cat, # [N', F_c]
            y_food_real = y_food_real, # [N', F_r]
            
            # Prediction Window Target Variables (Overall Target of the model)
            y_real_target = y_real_target, # [N', T_r]
            target_scales = target_scales, # [2]
            
            x_dish_tensors_recorded = x_dish_tensors_recorded, # [N]
            y_dish_tensors_recorded = y_dish_tensors_recorded, # [N']
            
            encoder_length = torch.tensor(encoder_length, dtype=torch.int32).to(self.device), # scalar 
            prediction_length = torch.tensor(prediction_length, dtype=torch.int32).to(self.device),    # converting to tensor so that the collate function doesnt need special cases.        
            
            
            data_index = torch.tensor(idx, dtype=torch.int32).to(self.device), # scalar
            
            # Metadata about the slice
            metadata = dict(
                categorical_encoders = self.categorical_encoders,
                continuous_scalers = self.continuous_scalers,
                main_df_all_columns = self.main_df_all_columns,
                temporal_categoricals = self.temporal_categoricals,
                temporal_reals = self.temporal_reals,
                user_static_categoricals = self.user_static_categoricals,
                user_static_reals = self.user_static_reals,
                food_categoricals = self.food_categoricals,
                food_reals = self.food_reals,
                target_columns = self.target_columns,
                use_meal_level_food_covariates = self.use_meal_level_food_covariates,
                max_meals_per_timestep = self.max_meals_per_timestep
            )
        )
                
        return _response
        
    def inverse_transform_values(self, values, column_type: str) -> pd.DataFrame:
        """
        Inverse transforms the values based on the column names
        and returns a well organized pandas dataframe
        
        values will be a subset of the relevant columns and not the whole dataset item
        """
        # Mapping of column types to their corresponding column lists and transformers
        column_mapping = {
            'temporal_cat': (self.temporal_categoricals, self.categorical_encoders),
            'temporal_real': (self.temporal_reals, self.continuous_scalers),
            'user_cat': (self.user_static_categoricals, self.categorical_encoders),
            'user_real': (self.user_static_reals, self.continuous_scalers),
            'food_cat': (self.food_categoricals, self.categorical_encoders),
            'food_real': (self.food_reals, self.continuous_scalers),
            'target': (self.target_columns, self.continuous_scalers),
        }
        
        if column_type not in column_mapping:
            raise ValueError(f"Column type '{column_type}' not found. Valid types are: {list(column_mapping.keys())}")

        # Gather the column names and the encoder dict
        columns, encoder_dict = column_mapping[column_type]
        
        
        # Handle the case of nested tensors
        if values.is_nested:
            raise Exception("Inverse Transform in case of meal level food covariates is not implemented yet.")
        
        # Check that the values are 2D tensor and handle empty tensor case
        assert len(values.shape) == 2
        
        # Handle empty tensor case
        if 0 in values.shape:
            return pd.DataFrame()    # return None for empty tensor
        
        
        transformed = []
        for col_idx, column_name in enumerate(columns):
            encoder = encoder_dict[column_name]
            if column_type in ["temporal_real", "user_real", "food_real", "target"]:
                _inverted_value = encoder.inverse_transform(values[:, col_idx].reshape(-1, 1)).flatten()
            else:
                _inverted_value = encoder.inverse_transform(values[:, col_idx].to(torch.int16))

            transformed.append(_inverted_value)
                
        # Organize the transformed values into the same shape as the input values
        transformed = np.stack(np.array(transformed).T) # using np instead of torch, as it might have string values
        
        # Create a pandas dataframe 
        df = pd.DataFrame(transformed, columns=columns)
        
        return df
    
    def inverse_transform_item(self, item: dict) -> pd.DataFrame:
        """
        Inverse transforms the item based on the column names
        and returns a well organized pandas dataframe
        
        NOTE: When we provide more granual info per food in an item, 
        we need to change this function to accomodate that
        """        
        # Map item keys to their column types
        key_to_type = {
            "x_temporal_cat": "temporal_cat",
            "y_temporal_cat": "temporal_cat",
            "x_temporal_real": "temporal_real",
            "y_temporal_real": "temporal_real",
            "x_user_cat": "user_cat",
            "y_user_cat": "user_cat",
            "x_user_real": "user_real",
            "y_user_real": "user_real",
            "x_food_cat": "food_cat",
            "y_food_cat": "food_cat",
            "x_food_real": "food_real",
            "y_food_real": "food_real",
            "x_real_target": "target",
            "y_real_target": "target"
        }
                
        transformed_item = {}
        # Process each key-value pair
        for key, value in item.items():
            if key in key_to_type:
                # Get transformed dataframe for this component
                if value is not None: # this is necessary when the food covariates are not used from the prediction window
                    _transformed_df_columns = self.inverse_transform_values(value, key_to_type[key])
                    transformed_item[key] = _transformed_df_columns
                else:
                    transformed_item[key] = None
            else:
                logger.warning(f"Key {key} not found in key_to_type")
        
        
        # Merge encoder values horizontally
        encoder_df = pd.concat([
            transformed_item["x_temporal_cat"],
            transformed_item["x_temporal_real"],
            transformed_item["x_user_cat"],
            transformed_item["x_user_real"],
            transformed_item["x_food_cat"],
            transformed_item["x_food_real"],
            transformed_item["x_real_target"]
        ], axis=1)
        decoder_df = pd.concat([
            transformed_item["y_temporal_cat"],
            transformed_item["y_temporal_real"],
            transformed_item["y_user_cat"],
            transformed_item["y_user_real"],
            transformed_item["y_food_cat"],
            transformed_item["y_food_real"],
            transformed_item["y_real_target"]
        ], axis=1)
        
        # Merger encoder and decoder vertically        
        aggregated_df = pd.concat([encoder_df, decoder_df], axis=0)

        return aggregated_df, encoder_df, decoder_df

    def save(self, fname: str) -> None:
        """
        Save dataset to disk

        Args:
            fname (str): filename to save to
        """
        torch.save(self, fname)

    @classmethod
    def load(cls, fname: str):
        """
        Load dataset from disk

        Args:
            fname (str): filename to load from

        Returns:
            TimeSeriesDataSet
        """
        obj = torch.load(fname)
        assert isinstance(obj, cls), f"Loaded file is not of class {cls}"
        return obj


class PPGRToMealGlucoseWrapper(Dataset):
    """
    A wrapper that converts a PPGRTimeSeriesDataset sample into the 6-tuple
    expected by the MealGlucoseForecastModel.
    """
    def __init__(self, ppgr_dataset: Dataset, cache_size: int = None):
        """
        Args:
            ppgr_dataset (Dataset): An instance of PPGRTimeSeriesDataset.
            cache_size (int, optional): Maximum number of items to cache. 
                                       If None, cache all items. Default: None.
        """
        self.ppgr_dataset = ppgr_dataset
        
        # First validate all requirements
        self._validate_dataset()
        
        # Then initialize all attributes
        self._initialize_attributes()
        
        self._setup_cache(cache_size=cache_size)
        self.warmup_cache()
    
    def _setup_cache(self, cache_size: int):
        # Initialize the cache
        self.cache = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Add a method to report cache stats
        self.total_requests = 0

    def _validate_dataset(self):
        """Validate that the dataset meets all requirements for this wrapper."""
        # Food-related validations
        assert hasattr(self.ppgr_dataset, "categorical_encoders"), "Dataset missing categorical_encoders attribute"
        assert "food_id" in self.ppgr_dataset.categorical_encoders, "food_id missing from categorical_encoders"
        assert hasattr(self.ppgr_dataset, "food_names"), "Dataset missing food_names attribute"
        assert hasattr(self.ppgr_dataset, "food_group_names"), "Dataset missing food_group_names attribute"
        
        # Nutrient-related validations
        assert hasattr(self.ppgr_dataset, "food_reals"), "Dataset missing food_reals attribute"

    def _initialize_attributes(self):
        """Initialize all required attributes after validation has passed."""
        # Initialize max meals
        if hasattr(self.ppgr_dataset, "max_meals_per_timestep"):
            self.max_meals = self.ppgr_dataset.max_meals_per_timestep
        else:
            self.max_meals = 11  # default fallback
            
        # Initialize food attributes
        self.food_names = self.ppgr_dataset.food_names
        self.food_group_names = self.ppgr_dataset.food_group_names
        self.num_foods = len(self.food_names)
            
        # Initialize nutrient dimensions
        self.num_nutrients = len(self.ppgr_dataset.food_reals)

    def __len__(self):
        return len(self.ppgr_dataset)

    def get_food_id_embeddings(self):
        """
        Get the food id embeddings layer to bootstrap the model's food id embeddings
        """
        return self.ppgr_dataset.get_food_id_embeddings_layer()

    def pad_or_truncate(self, x: torch.Tensor, target_size: int, dim: int = 1):
        """
        Pad (with zeros) or truncate the tensor along the specified dimension so that its size equals target_size.
        """
        current = x.size(dim)
        if current == target_size:
            return x
        elif current < target_size:
            pad_size = [0] * (2 * x.dim())
            # Pad at the end along the specified dimension by setting the right padding value
            # For a given dimension, the padding tuple is (left_pad, right_pad)
            # We want right_pad, so we set the right side: 2 * (x.dim() - dim - 1) + 1
            pad_size[2 * (x.dim() - dim - 1) + 1] = target_size - current
            return F.pad(x, pad=pad_size, mode='constant', value=0)
        else:
            slices = [slice(None)] * x.dim()
            slices[dim] = slice(0, target_size)
            return x[tuple(slices)]
        
    def __getitem__(self, idx):
        """Get a data sample from the dataset, using cache if available."""
        self.total_requests += 1
        
        # Check if the item is in the cache
        if idx in self.cache:
            self.cache_hits += 1
            return self.cache[idx]
        
        # If not in cache, compute the item
        self.cache_misses += 1
        item = self._compute_item(idx)
        
        # Add to cache if cache_size limit allows
        if self.cache_size is None or len(self.cache) < self.cache_size:
            self.cache[idx] = item
        elif self.cache_size > 0:
            # Optional: implement a replacement policy here if needed
            # Simple approach: replace a random item
            if len(self.cache) >= self.cache_size:
                # Only replace if we've reached capacity
                key_to_remove = random.choice(list(self.cache.keys()))
                del self.cache[key_to_remove]
                self.cache[idx] = item
        
        return item
    
    def _compute_item(self, idx):
        """Compute a data sample at idx from scratch."""
        # Original __getitem__ implementation
        item = self.ppgr_dataset[idx]
        device = self.ppgr_dataset.device
        
        # Store the actual encoder length for this sample
        encoder_length = torch.tensor(item["encoder_length"].item(), dtype=torch.int32).to(device)
        
        # ---------------------------
        # Past (encoder) side:
        # ---------------------------
        # Past glucose: assume "x_real_target" has shape [T_enc, 1] → squeeze last dimension.
        past_glucose = item["x_real_target"].squeeze(-1)  # shape: [T_enc]
        
        # Convert nested meal-level tensors to padded dense tensors.
        if item["x_food_cat"].size(0) == 0 or all(t.size(0) == 0 for t in item["x_food_cat"]):
            # Create zero tensors of desired shape
            # [T_enc, max_meals, num_features]
            x_length = item["x_real_target"].shape[0]
            num_cat_features = item["x_food_cat"][0].shape[-1]
            num_real_features = item["x_food_real"][0].shape[-1]
            x_food_cat = torch.zeros(x_length, self.max_meals, num_cat_features)
            x_food_real = torch.zeros(x_length, self.max_meals, num_real_features)
        else:
            x_food_cat = torch.nested.to_padded_tensor(item["x_food_cat"], padding=0)
            x_food_real = torch.nested.to_padded_tensor(item["x_food_real"], padding=0)
        
        # Get the index of the food_id column in the food_cat dictionary
        self.food_id_col_idx = self.ppgr_dataset.food_categoricals.index("food_id")
        
        past_meal_ids = x_food_cat[:, :, self.food_id_col_idx].long()  # shape: [T_enc, max_meals]
        past_meal_ids = self.pad_or_truncate(past_meal_ids, self.max_meals, dim=1)
        
        # For the meal macros, assume columns 2:5 hold the desired macronutrients.
        past_meal_macros = x_food_real[:, :, :].float()  # shape: [T_enc, max_meals, num_nutrients]
        past_meal_macros = self.pad_or_truncate(past_meal_macros, self.max_meals, dim=1)
        
        # ---------------------------
        # Future (prediction) side:
        # ---------------------------
        future_glucose = item["y_real_target"].squeeze(-1)  # shape: [T_pred]
        
        # Handle empty nested tensors for food data
        if item["y_food_cat"].size(0) == 0 or all(t.size(0) == 0 for t in item["y_food_cat"]):
            # Create zero tensors of desired shape
            # [T_pred, max_meals, num_features]
            y_length = item["y_real_target"].shape[0]
            num_cat_features = item["y_food_cat"][0].shape[-1]
            num_real_features = item["y_food_real"][0].shape[-1]
            y_food_cat = torch.zeros(y_length, self.max_meals, num_cat_features).to(device)
            y_food_real = torch.zeros(y_length, self.max_meals, num_real_features).to(device)
        else:
            y_food_cat = torch.nested.to_padded_tensor(item["y_food_cat"], padding=0)
            y_food_real = torch.nested.to_padded_tensor(item["y_food_real"], padding=0)
        
                
        future_meal_ids = y_food_cat[:, :, self.food_id_col_idx].long()  # shape: [T_pred, max_meals]
        future_meal_ids = self.pad_or_truncate(future_meal_ids, self.max_meals, dim=1)
        
        future_meal_macros = y_food_real[:, :, :].float()  # shape: [T_pred, max_meals, num_nutrients]
        future_meal_macros = self.pad_or_truncate(future_meal_macros, self.max_meals, dim=1)
        
        target_scales = item["target_scales"].squeeze() # TODO: check the provenance of this tensor and if we really need to do this squeeze here
    
        # Return the 6-tuple.
        return (past_glucose.float(),           # [T_enc]
                past_meal_ids,          # [T_enc, max_meals]
                past_meal_macros.float(),       # [T_enc, max_meals, num_nutrients]
                future_meal_ids,        # [T_pred, max_meals]
                future_meal_macros.float(),     # [T_pred, max_meals, num_nutrients]
                future_glucose.float(),         # [T_pred]
                target_scales,         # [2]
                encoder_length)         # [1]
    
    def get_cache_stats(self):
        """Return statistics about the cache performance."""
        if self.total_requests == 0:
            hit_rate = 0
        else:
            hit_rate = self.cache_hits / self.total_requests * 100
            
        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": f"{hit_rate:.2f}%",
            "cache_size": len(self.cache),
            "max_cache_size": self.cache_size
        }
    
    def clear_cache(self):
        """Clear the cache and reset statistics."""
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_requests = 0
        
    def warmup_cache(self, indices=None):
        """
        Warmup the cache with specific indices or the entire dataset.
        
        Args:
            indices (list, optional): List of indices to preload. If None, 
                                     preloads the entire dataset. Default: None.
        """
        if indices is None:
            indices = range(len(self))
            
        for i in tqdm(indices, desc="Preloading cache"):
            if i not in self.cache:
                if self.cache_size is None or len(self.cache) < self.cache_size:
                    self.cache[i] = self._compute_item(i)
                else:
                    # Cache is full
                    break

def meal_glucose_collate_fn(batch):
    """
    Custom collate function for the PPGRToMealGlucoseWrapper that handles variable-length
    encoder sequences with left padding.
    
    Args:
        batch: A list of tuples from PPGRToMealGlucoseWrapper.__getitem__
        
    Returns:
        A tuple of padded tensors ready for the model.
    """
    # Unpack the batch into separate lists
    past_glucose_list, past_meal_ids_list, past_meal_macros_list, future_meal_ids_list, \
    future_meal_macros_list, future_glucose_list, target_scales_list, encoder_lengths = zip(*batch)

    
    # Convert encoder_lengths to a tensor
    encoder_lengths = torch.stack(encoder_lengths)
    max_encoder_len = max(encoder_lengths).item()
    device = encoder_lengths[0].device
        
    # Apply left padding directly using pad_sequence
    past_glucose_batch = torch.nn.utils.rnn.pad_sequence(
        past_glucose_list, batch_first=True, padding_value=0.0, padding_side="left"
    )
    
    past_meal_ids_batch = torch.nn.utils.rnn.pad_sequence(
        past_meal_ids_list, batch_first=True, padding_value=0, padding_side="left"
    )
    
    past_meal_macros_batch = torch.nn.utils.rnn.pad_sequence(
        past_meal_macros_list, batch_first=True, padding_value=0, padding_side="left"
    )    
            
    # Future sequences should all have the same length, so we can just stack them
    future_glucose_batch = torch.stack(future_glucose_list)
    future_meal_ids_batch = torch.stack(future_meal_ids_list)
    future_meal_macros_batch = torch.stack(future_meal_macros_list)
    
    # Stack target scales
    target_scales_batch = torch.stack(target_scales_list)
    
    # Create encoder padding mask (1 = valid data, 0 = padding)
    # Create position indices along the time dimension
    positions = torch.arange(max_encoder_len).to(device)
    
    # For left padding: mask positions where position_index >= (max_len - sequence_length)
    # This creates a [batch_size, max_encoder_len] boolean mask
    start_positions = max_encoder_len - encoder_lengths.unsqueeze(1)
    encoder_padding_mask = positions.unsqueeze(0) >= start_positions
    
    
    _response = {
        "past_glucose": past_glucose_batch, 
        "past_meal_ids": past_meal_ids_batch, 
        "past_meal_macros": past_meal_macros_batch, 
        "future_meal_ids": future_meal_ids_batch, 
        "future_meal_macros": future_meal_macros_batch, 
        "future_glucose": future_glucose_batch, 
        "target_scales": target_scales_batch,
        "encoder_lengths": encoder_lengths,
        "encoder_padding_mask": encoder_padding_mask
    }
    return _response
    

# -----------------------------------------------------------------------------
# create_cached_dataset function that caches all splits + encoders/scalers
# -----------------------------------------------------------------------------
def create_cached_dataset(
    dataset_version,
    debug_mode,
    validation_percentage,
    test_percentage,
    min_encoder_length,
    max_encoder_length,
    prediction_length,
    encoder_length_randomization,
    is_food_anchored,
    sliding_window_stride,
    use_meal_level_food_covariates,
    use_microbiome_embeddings,
    use_bootstraped_food_embeddings,
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
        "max_encoder_length": max_encoder_length,
        "prediction_length": prediction_length,
        "encoder_length_randomization": encoder_length_randomization,
        "is_food_anchored": is_food_anchored,
        "sliding_window_stride": sliding_window_stride,
        "use_meal_level_food_covariates": use_meal_level_food_covariates,
        "use_microbiome_embeddings": use_microbiome_embeddings,
        "use_bootstraped_food_embeddings": use_bootstraped_food_embeddings,
        "group_by_columns": group_by_columns,
        "temporal_categoricals": temporal_categoricals,
        "temporal_reals": temporal_reals,
        "user_static_categoricals": user_static_categoricals,
        "user_static_reals": user_static_reals,
        "food_categoricals": food_categoricals,
        "food_reals": food_reals,
        "targets": targets,
        "device": "cpu" # the dataset always runs on CPU
    }
    
    logger.info(f"Loading dataset with the following config: {config}")
    

    # 2) Compute a unique hash from the config
    import pickle  # Only used for hashing the config; caching itself is now handled by torch.save
    config_bytes = pickle.dumps(config)
    config_hash = hashlib.md5(config_bytes).hexdigest()
    cache_file = os.path.join(cache_dir, f"{config_hash}_all_splits.pt")  # Updated extension to .pt

    # 3) If cache exists and caching is enabled, load & return immediately using torch.load
    if use_cache and os.path.exists(cache_file):
        logger.info(f"[CACHE-HIT] Loading pipeline from: {cache_file}")
        cached_data = torch.load(cache_file, weights_only=False)
        return (
            cached_data["training_dataset"],
            cached_data["validation_dataset"],
            cached_data["test_dataset"],
            cached_data["categorical_encoders"],
            cached_data["continuous_scalers"],
        )

    # 4) Otherwise, proceed to load & build everything
    logger.info("[CACHE-MISS] Building dataset from scratch...")
    os.makedirs(cache_dir, exist_ok=True)

    # Load base dataframes
    ppgr_df, users_demographics_df, dishes_df, microbiome_embeddings_df, food_embeddings_df = load_dataframe(
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
    
    if encoder_length_randomization == "training_only":
        disable_encoder_length_randomization = {
            "training_set": False,
            "validation_set": True,
            "test_set": True
        }
    elif encoder_length_randomization == "all_sets":
        disable_encoder_length_randomization = {
            "training_set": False,
            "validation_set": False,
            "test_set": False
        }
    elif encoder_length_randomization == "none":
        disable_encoder_length_randomization = {
            "training_set": True,
            "validation_set": True,
            "test_set": True
        }
    else:
        raise ValueError(f"Invalid encoder_length_randomization: {encoder_length_randomization}")

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
        max_encoder_length=max_encoder_length,
        prediction_length=prediction_length,
        disable_encoder_length_randomization=disable_encoder_length_randomization["training_set"],
        use_food_covariates_from_prediction_window=True,
        use_meal_level_food_covariates=use_meal_level_food_covariates,
        use_microbiome_embeddings=use_microbiome_embeddings,
        microbiome_embeddings_df=microbiome_embeddings_df,
        use_bootstraped_food_embeddings=use_bootstraped_food_embeddings,
        food_embeddings_df=food_embeddings_df,
        temporal_categoricals=temporal_categoricals,
        temporal_reals=temporal_reals,
        user_static_categoricals=user_static_categoricals,
        user_static_reals=user_static_reals,
        food_categoricals=food_categoricals,
        food_reals=food_reals,
        categorical_encoders=categorical_encoders,
        continuous_scalers=continuous_scalers,
        device="cpu"
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
        max_encoder_length=max_encoder_length,
        prediction_length=prediction_length,
        disable_encoder_length_randomization=disable_encoder_length_randomization["validation_set"],
        use_food_covariates_from_prediction_window=True,
        use_meal_level_food_covariates=use_meal_level_food_covariates,
        use_microbiome_embeddings=use_microbiome_embeddings,
        microbiome_embeddings_df=microbiome_embeddings_df,
        use_bootstraped_food_embeddings=use_bootstraped_food_embeddings,
        food_embeddings_df=food_embeddings_df,        
        temporal_categoricals=temporal_categoricals,
        temporal_reals=temporal_reals,
        user_static_categoricals=user_static_categoricals,
        user_static_reals=user_static_reals,
        food_categoricals=food_categoricals,
        food_reals=food_reals,
        categorical_encoders=categorical_encoders,
        continuous_scalers=continuous_scalers,
        device="cpu"
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
        max_encoder_length=max_encoder_length,
        prediction_length=prediction_length,
        disable_encoder_length_randomization=disable_encoder_length_randomization["test_set"],
        use_food_covariates_from_prediction_window=True,
        use_meal_level_food_covariates=use_meal_level_food_covariates,
        use_microbiome_embeddings=use_microbiome_embeddings,
        microbiome_embeddings_df=microbiome_embeddings_df,
        use_bootstraped_food_embeddings=use_bootstraped_food_embeddings,
        food_embeddings_df=food_embeddings_df,        
        temporal_categoricals=temporal_categoricals,
        temporal_reals=temporal_reals,
        user_static_categoricals=user_static_categoricals,
        user_static_reals=user_static_reals,
        food_categoricals=food_categoricals,
        food_reals=food_reals,
        categorical_encoders=categorical_encoders,
        continuous_scalers=continuous_scalers,
        device="cpu"
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
    logger.info(f"Dataset pipeline built and saved to cache: {cache_file}")
    return (
        wrapped_training_dataset,
        wrapped_validation_dataset,
        wrapped_test_dataset,
        categorical_encoders,
        continuous_scalers,
    )



if __name__ == "__main__":

    # dataset_version = "v0.4"
    # debug_mode = True

    # min_encoder_length = 8 * 4 # 8 hours with 4 timepoints per hour
    # prediction_length = 2 * 4 # 2 hours with 4 timepoints per hour
    
    # validation_percentage = 0.2
    # test_percentage = 0.2


    # is_food_anchored = True # When False, its the standard sliding window timeseries, but when True, every timeseries will have a food intake row at the last item of the context window
    # sliding_window_stride = None # This has to change everytime is_food_anchored is changed
    
    # use_meal_level_food_covariates = True
    
    # use_microbiome_embeddings = True
    
    
    # # Unique Grouping Column
    # group_by_columns = ["timeseries_block_id"]

    # # User 
    # user_static_categoricals = ["user_id", "user__edu_degree", "user__income", "user__household_desc", "user__job_status", "user__smoking", "user__health_state", "user__physical_activities_frequency"]
    # user_static_reals = ["user__age", "user__weight", "user__height", "user__bmi", "user__general_hunger_level", "user__morning_hunger_level", "user__mid_hunger_level", "user__evening_hunger_level"]

    # # Food Covariates
    # food_categoricals = []
    # if use_meal_level_food_covariates:
    #     food_categoricals = ['food__food_group_cname', 'food_id']
    # else:
    #     food_categoricals = [   'food__vegetables_fruits', 'food__grains_potatoes_pulses', 'food__unclassified',
    #                             'food__non_alcoholic_beverages', 'food__dairy_products_meat_fish_eggs_tofu',
    #                             'food__sweets_salty_snacks_alcohol', 'food__oils_fats_nuts'] 
    
    # food_reals = ['food__eaten_quantity_in_gram', 'food__energy_kcal_eaten',
    #     'food__carb_eaten', 'food__fat_eaten', 'food__protein_eaten',
    #     'food__fiber_eaten', 'food__alcohol_eaten']

    # # Temporal Covariates
    # temporal_categoricals = ["loc_eaten_dow", "loc_eaten_dow_type", "loc_eaten_season"]
    # temporal_reals = ["loc_eaten_hour"]

    # # Targets
    # targets = ["val"]

    # main_df_scaled_all_categorical_columns = user_static_categoricals + food_categoricals + temporal_categoricals
    # main_df_scaled_all_real_columns = user_static_reals + food_reals + temporal_reals + targets
    
    
    
    # # Load the data frames
    # ppgr_df, users_demographics_df, dishes_df, microbiome_embeddings_df = load_dataframe(dataset_version, debug_mode)

    # # Split the data frames into training, validation and test sets
    # training_df, validation_df, test_df = split_timeseries_df_based_on_food_intake_rows(ppgr_df, validation_percentage=validation_percentage, test_percentage=test_percentage)
    
    # # Validate the data frames
    # ppgr_df, users_demographics_df, dishes_df = enforce_column_types(  ppgr_df, 
    #                                                         users_demographics_df, 
    #                                                         dishes_df,
    #                                                         main_df_scaled_all_categorical_columns,
    #                                                         main_df_scaled_all_real_columns)

    # # Setup the scalers and encoders
    # categorical_encoders, continuous_scalers = setup_scalers_and_encoders(
    #     ppgr_df = ppgr_df,
    #     training_df = training_df,
    #     users_demographics_df = users_demographics_df,
    #     dishes_df=dishes_df,
    #     categorical_columns = main_df_scaled_all_categorical_columns,
    #     real_columns = main_df_scaled_all_real_columns,
    #     use_meal_level_food_covariates = use_meal_level_food_covariates # This determines which data to fit the encoders on
    # ) # Note: the encoders are fit on the full ppgr_df, and the scalers are fit on the training_df

    # # Create the training dataset
    # training_dataset = PPGRTimeSeriesDataset(ppgr_df = training_df, 
    #                                         user_demographics_df = users_demographics_df,
    #                                         dishes_df = dishes_df,
    #                                         time_idx = "read_at",
    #                                         target_columns = ["val"],                                                                                    
    #                                         group_by_columns = ["timeseries_block_id"],

    #                                         is_food_anchored = is_food_anchored, # When False, its the standard sliding window timeseries, but when True, every timeseries will have a food intake row at the last item of the context window
    #                                         sliding_window_stride = sliding_window_stride, # This has to change everytime is_food_anchored is changed

    #                                         min_encoder_length = min_encoder_length, # 8 hours with 4 timepoints per hour
    #                                         prediction_length = prediction_length, # 2 hours with 4 timepoints per hour
                                            
    #                                         use_food_covariates_from_prediction_window = True,
                                            
    #                                         use_meal_level_food_covariates = use_meal_level_food_covariates, # This uses the granular meal level food covariates instead of the food item level covariates
                                            
    #                                         use_microbiome_embeddings = use_microbiome_embeddings,
    #                                         microbiome_embeddings_df = microbiome_embeddings_df,
                                            
    #                                         temporal_categoricals = temporal_categoricals,
    #                                         temporal_reals = temporal_reals,

    #                                         user_static_categoricals = user_static_categoricals,
    #                                         user_static_reals = user_static_reals,
                                            
    #                                         food_categoricals = food_categoricals,
    #                                         food_reals = food_reals,
                                            
    #                                         categorical_encoders = categorical_encoders,
    #                                         continuous_scalers = continuous_scalers,
    #                                         )

    # print(f"Length of training dataset: {len(training_dataset)}")

    
    # for k in range(5):
    #     print("-"*50)
    #     print(f"Data point {k}")
    #     print("-"*50)
    #     data = training_dataset[k]
    #     print(data)
    #     print(training_dataset.get_item_from_df(k))
    #     break
    #     # aggregated_df, encoder_df, decoder_df = training_dataset.inverse_transform_item(data)
    #     # print("=="*100)
    #     # print(encoder_df)
    #     # print("=="*100)
    #     # print(decoder_df)
        
    
    
    
    
    
    # display(encoder_df)
    # display(decoder_df)

    # for data in tqdm(training_dataset):
    #     print(data["metadata"])
    #     # aggregated_df, encoder_df, decoder_df = training_dataset.inverse_transform_item(data)
    #     # print(encoder_df)
    #     # print(decoder_df)
    #     # break
    #     # display(decoder_df)
    #     break
            
    
    
    # train_loader = DataLoader(
    #     training_dataset,   # your PPGRTimeSeriesDataset instance
    #     batch_size=512,
    #     shuffle=True,
    #     num_workers=1, # When using the data directly on GPU, ensure num_workers is 1
    #     collate_fn=ppgr_collate_fn
    # )
    
    # # print(training_dataset[0])

    # # ppgr_collate_fn([training_dataset[0], training_dataset[1], training_dataset[2]])
    
    # # Example: Iterate through one batch
    # for batch in tqdm(train_loader):
    #     # Now batch is a dictionary where variable-length sequences have been padded.
    #     # print("x_temporal_cat shape:", batch["x_temporal_cat"].shape)
    #     # print("relative_time_idx shape:", batch.get("relative_time_idx", None))
    #     # Process your batch...
    #     # breakpoint()
    #     breakpoint()
    #     pass

    # wrapped_dataset = PPGRToMealGlucoseWrapper(training_dataset)
    # breakpoint()
    
    
    from main import ExperimentConfig, get_dataloaders
    
    config = ExperimentConfig(
        dataloader_num_workers=7,
        debug_mode=True,
        dataset_version="v0.5",
        use_bootstraped_food_embeddings=True,
        use_cache = False
    )

    (training_dataset, validation_dataset, test_dataset, categorical_encoders, continuous_scalers) = create_cached_dataset(
        dataset_version=config.dataset_version,
        debug_mode=config.debug_mode,
        validation_percentage=config.validation_percentage,
        test_percentage=config.test_percentage,
        min_encoder_length=config.min_encoder_length,
        max_encoder_length=config.max_encoder_length,
        prediction_length=config.prediction_length,
        encoder_length_randomization=config.encoder_length_randomization,
        is_food_anchored=config.is_food_anchored,
        sliding_window_stride=config.sliding_window_stride,
        use_meal_level_food_covariates=config.use_meal_level_food_covariates,
        use_microbiome_embeddings=config.use_microbiome_embeddings,
        use_bootstraped_food_embeddings=config.use_bootstraped_food_embeddings,
        group_by_columns=config.group_by_columns,
        temporal_categoricals=config.temporal_categoricals,
        temporal_reals=config.temporal_reals,
        user_static_categoricals=config.user_static_categoricals,
        user_static_reals=config.user_static_reals,
        food_categoricals=config.food_categoricals,
        food_reals=config.food_reals,
        targets=config.targets,
        cache_dir=config.cache_dir,
        use_cache=config.use_cache,
    )
    
    train_loader = DataLoader(
        training_dataset, 
        batch_size=config.batch_size, 
        num_workers=config.dataloader_num_workers, 
        pin_memory=True, 
        persistent_workers=True,
        shuffle=True,
        collate_fn=meal_glucose_collate_fn
    )    
    
    # for _ in tqdm(train_loader):
    #     pass
    
    batch = meal_glucose_collate_fn([
        training_dataset[x] for x in range(512)
    ])
    
    # breakpoint()
    
    # print(batch)
    
    
    # train_loader, val_loader, test_loader = get_dataloaders(config)
    # train_loader.dataset[0]
    
    
    
    