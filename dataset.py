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
                        max_encoder_length: int = False, # context window
                        prediction_length: int = 3 * 4, # prediction window
                        
                        use_food_covariates_from_prediction_window: bool = True, # decides if the food covariates are used from the prediction window as well
                        
                        use_meal_level_food_covariates: bool = True, # decides if the meal level food covariates are used (instead of the aggregated onces)
                        max_meals_per_timestep: int = 11, # maximum number of meals per timestep, the rest are discarded
                        
                        use_microbiome_embeddings: bool = True, # decides if the microbiome embeddings are used
                        microbiome_embeddings_df: Optional[pd.DataFrame] = None, # the microbiome embeddings dataframe
                        
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
                
        self.use_food_covariates_from_prediction_window = use_food_covariates_from_prediction_window
        
        self.use_meal_level_food_covariates = use_meal_level_food_covariates
        self.max_meals_per_timestep = max_meals_per_timestep
        
        self.use_microbiome_embeddings = use_microbiome_embeddings
        self.microbiome_embeddings_df = microbiome_embeddings_df
        
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
        assert self.max_encoder_length > 0 or self.max_encoder_length is False, "Maximum encoder length must be greater than 0 or set to False to disable randomization"
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
        
        # 1.1. Prepare the microbiome embeddings
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
        self.target_scales = torch.tensor(np.array(self.target_scales)).to(self.device)
        
        # reset the index so we can start from index 0 and maintain a continuous index that aligns with the torch indexing system
        self.df_scaled = self.df_scaled.reset_index(drop=True)
        
        # Create a tensor optimized version of the dataframe with only the relevant columns
        self.df_scaled_tensor = torch.tensor(self.df_scaled[self.main_df_all_columns].to_numpy()).to(self.device)
        
        # NOTE: It is important that the indices of both df_scaled and df_scaled_tensor align, 
        # which is why we did the reset_index just before
        
        ### 3. Handle the scaling of the dishes_df if required
        if self.use_meal_level_food_covariates:
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
            for row_idx in range(block_start, block_end + 1, self.sliding_window_stride):
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
        
        START_TOKEN = 0 # TODO: choose appropriate values later for this. Too brainfucked to think right now.
        
        # Create start/empty token tensors (using -100 to distinguish from regular -1 padding)
        start_cat_tensor = torch.full((1, num_cat_features), START_TOKEN, dtype=torch.int32, device=self.device)
        start_real_tensor = torch.full((1, num_real_features), START_TOKEN, dtype=torch.float32, device=self.device)
        
        # TODO: Setup sensible values for the start token
        max_meals_per_timestep = self.max_meals_per_timestep
        
        dish_tensors_recorded = []
        # Use range(start, end + 1) to make end inclusive
        for slice_row_idx in range(slice_start, slice_end + 1):
            dish_idxs_for_this_row = self.ppgr_df_row_idx_to_dishes_df_idxs.get(slice_row_idx, None)
            
            if dish_idxs_for_this_row is None or len(dish_idxs_for_this_row) == 0:
                # If no dishes, just use the start token
                dish_tensors_cat_for_this_slice.append(start_cat_tensor)
                dish_tensors_real_for_this_slice.append(start_real_tensor)
                
                dish_tensors_recorded.append(1) # adding 1 as we have a start token
            else:
                # If dishes exist, concatenate start token with dish tensors
                dish_tensors_for_this_row = self.dishes_df_scaled_tensor[dish_idxs_for_this_row]
                
                # Only use up to max_dishes_per_row 
                cat_tensor = torch.cat([
                    start_cat_tensor,
                    dish_tensors_for_this_row[:max_meals_per_timestep, self.food_categorical_col_idx]
                ], dim=0)
                real_tensor = torch.cat([
                    start_real_tensor,
                    dish_tensors_for_this_row[:max_meals_per_timestep, self.food_real_col_idx]
                ], dim=0)
                
                dish_tensors_cat_for_this_slice.append(cat_tensor)
                dish_tensors_real_for_this_slice.append(real_tensor)
                
                dish_tensors_recorded.append( len(dish_idxs_for_this_row) + 1) # +1 for the start token
        
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
    A wrapper that converts a PPGRTimeSeriesDataset sample (returning a dictionary)
    into the 6-tuple expected by the MealGlucoseForecastModel:
    
      (past_glucose, past_meal_ids, past_meal_macros, future_meal_ids, future_meal_macros, future_glucose)
    
    Additionally, it estimates:
      - The maximum number of meals per timestep (`max_meals`)
      - The total number of food IDs (`num_foods`), using the categorical encoder for "food_id"
      - The number of nutrient dimensions (`num_nutrients`) from the underlying dataset's `food_reals` list.
      
    Assumptions:
      - The PPGR sample uses meal-level food covariates.
      - In the food categorical tensor, column index 1 holds the food ID.
      - In the food real tensor, columns 2:5 (i.e. indices 2,3,4) are used for macronutrients.
        (You can change these slices as needed, but `num_nutrients` will be derived from the full list
         available in `ppgr_dataset.food_reals`.)
    """
    def __init__(self, ppgr_dataset: Dataset, max_meals: int = None, num_foods: int = None):
        """
        Args:
            ppgr_dataset (Dataset): An instance of PPGRTimeSeriesDataset.
            max_meals (int, optional): Maximum number of meals per timestep. If None, the wrapper
                                       will use ppgr_dataset.max_meals_per_timestep if available;
                                       otherwise, it defaults to 11.
            num_foods (int, optional): Total number of distinct food IDs. If None, the wrapper will
                                       attempt to extract it from ppgr_dataset.categorical_encoders["food_id"].
        """
        self.ppgr_dataset = ppgr_dataset
        
        # Determine max_meals from the dataset attribute if not explicitly provided
        if max_meals is None:
            if hasattr(ppgr_dataset, "max_meals_per_timestep"):
                self.max_meals = ppgr_dataset.max_meals_per_timestep
            else:
                self.max_meals = 11  # default fallback
        else:
            self.max_meals = max_meals

        # Determine num_foods from the food_id encoder if available
        if num_foods is None:
            if hasattr(ppgr_dataset, "categorical_encoders") and "food_id" in ppgr_dataset.categorical_encoders:
                food_encoder = ppgr_dataset.categorical_encoders["food_id"]
                # +1 for padding (assumed index 0)
                self.num_foods = len(food_encoder.classes_) + 1
            else:
                self.num_foods = None
        else:
            self.num_foods = num_foods

        # Determine the number of nutrient dimensions from the dataset's food_reals list.
        if hasattr(ppgr_dataset, "food_reals"):
            self.num_nutrients = len(ppgr_dataset.food_reals)
        else:
            self.num_nutrients = None

    def __len__(self):
        return len(self.ppgr_dataset)

    def pad_or_truncate(self, x: torch.Tensor, target_size: int, dim: int = 1):
        """
        Pad (with zeros) or truncate the tensor along the specified dimension so that its size equals target_size.
        """
        current = x.size(dim)
        if current == target_size:
            return x
        elif current < target_size:
            pad_size = [0] * (2 * x.dim())
            # Pad at the end along the specified dimension
            pad_size[2 * (x.dim() - dim - 1)] = target_size - current
            return F.pad(x, pad=pad_size, mode='constant', value=0)
        else:
            slices = [slice(None)] * x.dim()
            slices[dim] = slice(0, target_size)
            return x[tuple(slices)]
        
    def __getitem__(self, idx):
        # Retrieve the original PPGR sample (a dict)
        item = self.ppgr_dataset[idx]
        
        # ---------------------------
        # Past (encoder) side:
        # ---------------------------
        # Past glucose: assume "x_real_target" has shape [T_enc, 1] → squeeze last dimension.
        past_glucose = item["x_real_target"].squeeze(-1)  # shape: [T_enc]
        
        # Convert nested meal-level tensors to padded dense tensors.
        x_food_cat = torch.nested.to_padded_tensor(item["x_food_cat"], padding=0)
        x_food_real = torch.nested.to_padded_tensor(item["x_food_real"], padding=0)
        
        # Assume the food ID is stored in column index 1.
        past_meal_ids = x_food_cat[:, :, 1].long()  # shape: [T_enc, N]
        past_meal_ids = self.pad_or_truncate(past_meal_ids, self.max_meals, dim=1)
        
        # For the meal macros, assume columns 2:5 hold the desired macronutrients.
        past_meal_macros = x_food_real[:, :, :].float()  # shape: [T_enc, N, num_nutrients]
        past_meal_macros = self.pad_or_truncate(past_meal_macros, self.max_meals, dim=1)
        
        # ---------------------------
        # Future (prediction) side:
        # ---------------------------
        future_glucose = item["y_real_target"].squeeze(-1)  # shape: [T_pred]
        y_food_cat = torch.nested.to_padded_tensor(item["y_food_cat"], padding=0)
        y_food_real = torch.nested.to_padded_tensor(item["y_food_real"], padding=0)
        
        future_meal_ids = y_food_cat[:, :, 1].long()  # shape: [T_pred, N]
        future_meal_ids = self.pad_or_truncate(future_meal_ids, self.max_meals, dim=1)
        
        future_meal_macros = y_food_real[:, :, :].float()  # shape: [T_pred, N, num_nutrients]
        future_meal_macros = self.pad_or_truncate(future_meal_macros, self.max_meals, dim=1)
        
        target_scales = item["target_scales"]
        
        # Return the 6-tuple.
        return (past_glucose.float(),           # [T_enc]
                past_meal_ids,          # [T_enc, max_meals]
                past_meal_macros.float(),       # [T_enc, max_meals, num_nutrients]
                future_meal_ids,        # [T_pred, max_meals]
                future_meal_macros.float(),     # [T_pred, max_meals, num_nutrients]
                future_glucose.float(),         # [T_pred]
                target_scales)         # [2]

if __name__ == "__main__":

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
    training_dataset = PPGRTimeSeriesDataset(ppgr_df = training_df, 
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

    print(f"Length of training dataset: {len(training_dataset)}")

    
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




    wrapped_dataset = PPGRToMealGlucoseWrapper(training_dataset)
    breakpoint()