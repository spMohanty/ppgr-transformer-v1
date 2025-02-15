import pandas as pd
from pytorch_forecasting.data import TimeSeriesDataSet
from loguru import logger

from sklearn.preprocessing import StandardScaler

from typing import Tuple, Optional
from typing import Dict, List


from tqdm import tqdm

from utils import load_dataframe, enforce_column_types, setup_scalers_and_encoders

from pytorch_forecasting.data.encoders import (
    EncoderNormalizer,
    GroupNormalizer,
    MultiNormalizer,
    NaNLabelEncoder,
    TorchNormalizer,
)


import numpy as np

import torch
from torch.utils.data import Dataset


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
    
    # sort dataframe by user_id and read_at to ensure that the data is in the correct order
    df = df.sort_values(by=["user_id", "read_at"]).reset_index(drop=True)
    
    for user_id, group in tqdm(df.groupby("user_id"), total=len(df["user_id"].unique())):
        # Get indices where food intake occurred
        food_intake_indices = group[group["food_intake_row"] == 1].index.tolist()
        n_samples = len(food_intake_indices)
        
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
        assert validation_data_slices["read_at"].max() > training_data_slices["read_at"].max()
        assert test_data_slices["read_at"].max() > validation_data_slices["read_at"].max()
        
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
                        is_food_anchored: bool = True, # decides if the timeseries is food anchored or not
                        time_idx: str = "time_idx", # Unique integral column in each timeseries for each datapoint (should be in the same order as read_at) 

                        target_columns: List[str] = ["val"], # These are the columns that are the targets (should also be provided in the time_varying_unknown_reals
                        group_by_columns: List[str] = ["timeseries_block_id"], # Decides what the timeseries grouping/boundaries are
                        
                        min_encoder_length: int = 8 * 4, # context window
                        max_encoder_length: int = 12 * 4, # context window
                        prediction_length: int = 3 * 4, # prediction window
                        
                        use_food_covariates_from_prediction_window: bool = True, # decides if the food covariates are used from the prediction window as well
                        
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
                        
                        add_relative_time_idx: bool = True,
                        device: torch.device = torch.device("cpu")
                    ):
        self.ppgr_df = ppgr_df
        self.user_demographics_df = user_demographics_df
        
        self.is_food_anchored = is_food_anchored
        self.time_idx = time_idx
        
        self.target_columns = target_columns
        
        self.min_encoder_length = min_encoder_length
        self.max_encoder_length = max_encoder_length
        self.prediction_length = prediction_length
        
        self.use_food_covariates_from_prediction_window = use_food_covariates_from_prediction_window
        
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
        
        self.add_relative_time_idx = add_relative_time_idx
        self.device = device
        
        self.timeseries_slices_indices = [] # Will house a list of PPGRTimeSeriesSliceMetadata that will be referenced in __getitem__

        self.prepare_data()
    
    def prepare_data(self):
        
        # 0. Merge the ppgr_df and user_demographics_df
        self.merge_ppgr_and_user_demographics()
        
        # 0.1. Prepare the all dataset columns
        self.prepare_all_dataset_columns()                
                
        # 1. Scale and CatEmbed the dataset df
        self.prepare_scaled_and_catembed_dataset()
        
        # 1.1. Prepare the microbiome embeddings
        self.prepare_microbiome_embeddings()
        
        # 2. Sort by group_by_columns and time_idx
        self.ppgr_df = self.ppgr_df.sort_values(by=self.group_by_columns + [self.time_idx])
        
        if self.is_food_anchored:
            # 3.1. Prepare the data for food anchored timeseries
            self.prepare_food_anchored_data()
        else:
            # 3.2 Prepare the data for the standard sliding window timeseries    
            self.prepare_sliding_window_data()
    
    def merge_ppgr_and_user_demographics(self):
        self.ppgr_df = self.ppgr_df.merge(self.user_demographics_df, on="user_id", how="left")
    
    def prepare_all_dataset_columns(self):
        # Gather all columns in a single place
        self.all_categorical_columns = sorted(self.user_static_categoricals + self.food_categoricals + self.temporal_categoricals)
        self.all_real_columns = sorted(self.user_static_reals + self.food_reals + self.temporal_reals + self.target_columns)
        self.all_columns = self.all_categorical_columns + self.all_real_columns

        # Gather the indices of the individual columns for easy access using tensor indexing
        # Temporal Covariates   
        self.temporal_categorical_col_idx = [self.all_columns.index(col) for col in self.temporal_categoricals]
        self.temporal_real_col_idx = [self.all_columns.index(col) for col in self.temporal_reals]        
        
        # User Static Covariates
        self.user_static_categorical_col_idx = [self.all_columns.index(col) for col in self.user_static_categoricals]
        self.user_static_real_col_idx = [self.all_columns.index(col) for col in self.user_static_reals]
        
        # Food Categorical Columns
        self.food_categorical_col_idx = [self.all_columns.index(col) for col in self.food_categoricals]
        self.food_real_col_idx = [self.all_columns.index(col) for col in self.food_reals]
                
        # Target Columns
        self.target_col_idx = [self.all_columns.index(col) for col in self.target_columns]
    
    
    def prepare_scaled_and_catembed_dataset(self):
        """
        Prepares a scaled + categorical embedded version of the dataset that can be readily accessed in __getitem__
        """
        self.df_scaled = self.ppgr_df.copy() # NOTE: This should be the merged dataframe by now
        
        assert (self.df_scaled.index == self.ppgr_df.index).all()
        
        # Scale the continuous variables
        for col in self.continuous_scalers:
            logger.debug(f"Scaling {col}")
            self.df_scaled[col] = self.continuous_scalers[col].transform(
                self.df_scaled[col].to_numpy().reshape(-1, 1)
            )
            
        # Encode the categorical variables
        for col in self.categorical_encoders:
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
        self.df_scaled_tensor = torch.tensor(self.df_scaled[self.all_columns].to_numpy()).to(self.device)
        
        # NOTE: It is important that the indices of both df_scaled and df_scaled_tensor align, 
        # which is why we did the reset_index just before
        
        
    def prepare_microbiome_embeddings(self):
        """
        microbiome embeddings expect that all the user_ids are present in the microbiome embeddings dataframe
        and that the data is already normalized
        """
        self.microbiome_embeddings_df = self.microbiome_embeddings_df.sort_index()
        assert set(self.ppgr_df["user_id"].astype(int).unique()) == set(self.microbiome_embeddings_df.index), "All user_ids in the ppgr_df must be present in the microbiome_embeddings_df, and vice versa"
        
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
            for row_idx, row in group.iterrows():                
                # 1. Identify the start and end of the context and prediction window
                slice_start = row_idx - self.min_encoder_length
                slice_end = row_idx + self.prediction_length
                
                # 2. Ignore the fringe points where the slice is not fully within the block
                if slice_start < block_start or slice_end > block_end: continue 
                
                print(f"slice_start: {slice_start}, slice_end: {slice_end}")
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
            food_intake_mask.iloc[:self.max_encoder_length] = False
            food_intake_mask.iloc[-self.prediction_length:] = False
            
            # 6. Get all the rows that have enough past and future data
            food_intake_rows = group[food_intake_mask]

            # 7. Iterate over all the food intake rows
            for row_idx, _ in food_intake_rows.iterrows():
                # 8. For each food intake row, identify the start and end of the context and prediction window
                slice_start = row_idx - self.min_encoder_length
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
    
    def __len__(self):
        return len(self.timeseries_slices_indices)
    
    def _get_slice_start_and_end(self, slice_metadata: PPGRTimeSeriesSliceMetadata):
        
        # Get block boundaries and anchor point
        block_start = slice_metadata.block_start
        slice_anchor_row_idx = slice_metadata.anchor_row_idx
        slice_end = slice_metadata.slice_end
        
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
        slice_start = slice_anchor_row_idx - encoder_length        
        slice_end = slice_metadata.slice_end
        
        return slice_start, slice_end, encoder_length
    
    def __getitem__(self, idx):
        slice_metadata = self.timeseries_slices_indices[idx]
        

        slice_start, slice_end, encoder_length = self._get_slice_start_and_end(slice_metadata) # Takes care of randomizing the encoder length between min and max encoder lengths
        prediction_length = slice_end - slice_metadata.anchor_row_idx 
        
        # Get the slice from the dataframe
        data_slice_tensor = self.df_scaled_tensor[slice_start:slice_end, :].clone() # [T, C] where T is the number of rows in the slice and C is the number of all relevant columns
                
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
            y_food_cat = data_slice_prediction_window_tensor[:, self.food_categorical_col_idx]
            y_food_real = data_slice_prediction_window_tensor[:, self.food_real_col_idx]
        else:
            y_food_cat = None
            y_food_real = None

        # Target Variables
        y_real_target = data_slice_prediction_window_tensor[:, self.target_col_idx]
        target_scales = self.target_scales # mean, std : This can be varying later if we want to normalize by user_id or other columns

        # Calculate relative time indices
        # Encoder window: [-encoder_length, ..., 0]
        # Prediction window: [0, 1, ..., prediction_length-1]
        encoder_time_idx = torch.arange(-encoder_length + 1, 0+1).to(self.device) # the current anchor needs to be at index 0
        prediction_time_idx = torch.arange(1, prediction_length+1).to(self.device)
        relative_time_idx = torch.cat([encoder_time_idx, prediction_time_idx])

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
            
            # Metadata about the slice
            
            metadata = dict(
                encoder_length = encoder_length, # scalar 
                prediction_length = prediction_length,
                categorical_encoders = self.categorical_encoders,
                continuous_scalers = self.continuous_scalers,
                all_columns = self.all_columns,
                temporal_categoricals = self.temporal_categoricals,
                temporal_reals = self.temporal_reals,
                user_static_categoricals = self.user_static_categoricals,
                user_static_reals = self.user_static_reals,
                food_categoricals = self.food_categoricals,
                food_reals = self.food_reals,
                target_columns = self.target_columns,
            )
        )
        
        if self.add_relative_time_idx:
            _response["relative_time_idx"] = relative_time_idx
        
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
        transformed_dfs = []
        
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
        
        relative_time_idx = item["relative_time_idx"]
        
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
        if self.add_relative_time_idx:
            aggregated_df["relative_time_idx"] = relative_time_idx

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



if __name__ == "__main__":

    dataset_version = "v0.4"
    debug_mode = True

    max_encoder_length = 48*4
    max_prediction_length = 2*4
    validation_percentage = 0.2
    test_percentage = 0.2


    # Unique Grouping Column
    group_by_columns = ["timeseries_block_id"]

    # User 
    user_static_categoricals = ["user_id", "user__edu_degree", "user__income", "user__household_desc", "user__job_status", "user__smoking", "user__health_state", "user__physical_activities_frequency"]
    user_static_reals = ["user__age", "user__weight", "user__height", "user__bmi", "user__general_hunger_level", "user__morning_hunger_level", "user__mid_hunger_level", "user__evening_hunger_level"]

    # Food Covariates
    food_categoricals = []
    food_reals = ['food__eaten_quantity_in_gram', 'food__energy_kcal_eaten',
        'food__carb_eaten', 'food__fat_eaten', 'food__protein_eaten',
        'food__fiber_eaten', 'food__alcohol_eaten', 'food__vegetables_fruits',
        'food__grains_potatoes_pulses', 'food__unclassified',
        'food__non_alcoholic_beverages',
        'food__dairy_products_meat_fish_eggs_tofu',
        'food__sweets_salty_snacks_alcohol', 'food__oils_fats_nuts']


    # Temporal Covariates
    temporal_categoricals = ["loc_eaten_dow", "loc_eaten_dow_type", "loc_eaten_season"]
    temporal_reals = ["loc_eaten_hour"]

    # Targets
    targets = ["val"]

    all_categorical_columns = user_static_categoricals + food_categoricals + temporal_categoricals
    all_real_columns = user_static_reals + food_reals + temporal_reals + targets
    
    
    # Load the data frames
    ppgr_df, users_demographics_df, microbiome_embeddings_df = load_dataframe(dataset_version, debug_mode)

    # Split the data frames into training, validation and test sets
    training_df, validation_df, test_df = split_timeseries_df_based_on_food_intake_rows(ppgr_df, validation_percentage=validation_percentage, test_percentage=test_percentage)
    
    # Validate the data frames
    ppgr_df, users_demographics_df = enforce_column_types(  ppgr_df, 
                                                            users_demographics_df, 
                                                            all_categorical_columns,
                                                            all_real_columns)

    # Setup the scalers and encoders
    categorical_encoders, continuous_scalers = setup_scalers_and_encoders(
        ppgr_df = ppgr_df,
        training_df = training_df,
        users_demographics_df = users_demographics_df,
        categorical_columns = all_categorical_columns,
        real_columns = all_real_columns
    ) # Note: the encoders are fit on the full ppgr_df, and the scalers are fit on the training_df

    # Create the training dataset
    training_dataset = PPGRTimeSeriesDataset(ppgr_df = training_df, 
                                            user_demographics_df = users_demographics_df,
                                            is_food_anchored = False, # When False, its the standard sliding window timeseries, but when True, every timeseries will have a food intake row at the last item of the context window
                                            time_idx = "read_at",
                                            target_columns = ["val"],
                                                                                    
                                            group_by_columns = ["timeseries_block_id"],

                                            min_encoder_length = 8 * 4, # 8 hours with 4 timepoints per hour
                                            max_encoder_length = 12 * 4, # 12 hours with 4 timepoints per hour
                                            
                                            prediction_length = 2 * 4, # 2 hours with 4 timepoints per hour
                                            
                                            add_relative_time_idx = True,
                                            use_food_covariates_from_prediction_window = True,
                                            
                                            use_microbiome_embeddings = True,
                                            microbiome_embeddings_df = microbiome_embeddings_df,
                                            
                                            temporal_categoricals = temporal_categoricals,
                                            temporal_reals = temporal_reals,

                                            user_static_categoricals = user_static_categoricals,
                                            user_static_reals = user_static_reals,
                                            
                                            food_categoricals = food_categoricals,
                                            food_reals = food_reals,
                                            
                                            categorical_encoders = categorical_encoders,
                                            continuous_scalers = continuous_scalers)

    print(f"Length of training dataset: {len(training_dataset)}")


    # data = training_dataset[0]

    # aggregated_df, encoder_df, decoder_df = training_dataset.inverse_transform_item(data)
    # display(encoder_df)
    # display(decoder_df)

    for data in tqdm(training_dataset):
        print(data["metadata"])
        # aggregated_df, encoder_df, decoder_df = training_dataset.inverse_transform_item(data)
        # print(encoder_df)
        # print(decoder_df)
        # break
        # display(decoder_df)
        break
        
