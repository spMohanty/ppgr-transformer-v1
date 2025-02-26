from dataclasses import dataclass, asdict

# -----------------------------------------------------------------------------
# Experiment Configuration
# -----------------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    # Dataset / caching settings
    random_seed: int = 42
    dataset_version: str = "v0.5"
    cache_dir: str = "/scratch/mohanty/food/ppgr-v1/datasets-cache"
    use_cache: bool = True
    debug_mode: bool = False
    dataloader_num_workers: int = 7 

    # Data splitting & sequence parameters
    min_encoder_length: int = 8 * 4    # e.g., 8hrs * 4
    max_encoder_length: int = 12 * 4   # e.g., 12hrs * 4
    prediction_length: int = 4 * 4     # e.g.,  4hrs * 4
    eval_window: int = 2 * 4            # e.g., 2hrs * 4
    validation_percentage: float = 0.1
    test_percentage: float = 0.1
    
    encoder_length_randomization: str = "training_only" # "training_only", "all_sets", "none"
    
    # Aggregation
    patch_size: int = 1 * 4 # 1 hour patch size
    patch_stride: int = 2  # 30 min stride
    
    meal_aggregator_type: str = "set"

    # Data options
    is_food_anchored: bool = True
    sliding_window_stride: int = None
    use_meal_level_food_covariates: bool = True
    use_bootstraped_food_embeddings: bool = True
    freeze_food_id_embeddings: bool = True
    ignore_food_macro_features: bool = False
    use_microbiome_embeddings: bool = True
    group_by_columns: list = None

    # Feature lists (users, food, temporal)
    user_static_categoricals: list = None
    user_static_reals: list = None
    food_categoricals: list = None
    food_reals: list = None
    temporal_categoricals: list = None
    temporal_reals: list = None
    targets: list = None

    # Model hyperparameters
    max_meals: int = 11  # Maximum number of meals to consider
    food_embed_dim: int = 512 # the number of dimensions from the pre-trained embeddings to use
    hidden_dim: int = 256
    num_heads: int = 4
    transformer_encoder_layers: int = 2
    transformer_encoder_layers_share_weights: bool = True
    transformer_decoder_layers: int = 2
    transformer_decoder_layers_share_weights: bool = True
    add_residual_connection_before_predictions: bool = False
    num_quantiles: int = 7
    loss_iauc_weight: float = 0.00
    
    # New dropout hyperparameters
    dropout_rate: float = 0.1          # Used for projections, cross-attention, forecast MLP, etc.
    transformer_dropout: float = 0.1   # Used within Transformer layers

    # Training hyperparameters
    batch_size: int = 1024 * 2
    max_epochs: int = 50
    optimizer_lr: float = 1e-4
    optimizer_lr_scheduler_pct_start: float = 0.1
    weight_decay: float = 1e-5
    gradient_clip_val: float = 0.1  # Added gradient clipping parameter


    # Checkpoint Settings
    checkpoint_base_dir: str = "/scratch/mohanty/checkpoints/ppgr-meal-representation"  # Directory to save checkpoints
    checkpoint_monitor: str = "val_q_loss"  # Metric to monitor
    checkpoint_mode: str = "min"  # "min" or "max"
    checkpoint_top_k: int = 5  # Number of best checkpoints to keep

    # WandB logging
    wandb_project: str = "meal-representations-learning-v0"
    wandb_run_name: str = "MealGlucoseForecastModel_Run"
    wandb_log_embeddings: bool = False

    # Precision
    precision: str = "bf16"

    # Batch size for projecting food embeddings when logging
    food_embedding_projection_batch_size: int = 1024 * 4

    # Plots (default: plots enabled)
    disable_plots: bool = False

    def __post_init__(self):
        # Set default lists if not provided.
        if self.group_by_columns is None:
            self.group_by_columns = ["timeseries_block_id"]
        if self.user_static_categoricals is None:
            self.user_static_categoricals = [
                "user_id", "user__edu_degree", "user__income",
                "user__household_desc", "user__job_status", "user__smoking",
                "user__health_state", "user__physical_activities_frequency",
            ]
        if self.user_static_reals is None:
            self.user_static_reals = [
                "user__age", "user__weight", "user__height",
                "user__bmi", "user__general_hunger_level",
                "user__morning_hunger_level", "user__mid_hunger_level",
                "user__evening_hunger_level",
            ]
        if self.use_meal_level_food_covariates:
            self.food_categoricals = ["food__food_group_cname", "food_id"]
        else:
            self.food_categoricals = [
                "food__vegetables_fruits",
                "food__grains_potatoes_pulses",
                "food__unclassified",
                "food__non_alcoholic_beverages",
                "food__dairy_products_meat_fish_eggs_tofu",
                "food__sweets_salty_snacks_alcohol",
                "food__oils_fats_nuts",
            ]
        if self.food_reals is None:
            self.food_reals = [
                "food__eaten_quantity_in_gram", "food__energy_kcal_eaten",
                "food__carb_eaten", "food__fat_eaten",
                "food__protein_eaten", "food__fiber_eaten",
                "food__alcohol_eaten",
            ]
        if self.temporal_categoricals is None:
            self.temporal_categoricals = ["loc_eaten_dow", "loc_eaten_dow_type", "loc_eaten_season"]
        if self.temporal_reals is None:
            self.temporal_reals = ["loc_eaten_hour"]
        if self.targets is None:
            self.targets = ["val"]
            
            
def generate_experiment_name(config: ExperimentConfig, kwargs: dict) -> str:
    """
    Generate a meaningful experiment name based on modified parameters.
    
    Args:
        config: The ExperimentConfig instance
        kwargs: Dictionary of parameters passed via command line
    """
    # Get default config for comparison
    default_config = ExperimentConfig()
    
    # List of important parameters to include in name
    key_params = {
        # Dataset/caching
        'dataset_version': 'ds_ver',
        'use_cache': 'cache',
        'debug_mode': 'debug',
        'dataloader_num_workers': 'workers',
        
        # Sequence parameters
        'min_encoder_length': 'enc',
        'prediction_length': 'pred',
        'encoder_length_randomization': 'encL_rand',
        'eval_window': 'eval',
        'validation_percentage': 'val_pct',
        'test_percentage': 'test_pct',
        
        # Aggregation
        'patch_size': 'patch',
        'patch_stride': 'stride',
        
        # Data options
        'is_food_anchored': 'food_anchor',
        'sliding_window_stride': 'slide_stride',
        'use_meal_level_food_covariates': 'meal_cov',
        'use_bootstraped_food_embeddings': 'boot_emb',
        'use_microbiome_embeddings': 'micro_emb',
        
        # Model architecture
        'food_embed_dim': 'food_emb',
        'food_embed_adapter_dim': 'food_adapt',
        'hidden_dim': 'h',
        'num_heads': 'heads',
        'transformer_encoder_layers': 'enc_layers',
        'transformer_decoder_layers': 'dec_layers',
        'add_residual_connection_before_predictions': 'res_pred',
        'num_quantiles': 'quantiles',
        'loss_iauc_weight': 'iauc_wt',
        'add_glucose_causal_mask': 'gluc_causal_mask',
        
        # Dropout
        'dropout_rate': 'drop',
        'transformer_dropout': 'tdrop',
        
        # Training
        'batch_size': 'bs',
        'max_epochs': 'epochs',
        'optimizer_lr': 'lr',
        'weight_decay': 'wd',
        'gradient_clip_val': 'clip',
        
        # Logging
        'wandb_project': 'wandb',
        'wandb_run_name': 'run',
        
        # Precision
        'precision': 'prec',
        
        # Food embedding projection
        'food_embedding_projection_batch_size': 'proj_bs',
        
        # Plots
        'disable_plots': 'no_plots',
        
        # WandB logging
        'wandb_log_embeddings': 'log_emb'
    }
    
    # Build name components for modified parameters
    name_parts = []
    for param, shorthand in key_params.items():
        if param in kwargs and getattr(config, param) != getattr(default_config, param):
            value = getattr(config, param)
            # Handle boolean values with more readable format
            if isinstance(value, bool):
                value = "T" if value else "F"
            # Format numbers to remove trailing zeros
            elif isinstance(value, float):
                value = f"{value:.3f}".rstrip('0').rstrip('.')
            name_parts.append(f"{shorthand}{value}")
    
    # Create base name
    if name_parts:
        experiment_name = "ppgr_" + "_".join(name_parts)
    else:
        from faker import Faker
        fake = Faker()
        experiment_name = f"ppgr_default_{fake.word()}"
        
    # Add timestamp for uniqueness
    # timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    # experiment_name = f"{experiment_name}_{timestamp}"
    
    return experiment_name