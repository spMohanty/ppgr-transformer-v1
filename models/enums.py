"""
Enums and constants used in the models.
"""
from enum import Enum, auto

class FusionBlockType(Enum):
    """Enum for fusion block types"""
    CROSS_MODAL_FUSION = "cmf"
    VARIABLE_SELECTION = "vsn"

class FusionMode(Enum):
    """Enum for fusion modes (used by CrossModalFusion)"""
    MEAN_POOLING = "mean_pooling"
    QUERY_TOKEN = "query_token"

# Modality names constants
BASE_MODALITIES_PAST = ["glucose", "meal", "temporal", "microbiome"]
BASE_MODALITIES_FUTURE = ["meal", "temporal", "microbiome"] 