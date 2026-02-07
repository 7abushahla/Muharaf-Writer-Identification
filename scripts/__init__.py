"""
Writer Identification Training Scripts Package
"""

__version__ = '1.0.0'
__author__ = 'Muharaf Writer Identification Team'

from .custom_layers import SpatialPyramidPooling, NetVLADLayer, L2Normalization, SelfAttentionBlock
from .custom_metrics import MacroPrecision, MacroRecall, MacroF1Score
from .custom_callbacks import ClearOutputEveryNEpochs, PeriodicModelCheckpoint
from .model_builder import build_writer_identification_model, get_backbone_model
from .data_utils import load_dataset, prepare_data_splits, create_data_generators

__all__ = [
    'SpatialPyramidPooling',
    'NetVLADLayer',
    'L2Normalization',
    'SelfAttentionBlock',
    'MacroPrecision',
    'MacroRecall',
    'MacroF1Score',
    'ClearOutputEveryNEpochs',
    'PeriodicModelCheckpoint',
    'build_writer_identification_model',
    'get_backbone_model',
    'load_dataset',
    'prepare_data_splits',
    'create_data_generators',
]
