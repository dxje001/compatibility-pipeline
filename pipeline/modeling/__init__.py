"""Modeling module for compatibility prediction models."""

from .trainer import CompatibilityModelTrainer
from .feature_importance import extract_feature_importance, FeatureImportanceReport

__all__ = [
    "CompatibilityModelTrainer",
    "extract_feature_importance",
    "FeatureImportanceReport"
]
