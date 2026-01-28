"""
Inference module for compatibility scoring.

This module provides the inference pipeline for computing compatibility
scores from UI questionnaire responses.
"""

from .schema import QuestionnaireResponse, PersonalityAnswers, InterestsAnswers
from .predict import CompatibilityPredictor

__all__ = [
    "QuestionnaireResponse",
    "PersonalityAnswers",
    "InterestsAnswers",
    "CompatibilityPredictor",
]
