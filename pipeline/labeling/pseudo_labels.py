"""
Pseudo-label generation for compatibility modeling.

This module creates synthetic compatibility labels based on theoretically
motivated similarity functions. These pseudo-labels are used to train
ML models that serve as calibration layers for compatibility scoring.

Key Design Decisions:
- No ground-truth compatibility labels exist in the data
- Compatibility is a latent variable defined by the model
- Pseudo-labels are based on similarity metrics + noise
- Noise prevents trivial identity mapping learning
- All parameters are configurable for reproducibility

IPIP Pseudo-Label Formula:
    sim_ocean = (cosine(OCEAN_A, OCEAN_B) + 1) / 2
    sim_raw = (cosine(raw50_A, raw50_B) + 1) / 2
    pseudo = 0.7 * sim_ocean + 0.3 * sim_raw + noise
    label = clip(pseudo, 0, 1)

OkCupid Pseudo-Label Formula:
    sim_text = cosine(tfidf_A, tfidf_B)
    sim_cat = (cosine(cat_A, cat_B) + 1) / 2
    sim_num = (cosine(num_A, num_B) + 1) / 2
    pseudo = w_text * sim_text + w_cat * sim_cat + w_num * sim_num + noise
    label = clip(pseudo, 0, 1)
"""

import logging
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import json

import numpy as np
from scipy.sparse import csr_matrix

from ..feature_engineering.pairwise_features import (
    compute_similarity_scores_ipip,
    compute_similarity_scores_okcupid
)

logger = logging.getLogger(__name__)


@dataclass
class PseudoLabelConfig:
    """
    Configuration for pseudo-label generation.

    This dataclass stores all parameters used for pseudo-labeling,
    ensuring full reproducibility of the labeling process.
    """
    # IPIP model weights
    weight_ocean: float = 0.7
    weight_raw: float = 0.3

    # OkCupid model weights
    weight_text: float = 0.5
    weight_categorical: float = 0.3
    weight_numeric: float = 0.2

    # Noise configuration
    noise_std_personality: float = 0.08
    noise_std_interests: float = 0.08

    # Random seed for noise generation
    random_seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PseudoLabelConfig":
        """Create from dictionary."""
        return cls(**d)

    def save(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved pseudo-label config to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "PseudoLabelConfig":
        """Load configuration from JSON file."""
        with open(filepath, "r") as f:
            d = json.load(f)
        return cls.from_dict(d)

    @classmethod
    def from_config(cls, config: Dict[str, Any], random_seed: int) -> "PseudoLabelConfig":
        """
        Create from main config dictionary.

        Args:
            config: Main config dictionary
            random_seed: Random seed for noise

        Returns:
            PseudoLabelConfig instance
        """
        labeling_config = config.get("labeling", {})
        personality_config = labeling_config.get("personality", {})
        interests_config = labeling_config.get("interests", {})

        # Extract interests weights
        interests_weights = interests_config.get("weights", {})

        return cls(
            weight_ocean=personality_config.get("weight_ocean", 0.7),
            weight_raw=personality_config.get("weight_raw", 0.3),
            weight_text=interests_weights.get("text_similarity", 0.5),
            weight_categorical=interests_weights.get("categorical_similarity", 0.3),
            weight_numeric=interests_weights.get("numeric_similarity", 0.2),
            noise_std_personality=personality_config.get("noise_std", 0.08),
            noise_std_interests=interests_config.get("noise_std", 0.08),
            random_seed=random_seed
        )


def generate_pseudo_labels_ipip(
    raw_features: np.ndarray,
    ocean_features: np.ndarray,
    indices_a: np.ndarray,
    indices_b: np.ndarray,
    config: PseudoLabelConfig
) -> np.ndarray:
    """
    Generate pseudo-labels for IPIP personality compatibility.

    Formula:
        sim_ocean = (cosine(OCEAN_A, OCEAN_B) + 1) / 2
        sim_raw = (cosine(raw50_A, raw50_B) + 1) / 2
        pseudo = weight_ocean * sim_ocean + weight_raw * sim_raw + noise
        label = clip(pseudo, 0, 1)

    Args:
        raw_features: Raw question features for all persons (N x 50)
        ocean_features: OCEAN features for all persons (N x 5)
        indices_a: First person indices for each pair
        indices_b: Second person indices for each pair
        config: Pseudo-label configuration

    Returns:
        Array of pseudo-labels in [0, 1] for each pair
    """
    logger.info("Generating IPIP pseudo-labels")
    logger.info(f"Config: ocean_weight={config.weight_ocean}, "
               f"raw_weight={config.weight_raw}, noise_std={config.noise_std_personality}")

    # Extract features for each pair
    raw_a = raw_features[indices_a]
    raw_b = raw_features[indices_b]
    ocean_a = ocean_features[indices_a]
    ocean_b = ocean_features[indices_b]

    # Compute similarity scores
    sim_raw, sim_ocean = compute_similarity_scores_ipip(
        raw_a, raw_b, ocean_a, ocean_b
    )

    # Combine similarities with weights
    pseudo = config.weight_ocean * sim_ocean + config.weight_raw * sim_raw

    # Add noise
    rng = np.random.RandomState(config.random_seed)
    noise = rng.normal(0, config.noise_std_personality, size=len(pseudo))
    pseudo = pseudo + noise

    # Clip to [0, 1]
    labels = np.clip(pseudo, 0, 1)

    logger.info(f"Generated {len(labels)} pseudo-labels")
    logger.info(f"Label statistics: mean={labels.mean():.4f}, std={labels.std():.4f}, "
               f"min={labels.min():.4f}, max={labels.max():.4f}")

    return labels


def generate_pseudo_labels_okcupid(
    numeric_features: np.ndarray,
    categorical_features: np.ndarray,
    text_features: csr_matrix,
    indices_a: np.ndarray,
    indices_b: np.ndarray,
    config: PseudoLabelConfig
) -> np.ndarray:
    """
    Generate pseudo-labels for OkCupid interests compatibility.

    Formula:
        sim_text = cosine(tfidf_A, tfidf_B)  # Already in [0, 1] for TF-IDF
        sim_cat = (cosine(cat_A, cat_B) + 1) / 2
        sim_num = (cosine(num_A, num_B) + 1) / 2
        pseudo = w_text * sim_text + w_cat * sim_cat + w_num * sim_num + noise
        label = clip(pseudo, 0, 1)

    Args:
        numeric_features: Numeric features for all persons
        categorical_features: Categorical features for all persons
        text_features: TF-IDF features for all persons (sparse)
        indices_a: First person indices for each pair
        indices_b: Second person indices for each pair
        config: Pseudo-label configuration

    Returns:
        Array of pseudo-labels in [0, 1] for each pair
    """
    logger.info("Generating OkCupid pseudo-labels")
    logger.info(f"Config: text_weight={config.weight_text}, "
               f"cat_weight={config.weight_categorical}, "
               f"num_weight={config.weight_numeric}, "
               f"noise_std={config.noise_std_interests}")

    # Extract features for each pair
    numeric_a = numeric_features[indices_a]
    numeric_b = numeric_features[indices_b]
    categorical_a = categorical_features[indices_a]
    categorical_b = categorical_features[indices_b]
    text_a = text_features[indices_a]
    text_b = text_features[indices_b]

    # Compute similarity scores
    sim_numeric, sim_categorical, sim_text = compute_similarity_scores_okcupid(
        numeric_a, numeric_b,
        categorical_a, categorical_b,
        text_a, text_b
    )

    # Combine similarities with weights
    pseudo = (
        config.weight_text * sim_text +
        config.weight_categorical * sim_categorical +
        config.weight_numeric * sim_numeric
    )

    # Add noise
    rng = np.random.RandomState(config.random_seed + 1)  # Different seed from IPIP
    noise = rng.normal(0, config.noise_std_interests, size=len(pseudo))
    pseudo = pseudo + noise

    # Clip to [0, 1]
    labels = np.clip(pseudo, 0, 1)

    logger.info(f"Generated {len(labels)} pseudo-labels")
    logger.info(f"Label statistics: mean={labels.mean():.4f}, std={labels.std():.4f}, "
               f"min={labels.min():.4f}, max={labels.max():.4f}")

    return labels
