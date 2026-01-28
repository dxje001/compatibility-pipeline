"""
Late fusion for combining compatibility scores.

This module implements score-level fusion of the personality and interests
compatibility models. Late fusion is used because:
- The datasets do not share user identifiers
- Each model captures different aspects of compatibility
- Score fusion allows flexible weighting

Fusion Formula:
    final_score = alpha * score_personality + (1 - alpha) * score_interests

Alpha can be:
- Fixed: Constant weight (default)
- Confidence-based: Weighted by model prediction confidence (optional)
"""

import logging
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import json

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FusionConfig:
    """
    Configuration for late fusion.

    Attributes:
        alpha: Weight for personality score (1-alpha for interests)
        mode: "fixed" or "confidence_weighted"
        min_weight: Minimum weight for either model (for confidence mode)
    """
    alpha: float = 0.5
    mode: str = "fixed"
    min_weight: float = 0.2

    def validate(self) -> None:
        """Validate configuration values."""
        if not 0 <= self.alpha <= 1:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        if self.mode not in ["fixed", "confidence_weighted"]:
            raise ValueError(f"Unknown fusion mode: {self.mode}")
        if not 0 <= self.min_weight <= 0.5:
            raise ValueError(f"min_weight must be in [0, 0.5], got {self.min_weight}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FusionConfig":
        """Create from dictionary."""
        return cls(**d)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FusionConfig":
        """Create from main config dictionary."""
        fusion_config = config.get("fusion", {})
        confidence_config = fusion_config.get("confidence", {})

        return cls(
            alpha=fusion_config.get("alpha", 0.5),
            mode=fusion_config.get("mode", "fixed"),
            min_weight=confidence_config.get("min_weight", 0.2)
        )

    def save(self, filepath: str) -> None:
        """Save to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved fusion config to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "FusionConfig":
        """Load from JSON file."""
        with open(filepath, "r") as f:
            d = json.load(f)
        return cls.from_dict(d)


class LateFusion:
    """
    Late fusion combiner for compatibility scores.

    Combines predictions from personality and interests models
    into a single final compatibility score.

    The fusion happens at the score level, not the feature level,
    because the two datasets do not share user identifiers.

    Attributes:
        config: FusionConfig with fusion parameters
    """

    def __init__(self, config: FusionConfig):
        """
        Initialize the fusion combiner.

        Args:
            config: FusionConfig instance
        """
        self.config = config
        self.config.validate()
        logger.info(f"Initialized LateFusion with mode={config.mode}, alpha={config.alpha}")

    def fuse(
        self,
        score_personality: np.ndarray,
        score_interests: np.ndarray,
        return_components: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Combine personality and interests scores.

        Args:
            score_personality: Personality model scores (N,)
            score_interests: Interests model scores (N,)
            return_components: Whether to include component scores in output

        Returns:
            Dictionary with at least 'final_score', optionally including
            'score_personality', 'score_interests', and 'weights_used'
        """
        if len(score_personality) != len(score_interests):
            raise ValueError(
                f"Score arrays must have same length: "
                f"{len(score_personality)} vs {len(score_interests)}"
            )

        if self.config.mode == "fixed":
            final_score = self._fuse_fixed(score_personality, score_interests)
            weights = np.full((len(score_personality), 2), [self.config.alpha, 1 - self.config.alpha])
        elif self.config.mode == "confidence_weighted":
            final_score, weights = self._fuse_confidence_weighted(
                score_personality, score_interests
            )
        else:
            raise ValueError(f"Unknown fusion mode: {self.config.mode}")

        result = {"final_score": final_score}

        if return_components:
            result["score_personality"] = score_personality
            result["score_interests"] = score_interests
            result["weights_used"] = weights

        return result

    def _fuse_fixed(
        self,
        score_personality: np.ndarray,
        score_interests: np.ndarray
    ) -> np.ndarray:
        """
        Fixed-weight fusion.

        Formula: final = alpha * personality + (1 - alpha) * interests

        Args:
            score_personality: Personality scores
            score_interests: Interests scores

        Returns:
            Final fused scores
        """
        alpha = self.config.alpha
        return alpha * score_personality + (1 - alpha) * score_interests

    def _fuse_confidence_weighted(
        self,
        score_personality: np.ndarray,
        score_interests: np.ndarray
    ) -> tuple:
        """
        Confidence-weighted fusion.

        Uses prediction confidence (distance from 0.5) to weight models.
        More confident predictions get higher weight.

        Args:
            score_personality: Personality scores
            score_interests: Interests scores

        Returns:
            Tuple of (final_scores, weights_array)
        """
        # Compute confidence as distance from 0.5 (uncertainty)
        # Higher distance = more confident prediction
        conf_personality = np.abs(score_personality - 0.5)
        conf_interests = np.abs(score_interests - 0.5)

        # Compute raw weights proportional to confidence
        # Add small epsilon to avoid division by zero
        eps = 1e-6
        total_conf = conf_personality + conf_interests + eps

        weight_personality = conf_personality / total_conf
        weight_interests = conf_interests / total_conf

        # Apply minimum weight constraint
        min_w = self.config.min_weight
        max_w = 1 - min_w

        weight_personality = np.clip(weight_personality, min_w, max_w)
        weight_interests = 1 - weight_personality

        # Also apply base alpha as prior
        alpha = self.config.alpha
        weight_personality = alpha * 0.5 + weight_personality * 0.5
        weight_interests = 1 - weight_personality

        # Compute final score
        final_score = weight_personality * score_personality + weight_interests * score_interests

        weights = np.column_stack([weight_personality, weight_interests])

        return final_score, weights

    def get_effective_weights(self) -> Dict[str, float]:
        """
        Get effective fusion weights (for fixed mode).

        Returns:
            Dictionary with personality and interests weights
        """
        return {
            "personality": self.config.alpha,
            "interests": 1 - self.config.alpha
        }


def create_fusion_from_config(config: Dict[str, Any]) -> LateFusion:
    """
    Factory function to create LateFusion from config.

    Args:
        config: Main configuration dictionary

    Returns:
        Configured LateFusion instance
    """
    fusion_config = FusionConfig.from_config(config)
    return LateFusion(fusion_config)
