"""
Feature importance extraction and reporting.

This module provides utilities for extracting feature importance
from trained models and generating reports that support Iteration 2
UI question derivation.

Feature Importance Purpose:
- Identify which pairwise features most influence compatibility
- Support derivation of UI questions in Iteration 2
- Enable stability analysis across training runs
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FeatureImportanceReport:
    """
    Report of feature importances with ranking and statistics.

    This class stores and analyzes feature importance data
    from trained compatibility models.

    Attributes:
        model_name: Name of the model (e.g., "personality", "interests")
        importances: Dict mapping feature name to importance score
        sorted_features: List of (feature_name, importance) sorted by importance
        top_k: Number of top features to highlight
    """
    model_name: str
    importances: Dict[str, float]
    top_k: int = 20
    sorted_features: List[tuple] = field(default_factory=list)

    def __post_init__(self):
        """Sort features by importance after initialization."""
        self.sorted_features = sorted(
            self.importances.items(),
            key=lambda x: x[1],
            reverse=True
        )

    def get_top_features(self, k: Optional[int] = None) -> List[tuple]:
        """
        Get top-k most important features.

        Args:
            k: Number of features to return (default: self.top_k)

        Returns:
            List of (feature_name, importance) tuples
        """
        k = k or self.top_k
        return self.sorted_features[:k]

    def get_feature_rank(self, feature_name: str) -> Optional[int]:
        """
        Get the rank of a specific feature (1-indexed).

        Args:
            feature_name: Name of the feature

        Returns:
            Rank (1 = most important), or None if not found
        """
        for rank, (name, _) in enumerate(self.sorted_features, start=1):
            if name == feature_name:
                return rank
        return None

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert to pandas DataFrame.

        Returns:
            DataFrame with columns: feature, importance, rank
        """
        df = pd.DataFrame(self.sorted_features, columns=["feature", "importance"])
        df["rank"] = range(1, len(df) + 1)
        df["model"] = self.model_name
        return df

    def save_csv(self, filepath: str) -> None:
        """
        Save report to CSV file.

        Args:
            filepath: Path to save the CSV
        """
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)
        logger.info(f"Saved feature importance report to {filepath}")

    @classmethod
    def load_csv(cls, filepath: str, model_name: str) -> "FeatureImportanceReport":
        """
        Load report from CSV file.

        Args:
            filepath: Path to the CSV file
            model_name: Name to assign to the report

        Returns:
            FeatureImportanceReport instance
        """
        df = pd.read_csv(filepath)
        importances = dict(zip(df["feature"], df["importance"]))
        return cls(model_name=model_name, importances=importances)

    def summary(self) -> str:
        """
        Generate a text summary of the report.

        Returns:
            Multi-line string with summary statistics
        """
        lines = [
            f"Feature Importance Report: {self.model_name}",
            f"Total features: {len(self.importances)}",
            f"",
            f"Top {self.top_k} features:",
        ]

        for rank, (name, imp) in enumerate(self.get_top_features(), start=1):
            lines.append(f"  {rank:3d}. {name}: {imp:.6f}")

        # Statistics
        values = list(self.importances.values())
        lines.extend([
            "",
            "Statistics:",
            f"  Mean importance: {np.mean(values):.6f}",
            f"  Std importance: {np.std(values):.6f}",
            f"  Max importance: {np.max(values):.6f}",
            f"  Min importance: {np.min(values):.6f}",
        ])

        return "\n".join(lines)


def extract_feature_importance(
    model,
    feature_names: List[str],
    model_name: str,
    top_k: int = 20
) -> FeatureImportanceReport:
    """
    Extract feature importance from a trained model.

    Args:
        model: CompatibilityModelTrainer instance
        feature_names: List of feature names
        model_name: Name for the report
        top_k: Number of top features to highlight

    Returns:
        FeatureImportanceReport instance
    """
    importances = model.get_feature_importance()

    report = FeatureImportanceReport(
        model_name=model_name,
        importances=importances,
        top_k=top_k
    )

    logger.info(f"Extracted importance for {len(importances)} features")
    return report


def compute_importance_stability(
    reports: List[FeatureImportanceReport],
    top_k: int = 20
) -> Dict[str, Any]:
    """
    Compute stability metrics across multiple importance reports.

    Metrics computed:
    - Jaccard overlap of top-k features
    - Spearman correlation of full importance vectors
    - Per-feature rank variance

    Args:
        reports: List of reports from different runs
        top_k: Number of top features for Jaccard computation

    Returns:
        Dictionary with stability metrics
    """
    if len(reports) < 2:
        logger.warning("Need at least 2 reports for stability analysis")
        return {"error": "Insufficient reports"}

    n_reports = len(reports)

    # Compute pairwise Jaccard overlap of top-k features
    jaccard_scores = []
    for i in range(n_reports):
        for j in range(i + 1, n_reports):
            top_i = set(name for name, _ in reports[i].get_top_features(top_k))
            top_j = set(name for name, _ in reports[j].get_top_features(top_k))
            intersection = len(top_i & top_j)
            union = len(top_i | top_j)
            jaccard = intersection / union if union > 0 else 0
            jaccard_scores.append(jaccard)

    # Get common features across all reports
    common_features = set(reports[0].importances.keys())
    for report in reports[1:]:
        common_features &= set(report.importances.keys())

    common_features = sorted(common_features)

    # Compute Spearman correlation of importance vectors
    from scipy.stats import spearmanr

    spearman_scores = []
    for i in range(n_reports):
        for j in range(i + 1, n_reports):
            vec_i = [reports[i].importances[f] for f in common_features]
            vec_j = [reports[j].importances[f] for f in common_features]
            corr, _ = spearmanr(vec_i, vec_j)
            spearman_scores.append(corr)

    # Compute per-feature rank variance
    feature_ranks = {f: [] for f in common_features}
    for report in reports:
        for f in common_features:
            rank = report.get_feature_rank(f)
            if rank is not None:
                feature_ranks[f].append(rank)

    rank_variances = {
        f: np.var(ranks) if len(ranks) > 1 else 0
        for f, ranks in feature_ranks.items()
    }

    # Find most stable features (lowest rank variance)
    stable_features = sorted(rank_variances.items(), key=lambda x: x[1])[:top_k]

    results = {
        "n_reports": n_reports,
        "jaccard_overlap": {
            "mean": np.mean(jaccard_scores),
            "std": np.std(jaccard_scores),
            "min": np.min(jaccard_scores),
            "max": np.max(jaccard_scores),
            "all_scores": jaccard_scores
        },
        "spearman_correlation": {
            "mean": np.mean(spearman_scores),
            "std": np.std(spearman_scores),
            "min": np.min(spearman_scores),
            "max": np.max(spearman_scores),
            "all_scores": spearman_scores
        },
        "most_stable_features": stable_features,
        "n_common_features": len(common_features)
    }

    logger.info(f"Computed stability over {n_reports} reports")
    logger.info(f"Mean Jaccard overlap (top-{top_k}): {results['jaccard_overlap']['mean']:.4f}")
    logger.info(f"Mean Spearman correlation: {results['spearman_correlation']['mean']:.4f}")

    return results
