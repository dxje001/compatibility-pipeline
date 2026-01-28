"""
Evaluation metrics for compatibility models.

Since there are NO true labels for compatibility, evaluation focuses on:
1. Score distribution analysis
2. Stability across multiple runs
3. Sanity checks (monotonicity: higher similarity should give higher scores)
4. Feature importance consistency

This module DOES NOT claim real-world predictive accuracy.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

logger = logging.getLogger(__name__)


@dataclass
class ScoreDistributionStats:
    """Statistics about score distribution."""
    mean: float
    std: float
    min: float
    max: float
    quantiles: Dict[str, float]  # e.g., {"p10": 0.2, "p50": 0.5, "p90": 0.8}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": float(self.mean),
            "std": float(self.std),
            "min": float(self.min),
            "max": float(self.max),
            "quantiles": {k: float(v) for k, v in self.quantiles.items()}
        }


@dataclass
class StabilityMetrics:
    """Stability metrics across multiple runs."""
    n_runs: int
    score_std_mean: float  # Mean std of scores across runs
    score_std_std: float   # Std of std (meta-variance)
    prediction_correlation_mean: float  # Mean correlation between runs
    importance_jaccard_mean: float  # Mean Jaccard overlap of top features
    importance_spearman_mean: float  # Mean Spearman correlation of importances

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_runs": int(self.n_runs),
            "score_std_mean": float(self.score_std_mean),
            "score_std_std": float(self.score_std_std),
            "prediction_correlation_mean": float(self.prediction_correlation_mean),
            "importance_jaccard_mean": float(self.importance_jaccard_mean),
            "importance_spearman_mean": float(self.importance_spearman_mean)
        }


@dataclass
class MonotonicityCheck:
    """Results of monotonicity sanity check."""
    correlation_with_similarity: float
    is_monotonic: bool
    n_violations: int
    violation_rate: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "correlation_with_similarity": float(self.correlation_with_similarity),
            "is_monotonic": bool(self.is_monotonic),
            "n_violations": int(self.n_violations),
            "violation_rate": float(self.violation_rate)
        }


@dataclass
class EvaluationReport:
    """
    Complete evaluation report for a model.

    Contains distribution statistics, stability metrics, and sanity checks.
    This report documents model behavior WITHOUT claiming predictive validity.
    """
    model_name: str
    distribution_stats: ScoreDistributionStats
    stability_metrics: Optional[StabilityMetrics] = None
    monotonicity_check: Optional[MonotonicityCheck] = None
    additional_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "model_name": self.model_name,
            "distribution_stats": self.distribution_stats.to_dict(),
            "additional_metrics": self.additional_metrics
        }
        if self.stability_metrics:
            result["stability_metrics"] = self.stability_metrics.to_dict()
        if self.monotonicity_check:
            result["monotonicity_check"] = self.monotonicity_check.to_dict()
        return result

    def save(self, filepath: str) -> None:
        """Save report to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved evaluation report to {filepath}")

    def summary(self) -> str:
        """Generate text summary of the report."""
        lines = [
            f"Evaluation Report: {self.model_name}",
            "=" * 50,
            "",
            "Score Distribution:",
            f"  Mean: {self.distribution_stats.mean:.4f}",
            f"  Std:  {self.distribution_stats.std:.4f}",
            f"  Min:  {self.distribution_stats.min:.4f}",
            f"  Max:  {self.distribution_stats.max:.4f}",
        ]

        for q_name, q_value in self.distribution_stats.quantiles.items():
            lines.append(f"  {q_name}: {q_value:.4f}")

        if self.stability_metrics:
            lines.extend([
                "",
                f"Stability Metrics ({self.stability_metrics.n_runs} runs):",
                f"  Score Std (mean): {self.stability_metrics.score_std_mean:.4f}",
                f"  Prediction Correlation: {self.stability_metrics.prediction_correlation_mean:.4f}",
                f"  Importance Jaccard: {self.stability_metrics.importance_jaccard_mean:.4f}",
                f"  Importance Spearman: {self.stability_metrics.importance_spearman_mean:.4f}",
            ])

        if self.monotonicity_check:
            lines.extend([
                "",
                "Monotonicity Check:",
                f"  Correlation with similarity: {self.monotonicity_check.correlation_with_similarity:.4f}",
                f"  Is monotonic: {self.monotonicity_check.is_monotonic}",
                f"  Violation rate: {self.monotonicity_check.violation_rate:.2%}",
            ])

        return "\n".join(lines)


def compute_score_distribution_stats(
    scores: np.ndarray,
    quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9]
) -> ScoreDistributionStats:
    """
    Compute distribution statistics for scores.

    Args:
        scores: Array of compatibility scores
        quantiles: Quantile values to compute (default: p10, p25, p50, p75, p90)

    Returns:
        ScoreDistributionStats instance
    """
    quantile_dict = {
        f"p{int(q * 100)}": float(np.percentile(scores, q * 100))
        for q in quantiles
    }

    return ScoreDistributionStats(
        mean=float(np.mean(scores)),
        std=float(np.std(scores)),
        min=float(np.min(scores)),
        max=float(np.max(scores)),
        quantiles=quantile_dict
    )


def compute_stability_metrics(
    run_scores: List[np.ndarray],
    run_importances: List[Dict[str, float]],
    top_k: int = 20
) -> StabilityMetrics:
    """
    Compute stability metrics across multiple training runs.

    Args:
        run_scores: List of score arrays from different runs
        run_importances: List of importance dicts from different runs
        top_k: Number of top features for Jaccard computation

    Returns:
        StabilityMetrics instance
    """
    n_runs = len(run_scores)

    if n_runs < 2:
        logger.warning("Need at least 2 runs for stability analysis")
        return StabilityMetrics(
            n_runs=n_runs,
            score_std_mean=0.0,
            score_std_std=0.0,
            prediction_correlation_mean=1.0,
            importance_jaccard_mean=1.0,
            importance_spearman_mean=1.0
        )

    # Stack scores for analysis (assumes same ordering)
    scores_matrix = np.vstack(run_scores)  # (n_runs, n_samples)

    # Compute per-sample std across runs
    per_sample_std = np.std(scores_matrix, axis=0)
    score_std_mean = float(np.mean(per_sample_std))
    score_std_std = float(np.std(per_sample_std))

    # Compute pairwise prediction correlations
    pred_correlations = []
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            corr, _ = pearsonr(run_scores[i], run_scores[j])
            pred_correlations.append(corr)

    # Compute importance stability
    jaccard_scores = []
    spearman_scores = []

    # Get common features
    common_features = set(run_importances[0].keys())
    for imp in run_importances[1:]:
        common_features &= set(imp.keys())
    common_features = sorted(common_features)

    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            # Jaccard overlap of top-k
            top_i = set(sorted(run_importances[i].keys(),
                              key=lambda x: run_importances[i][x], reverse=True)[:top_k])
            top_j = set(sorted(run_importances[j].keys(),
                              key=lambda x: run_importances[j][x], reverse=True)[:top_k])
            intersection = len(top_i & top_j)
            union = len(top_i | top_j)
            jaccard = intersection / union if union > 0 else 0
            jaccard_scores.append(jaccard)

            # Spearman correlation of importances
            vec_i = [run_importances[i].get(f, 0) for f in common_features]
            vec_j = [run_importances[j].get(f, 0) for f in common_features]
            if len(vec_i) > 1:
                spearman, _ = spearmanr(vec_i, vec_j)
                spearman_scores.append(spearman)

    return StabilityMetrics(
        n_runs=n_runs,
        score_std_mean=score_std_mean,
        score_std_std=score_std_std,
        prediction_correlation_mean=float(np.mean(pred_correlations)),
        importance_jaccard_mean=float(np.mean(jaccard_scores)) if jaccard_scores else 0.0,
        importance_spearman_mean=float(np.mean(spearman_scores)) if spearman_scores else 0.0
    )


def sanity_check_monotonicity(
    predicted_scores: np.ndarray,
    similarity_scores: np.ndarray,
    threshold: float = 0.5
) -> MonotonicityCheck:
    """
    Check if predicted scores are monotonic with input similarity.

    Higher similarity should generally lead to higher compatibility scores.
    This is a sanity check, not a validation of predictive accuracy.

    Args:
        predicted_scores: Model predictions
        similarity_scores: Input similarity measures
        threshold: Correlation threshold for "is_monotonic" flag

    Returns:
        MonotonicityCheck instance
    """
    # Compute Spearman correlation (rank-based monotonicity)
    correlation, _ = spearmanr(similarity_scores, predicted_scores)

    # Count violations: pairs where similarity increases but score decreases
    n_samples = len(predicted_scores)
    n_comparisons = 0
    n_violations = 0

    # Sample pairs for efficiency (if dataset is large)
    if n_samples > 10000:
        sample_size = 10000
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(n_samples, size=sample_size, replace=False)
        pred_sample = predicted_scores[sample_idx]
        sim_sample = similarity_scores[sample_idx]
    else:
        pred_sample = predicted_scores
        sim_sample = similarity_scores
        sample_size = n_samples

    # Check pairs
    for i in range(min(sample_size, 1000)):  # Limit comparisons
        for j in range(i + 1, min(sample_size, 1000)):
            n_comparisons += 1
            sim_diff = sim_sample[j] - sim_sample[i]
            pred_diff = pred_sample[j] - pred_sample[i]
            # Violation: similarity increases but score decreases (or vice versa)
            if sim_diff * pred_diff < 0:
                n_violations += 1

    violation_rate = n_violations / n_comparisons if n_comparisons > 0 else 0

    return MonotonicityCheck(
        correlation_with_similarity=float(correlation),
        is_monotonic=correlation >= threshold,
        n_violations=n_violations,
        violation_rate=violation_rate
    )


def create_evaluation_report(
    model_name: str,
    scores: np.ndarray,
    similarity_scores: Optional[np.ndarray] = None,
    run_scores: Optional[List[np.ndarray]] = None,
    run_importances: Optional[List[Dict[str, float]]] = None,
    quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
    top_k: int = 20
) -> EvaluationReport:
    """
    Create a complete evaluation report.

    Args:
        model_name: Name of the model
        scores: Predicted scores
        similarity_scores: Input similarity scores (for monotonicity check)
        run_scores: Scores from multiple runs (for stability analysis)
        run_importances: Importances from multiple runs (for stability analysis)
        quantiles: Quantiles to compute
        top_k: Top-k features for importance stability

    Returns:
        EvaluationReport instance
    """
    # Distribution stats
    dist_stats = compute_score_distribution_stats(scores, quantiles)

    # Stability metrics
    stability = None
    if run_scores and run_importances:
        stability = compute_stability_metrics(run_scores, run_importances, top_k)

    # Monotonicity check
    monotonicity = None
    if similarity_scores is not None:
        monotonicity = sanity_check_monotonicity(scores, similarity_scores)

    return EvaluationReport(
        model_name=model_name,
        distribution_stats=dist_stats,
        stability_metrics=stability,
        monotonicity_check=monotonicity
    )
