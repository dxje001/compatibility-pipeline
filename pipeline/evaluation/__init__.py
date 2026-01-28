"""Evaluation module for compatibility model analysis."""

from .metrics import (
    compute_score_distribution_stats,
    compute_stability_metrics,
    sanity_check_monotonicity,
    EvaluationReport,
    create_evaluation_report
)

__all__ = [
    "compute_score_distribution_stats",
    "compute_stability_metrics",
    "sanity_check_monotonicity",
    "EvaluationReport",
    "create_evaluation_report"
]
