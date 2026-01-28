"""
Stability analysis script for multi-seed training runs.

Analyzes score distributions and feature importance stability across seeds.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations
from typing import Dict, List, Tuple, Any


def load_evaluation_reports(runs_dir: Path, seeds: List[int]) -> Dict[str, Dict[int, Dict]]:
    """Load evaluation reports for all seeds."""
    reports = {"personality": {}, "interests": {}}

    for seed in seeds:
        seed_dir = runs_dir / f"seed_{seed}"

        for model_type in ["personality", "interests"]:
            report_path = seed_dir / "reports" / f"evaluation_{model_type}.json"
            if report_path.exists():
                with open(report_path) as f:
                    reports[model_type][seed] = json.load(f)

    return reports


def load_feature_importance(runs_dir: Path, seeds: List[int]) -> Dict[str, Dict[int, pd.DataFrame]]:
    """Load feature importance CSVs for all seeds."""
    importance = {"personality": {}, "interests": {}}

    for seed in seeds:
        seed_dir = runs_dir / f"seed_{seed}"

        for model_type in ["personality", "interests"]:
            csv_path = seed_dir / "reports" / f"feature_importance_{model_type}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                importance[model_type][seed] = df

    return importance


def compute_score_distribution_stability(
    reports: Dict[int, Dict],
    baseline_report: Dict
) -> Dict[str, Any]:
    """Compute score distribution stability metrics."""
    seeds = sorted(reports.keys())

    # Extract stats per seed
    per_seed = []
    for seed in seeds:
        stats = reports[seed]["distribution_stats"]
        per_seed.append({
            "seed": seed,
            "mean": stats["mean"],
            "std": stats["std"],
            "min": stats["min"],
            "max": stats["max"]
        })

    df = pd.DataFrame(per_seed)

    # Compute deltas vs baseline
    baseline_stats = baseline_report["distribution_stats"]
    deltas = []
    for seed in seeds:
        stats = reports[seed]["distribution_stats"]
        deltas.append({
            "seed": seed,
            "delta_mean": stats["mean"] - baseline_stats["mean"],
            "delta_std": stats["std"] - baseline_stats["std"],
            "delta_min": stats["min"] - baseline_stats["min"],
            "delta_max": stats["max"] - baseline_stats["max"]
        })

    # Aggregate stability metrics
    return {
        "per_seed": per_seed,
        "deltas_vs_baseline": deltas,
        "aggregate": {
            "mean_of_means": float(df["mean"].mean()),
            "std_of_means": float(df["mean"].std()),
            "mean_of_stds": float(df["std"].mean()),
            "std_of_stds": float(df["std"].std()),
            "max_delta_mean": float(max(abs(d["delta_mean"]) for d in deltas)),
            "max_delta_std": float(max(abs(d["delta_std"]) for d in deltas))
        }
    }


def compute_feature_importance_stability(
    importance_dfs: Dict[int, pd.DataFrame],
    top_k: int = 20
) -> Dict[str, Any]:
    """Compute feature importance stability metrics."""
    seeds = sorted(importance_dfs.keys())

    # Get top-k features per seed
    top_features_per_seed = {}
    for seed in seeds:
        df = importance_dfs[seed].sort_values("importance", ascending=False)
        top_features_per_seed[seed] = list(df.head(top_k)["feature"])

    # Compute pairwise Jaccard overlap
    jaccard_scores = []
    for seed_i, seed_j in combinations(seeds, 2):
        set_i = set(top_features_per_seed[seed_i])
        set_j = set(top_features_per_seed[seed_j])
        intersection = len(set_i & set_j)
        union = len(set_i | set_j)
        jaccard = intersection / union if union > 0 else 0
        jaccard_scores.append({
            "seed_i": seed_i,
            "seed_j": seed_j,
            "jaccard": jaccard,
            "intersection_size": intersection
        })

    avg_jaccard = np.mean([j["jaccard"] for j in jaccard_scores])

    # Count feature frequency in top-k
    all_top_features = []
    for features in top_features_per_seed.values():
        all_top_features.extend(features)

    feature_counts = pd.Series(all_top_features).value_counts()

    # Features appearing in >= 4 out of 5 runs
    stable_features = list(feature_counts[feature_counts >= 4].index)
    unstable_features = list(feature_counts[feature_counts <= 2].index)

    return {
        "top_features_per_seed": {str(k): v for k, v in top_features_per_seed.items()},
        "pairwise_jaccard": jaccard_scores,
        "average_jaccard": float(avg_jaccard),
        "feature_frequency": feature_counts.to_dict(),
        "stable_features_gte_4": stable_features,
        "unstable_features_lte_2": unstable_features,
        "n_stable": len(stable_features),
        "n_unstable": len(unstable_features)
    }


def compute_rank_stability(
    importance_dfs: Dict[int, pd.DataFrame],
    top_k: int = 20
) -> Dict[str, Any]:
    """Compute rank stability for features appearing in top-k."""
    seeds = sorted(importance_dfs.keys())

    # Collect all features that appear in any top-k
    all_top_features = set()
    for seed in seeds:
        df = importance_dfs[seed].sort_values("importance", ascending=False)
        top_features = list(df.head(top_k)["feature"])
        all_top_features.update(top_features)

    # Compute rank for each feature in each seed
    rank_data = []
    for feature in all_top_features:
        ranks = []
        for seed in seeds:
            df = importance_dfs[seed].sort_values("importance", ascending=False).reset_index(drop=True)
            df["rank"] = df.index + 1
            feature_row = df[df["feature"] == feature]
            if not feature_row.empty:
                ranks.append(feature_row["rank"].iloc[0])

        if len(ranks) >= 2:
            rank_data.append({
                "feature": feature,
                "n_appearances": len(ranks),
                "avg_rank": float(np.mean(ranks)),
                "std_rank": float(np.std(ranks)),
                "min_rank": int(min(ranks)),
                "max_rank": int(max(ranks))
            })

    rank_df = pd.DataFrame(rank_data)

    # Sort by frequency, then by avg_rank
    rank_df = rank_df.sort_values(["n_appearances", "avg_rank"], ascending=[False, True])

    # Flag high volatility features (std > 5)
    high_volatility = rank_df[rank_df["std_rank"] > 5]["feature"].tolist()

    return {
        "rank_stats": rank_df.to_dict(orient="records"),
        "high_volatility_features": high_volatility,
        "n_high_volatility": len(high_volatility)
    }


def main():
    # Configuration
    project_dir = Path("C:/Users/Dusan/compatibility_pipeline")
    runs_dir = project_dir / "artifacts" / "runs"
    baseline_dir = project_dir / "artifacts"
    stability_dir = project_dir / "artifacts" / "stability"

    seeds = [11, 22, 33, 44, 55]
    top_k = 20

    # Create stability output directory
    stability_dir.mkdir(parents=True, exist_ok=True)

    # Load baseline reports (seed 42)
    baseline_reports = {}
    for model_type in ["personality", "interests"]:
        baseline_path = baseline_dir / "reports" / f"evaluation_{model_type}.json"
        if baseline_path.exists():
            with open(baseline_path) as f:
                baseline_reports[model_type] = json.load(f)

    # Load all reports and importance data
    reports = load_evaluation_reports(runs_dir, seeds)
    importance = load_feature_importance(runs_dir, seeds)

    # Compute stability metrics
    stability_results = {
        "seeds_analyzed": seeds,
        "baseline_seed": 42,
        "top_k_features": top_k,
        "personality": {},
        "interests": {}
    }

    for model_type in ["personality", "interests"]:
        print(f"\nAnalyzing {model_type} model stability...")

        # Score distribution stability
        score_stability = compute_score_distribution_stability(
            reports[model_type],
            baseline_reports[model_type]
        )
        stability_results[model_type]["score_distribution"] = score_stability

        # Feature importance stability
        importance_stability = compute_feature_importance_stability(
            importance[model_type],
            top_k=top_k
        )
        stability_results[model_type]["feature_importance"] = importance_stability

        # Rank stability
        rank_stability = compute_rank_stability(
            importance[model_type],
            top_k=top_k
        )
        stability_results[model_type]["rank_stability"] = rank_stability

        print(f"  Average Jaccard overlap (top-{top_k}): {importance_stability['average_jaccard']:.4f}")
        print(f"  Stable features (>=4/5 runs): {importance_stability['n_stable']}")
        print(f"  High volatility features: {rank_stability['n_high_volatility']}")

    # Save stability_summary.json
    json_path = stability_dir / "stability_summary.json"
    with open(json_path, "w") as f:
        json.dump(stability_results, f, indent=2)
    print(f"\nSaved: {json_path}")

    # Create human-readable CSV summary
    csv_rows = []

    # Score distribution table
    for model_type in ["personality", "interests"]:
        score_data = stability_results[model_type]["score_distribution"]
        for item in score_data["per_seed"]:
            csv_rows.append({
                "model": model_type,
                "metric": "score_distribution",
                "seed": item["seed"],
                "mean": item["mean"],
                "std": item["std"],
                "min": item["min"],
                "max": item["max"]
            })

        # Add baseline
        bs = baseline_reports[model_type]["distribution_stats"]
        csv_rows.append({
            "model": model_type,
            "metric": "score_distribution",
            "seed": 42,
            "mean": bs["mean"],
            "std": bs["std"],
            "min": bs["min"],
            "max": bs["max"]
        })

    score_df = pd.DataFrame(csv_rows)
    csv_path = stability_dir / "stability_summary.csv"
    score_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Create feature stability CSV
    feature_rows = []
    for model_type in ["personality", "interests"]:
        rank_stats = stability_results[model_type]["rank_stability"]["rank_stats"]
        for item in rank_stats:
            item_copy = item.copy()
            item_copy["model"] = model_type
            feature_rows.append(item_copy)

    feature_df = pd.DataFrame(feature_rows)
    feature_csv_path = stability_dir / "feature_rank_stability.csv"
    feature_df.to_csv(feature_csv_path, index=False)
    print(f"Saved: {feature_csv_path}")

    print("\nStability analysis complete!")
    return stability_results


if __name__ == "__main__":
    results = main()
