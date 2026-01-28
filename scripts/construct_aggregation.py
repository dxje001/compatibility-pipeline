"""
Construct-level aggregation for UI question selection.

Aggregates raw feature importances into interpretable constructs,
computes stability metrics across seeds, and ranks constructs.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import re


# =============================================================================
# CONSTRUCT DEFINITIONS (APPROVED)
# =============================================================================

# Personality constructs (IPIP)
PERSONALITY_CONSTRUCTS = {
    "Extraversion": {
        "items": ["EXT1", "EXT2", "EXT3", "EXT4", "EXT5", "EXT6", "EXT7", "EXT8", "EXT9", "EXT10"],
        "ocean_prefix": "ocean_extraversion"
    },
    "Neuroticism": {
        "items": ["EST1", "EST2", "EST3", "EST4", "EST5", "EST6", "EST7", "EST8", "EST9", "EST10"],
        "ocean_prefix": "ocean_neuroticism"
    },
    "Agreeableness": {
        "items": ["AGR1", "AGR2", "AGR3", "AGR4", "AGR5", "AGR6", "AGR7", "AGR8", "AGR9", "AGR10"],
        "ocean_prefix": "ocean_agreeableness"
    },
    "Conscientiousness": {
        "items": ["CSN1", "CSN2", "CSN3", "CSN4", "CSN5", "CSN6", "CSN7", "CSN8", "CSN9", "CSN10"],
        "ocean_prefix": "ocean_conscientiousness"
    },
    "Openness": {
        "items": ["OPN1", "OPN2", "OPN3", "OPN4", "OPN5", "OPN6", "OPN7", "OPN8", "OPN9", "OPN10"],
        "ocean_prefix": "ocean_openness"
    },
    "Global Similarity": {
        "exact_features": ["ocean_cosine_sim", "raw_cosine_sim"]
    }
}

# Interests constructs (OkCupid) - with approved merges and exclusions
INTERESTS_CONSTRUCTS = {
    # Global similarity constructs
    "Text Interests": {"exact_features": ["text_cosine_sim"]},
    "Categorical Similarity": {"exact_features": ["categorical_cosine_sim"]},
    "Numeric Similarity": {"exact_features": ["numeric_cosine_sim"]},

    # Numeric field constructs
    "Age": {"prefix": "age_"},
    "Height": {"prefix": "height_"},
    "Income": {"prefix": "income_"},

    # Categorical field constructs (separate)
    "Sex": {"prefix": "sex_"},
    "Orientation": {"prefix": "orientation_"},
    "Diet": {"prefix": "diet_"},
    "Body Type": {"prefix": "body_type_"},
    "Education": {"prefix": "education_"},
    "Job/Career": {"prefix": "job_"},
    "Religion": {"prefix": "religion_"},
    "Pets": {"prefix": "pets_"},

    # Merged constructs
    "Lifestyle Habits": {"prefixes": ["drinks_", "smokes_", "drugs_"]},
    "Family & Relationship": {"prefixes": ["status_", "offspring_"]}
}

# Excluded constructs (for reference)
EXCLUDED_PREFIXES = ["sign_"]


def feature_matches_construct(feature: str, construct_def: Dict) -> bool:
    """Check if a feature belongs to a construct definition."""
    # Exact feature match
    if "exact_features" in construct_def:
        return feature in construct_def["exact_features"]

    # Single prefix match
    if "prefix" in construct_def:
        return feature.startswith(construct_def["prefix"])

    # Multiple prefixes (merged constructs)
    if "prefixes" in construct_def:
        return any(feature.startswith(p) for p in construct_def["prefixes"])

    # Item-based match (personality)
    if "items" in construct_def:
        for item in construct_def["items"]:
            if feature.startswith(item + "_"):
                return True

    # OCEAN aggregate match
    if "ocean_prefix" in construct_def:
        return feature.startswith(construct_def["ocean_prefix"])

    return False


def is_excluded(feature: str) -> bool:
    """Check if feature should be excluded."""
    return any(feature.startswith(p) for p in EXCLUDED_PREFIXES)


def aggregate_to_constructs(
    importance_df: pd.DataFrame,
    constructs: Dict[str, Dict],
    model_type: str
) -> Dict[str, Dict]:
    """
    Aggregate feature importances to construct level.

    Returns dict with construct -> {sum_importance, mean_importance, n_features, features}
    """
    results = {}

    for construct_name, construct_def in constructs.items():
        # Find matching features
        matching_features = []
        for _, row in importance_df.iterrows():
            feature = row["feature"]
            if is_excluded(feature):
                continue
            if feature_matches_construct(feature, construct_def):
                matching_features.append({
                    "feature": feature,
                    "importance": row["importance"],
                    "abs_importance": abs(row["importance"])
                })

        if matching_features:
            abs_importances = [f["abs_importance"] for f in matching_features]
            results[construct_name] = {
                "sum_importance": sum(abs_importances),
                "mean_importance": np.mean(abs_importances),
                "n_features": len(matching_features),
                "top_features": sorted(matching_features,
                                       key=lambda x: x["abs_importance"],
                                       reverse=True)[:5]
            }
        else:
            results[construct_name] = {
                "sum_importance": 0.0,
                "mean_importance": 0.0,
                "n_features": 0,
                "top_features": []
            }

    return results


def load_all_importance_data(runs_dir: Path, seeds: List[int]) -> Dict[str, Dict[int, pd.DataFrame]]:
    """Load feature importance CSVs for all seeds."""
    data = {"personality": {}, "interests": {}}

    for seed in seeds:
        seed_dir = runs_dir / f"seed_{seed}"

        for model_type in ["personality", "interests"]:
            csv_path = seed_dir / "reports" / f"feature_importance_{model_type}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                data[model_type][seed] = df

    return data


def compute_construct_stability(
    importance_data: Dict[int, pd.DataFrame],
    constructs: Dict[str, Dict],
    model_type: str,
    top_k: int = 5
) -> pd.DataFrame:
    """
    Compute construct-level stability metrics across seeds.

    Returns DataFrame with columns:
    - construct, mean_sum_importance, std_sum_importance, mean_mean_importance,
      std_mean_importance, top_k_frequency, avg_rank
    """
    seeds = sorted(importance_data.keys())

    # Aggregate for each seed
    seed_aggregations = {}
    for seed in seeds:
        df = importance_data[seed]
        seed_aggregations[seed] = aggregate_to_constructs(df, constructs, model_type)

    # Compute per-construct stability
    construct_stats = []

    for construct_name in constructs.keys():
        sum_importances = []
        mean_importances = []
        ranks = []

        for seed in seeds:
            agg = seed_aggregations[seed]
            if construct_name in agg:
                sum_importances.append(agg[construct_name]["sum_importance"])
                mean_importances.append(agg[construct_name]["mean_importance"])

        # Compute ranks per seed
        for seed in seeds:
            # Rank constructs by sum_importance for this seed
            seed_ranking = sorted(
                [(c, seed_aggregations[seed][c]["sum_importance"]) for c in constructs.keys()],
                key=lambda x: x[1],
                reverse=True
            )
            for rank, (c, _) in enumerate(seed_ranking, 1):
                if c == construct_name:
                    ranks.append(rank)
                    break

        # Count frequency in top-K
        top_k_count = sum(1 for r in ranks if r <= top_k)

        construct_stats.append({
            "construct": construct_name,
            "mean_sum_importance": np.mean(sum_importances) if sum_importances else 0,
            "std_sum_importance": np.std(sum_importances) if len(sum_importances) > 1 else 0,
            "mean_mean_importance": np.mean(mean_importances) if mean_importances else 0,
            "std_mean_importance": np.std(mean_importances) if len(mean_importances) > 1 else 0,
            "top_k_frequency": top_k_count,
            "avg_rank": np.mean(ranks) if ranks else 999,
            "std_rank": np.std(ranks) if len(ranks) > 1 else 0,
            "n_seeds": len(sum_importances)
        })

    df = pd.DataFrame(construct_stats)
    df = df.sort_values("mean_sum_importance", ascending=False)

    return df


def get_representative_features(
    importance_data: Dict[int, pd.DataFrame],
    constructs: Dict[str, Dict],
    construct_name: str,
    top_n: int = 3
) -> List[Dict]:
    """
    Get representative features for a construct based on cross-seed stability.

    Returns features that appear most consistently across seeds with highest
    average importance.
    """
    seeds = sorted(importance_data.keys())
    construct_def = constructs[construct_name]

    # Collect feature importance across seeds
    feature_data = {}

    for seed in seeds:
        df = importance_data[seed]
        for _, row in df.iterrows():
            feature = row["feature"]
            if is_excluded(feature):
                continue
            if feature_matches_construct(feature, construct_def):
                if feature not in feature_data:
                    feature_data[feature] = {"importances": [], "seeds": []}
                feature_data[feature]["importances"].append(abs(row["importance"]))
                feature_data[feature]["seeds"].append(seed)

    # Score features by frequency * mean importance
    scored_features = []
    for feature, data in feature_data.items():
        freq = len(data["seeds"])
        mean_imp = np.mean(data["importances"])
        scored_features.append({
            "feature": feature,
            "frequency": freq,
            "mean_importance": mean_imp,
            "stability_score": freq * mean_imp
        })

    # Sort by stability score
    scored_features.sort(key=lambda x: x["stability_score"], reverse=True)

    return scored_features[:top_n]


def main():
    # Configuration
    project_dir = Path("C:/Users/Dusan/compatibility_pipeline")
    runs_dir = project_dir / "artifacts" / "runs"
    output_dir = project_dir / "artifacts" / "constructs"

    seeds = [11, 22, 33, 44, 55]
    top_k = 5  # For frequency calculation

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all importance data
    print("Loading feature importance data from all seeds...")
    importance_data = load_all_importance_data(runs_dir, seeds)

    # Compute construct stability for personality
    print("\nComputing personality construct stability...")
    personality_stability = compute_construct_stability(
        importance_data["personality"],
        PERSONALITY_CONSTRUCTS,
        "personality",
        top_k=top_k
    )
    print(personality_stability.to_string(index=False))

    # Compute construct stability for interests
    print("\nComputing interests construct stability...")
    interests_stability = compute_construct_stability(
        importance_data["interests"],
        INTERESTS_CONSTRUCTS,
        "interests",
        top_k=top_k
    )
    print(interests_stability.to_string(index=False))

    # Save construct stability reports
    personality_stability.to_csv(output_dir / "personality_construct_stability.csv", index=False)
    interests_stability.to_csv(output_dir / "interests_construct_stability.csv", index=False)
    print(f"\nSaved: {output_dir / 'personality_construct_stability.csv'}")
    print(f"Saved: {output_dir / 'interests_construct_stability.csv'}")

    # Get representative features for top constructs
    print("\n" + "=" * 60)
    print("REPRESENTATIVE FEATURES FOR TOP CONSTRUCTS")
    print("=" * 60)

    results = {
        "personality": {
            "construct_stability": personality_stability.to_dict(orient="records"),
            "representative_features": {}
        },
        "interests": {
            "construct_stability": interests_stability.to_dict(orient="records"),
            "representative_features": {}
        }
    }

    # Top personality constructs
    print("\n--- Personality ---")
    for _, row in personality_stability.head(6).iterrows():
        construct = row["construct"]
        print(f"\n{construct}:")
        print(f"  Avg rank: {row['avg_rank']:.1f}, Top-{top_k} frequency: {row['top_k_frequency']}/5")
        reps = get_representative_features(
            importance_data["personality"],
            PERSONALITY_CONSTRUCTS,
            construct,
            top_n=3
        )
        results["personality"]["representative_features"][construct] = reps
        for r in reps:
            print(f"    - {r['feature']}: freq={r['frequency']}/5, mean_imp={r['mean_importance']:.6f}")

    # Top interests constructs
    print("\n--- Interests ---")
    for _, row in interests_stability.head(10).iterrows():
        construct = row["construct"]
        print(f"\n{construct}:")
        print(f"  Avg rank: {row['avg_rank']:.1f}, Top-{top_k} frequency: {row['top_k_frequency']}/5")
        reps = get_representative_features(
            importance_data["interests"],
            INTERESTS_CONSTRUCTS,
            construct,
            top_n=3
        )
        results["interests"]["representative_features"][construct] = reps
        for r in reps:
            print(f"    - {r['feature']}: freq={r['frequency']}/5, mean_imp={r['mean_importance']:.6f}")

    # Save full results as JSON
    # Convert numpy types to Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(i) for i in obj]
        return obj

    results = convert_types(results)

    with open(output_dir / "construct_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_dir / 'construct_analysis.json'}")

    print("\nConstruct aggregation complete!")
    return results


if __name__ == "__main__":
    results = main()
