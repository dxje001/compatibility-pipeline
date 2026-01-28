"""
Main pipeline runner for the compatibility scoring system.

This is the single entrypoint for running the complete offline training pipeline.

Usage:
    python -m pipeline.run --config configs/config.yaml

The pipeline performs the following steps:
1. Load data (IPIP and OkCupid datasets)
2. Preprocess data
3. Generate person pairs
4. Create pseudo-labels
5. Engineer pairwise features
6. Train compatibility models
7. Evaluate models
8. Run stability analysis (multiple seeds)
9. Save all artifacts
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import json

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_logging(log_level: str) -> None:
    """Configure logging level from config."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(level)


def run_pipeline(
    config_path: str,
    single_seed: Optional[int] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run the complete compatibility pipeline.

    Args:
        config_path: Path to the configuration YAML file
        single_seed: If provided, run only with this seed (skip stability analysis)
        output_dir: If provided, write artifacts to this directory instead of config default

    Returns:
        Dictionary with pipeline results and paths to artifacts
    """
    # Import modules here to avoid circular imports
    from .configs import load_config, validate_config
    from .data_loading import load_ipip_data, load_okcupid_data, load_ipip_mapping
    from .preprocessing import IPIPPreprocessor, OkCupidPreprocessor
    from .pair_generation import PairGenerator
    from .feature_engineering import (
        compute_pairwise_features_ipip,
        compute_pairwise_features_okcupid,
        get_pairwise_feature_names_ipip,
        get_pairwise_feature_names_okcupid
    )
    from .labeling import generate_pseudo_labels_ipip, generate_pseudo_labels_okcupid, PseudoLabelConfig
    from .modeling import CompatibilityModelTrainer, extract_feature_importance
    from .fusion import LateFusion, FusionConfig
    from .evaluation import create_evaluation_report, compute_stability_metrics
    from .artifacts import ArtifactManager

    # =========================================================================
    # 1. Load and validate configuration
    # =========================================================================
    logger.info("=" * 60)
    logger.info("COMPATIBILITY PIPELINE - ITERATION 1")
    logger.info("=" * 60)

    config = load_config(config_path)
    issues = validate_config(config)
    if issues:
        for issue in issues:
            logger.warning(f"Config issue: {issue}")

    setup_logging(config.get("global", {}).get("log_level", "INFO"))

    # Get seeds for stability analysis
    if single_seed is not None:
        seeds = [single_seed]
        logger.info(f"Running with single seed: {single_seed}")
    else:
        seeds = config.get("evaluation", {}).get("stability_seeds", [11, 22, 33, 44, 55])
        logger.info(f"Running stability analysis with seeds: {seeds}")

    base_seed = config.get("global", {}).get("random_seed", 42)

    # Setup artifact manager
    effective_output_dir = output_dir or config.get("global", {}).get("output_dir", "artifacts")
    artifact_manager = ArtifactManager(effective_output_dir)

    # =========================================================================
    # 2. Load data
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Loading Data")
    logger.info("=" * 60)

    # Load IPIP data
    ipip_path = config["data"]["ipip"]["path"]
    ipip_mapping_path = config["data"]["ipip"]["mapping_file"]
    ipip_delimiter = config["data"]["ipip"].get("delimiter", "\t")

    try:
        df_ipip = load_ipip_data(ipip_path, delimiter=ipip_delimiter)
        ipip_mapping = load_ipip_mapping(ipip_mapping_path)
        logger.info(f"Loaded IPIP data: {len(df_ipip)} rows")
    except FileNotFoundError as e:
        logger.error(f"IPIP data not found: {e}")
        logger.info("Creating synthetic IPIP data for demonstration...")
        df_ipip, ipip_mapping = _create_synthetic_ipip_data()

    # Load OkCupid data
    okcupid_path = config["data"]["okcupid"]["path"]
    okcupid_delimiter = config["data"]["okcupid"].get("delimiter", ",")

    try:
        df_okcupid = load_okcupid_data(okcupid_path, delimiter=okcupid_delimiter)
        logger.info(f"Loaded OkCupid data: {len(df_okcupid)} rows")
    except FileNotFoundError as e:
        logger.error(f"OkCupid data not found: {e}")
        logger.info("Creating synthetic OkCupid data for demonstration...")
        df_okcupid = _create_synthetic_okcupid_data()

    # =========================================================================
    # 3. Preprocess data
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Preprocessing Data")
    logger.info("=" * 60)

    # IPIP preprocessing
    ipip_preprocessor = IPIPPreprocessor(ipip_mapping, config.get("preprocessing", {}).get("ipip", {}))
    df_ipip_processed, ipip_raw_features, ipip_ocean_features = ipip_preprocessor.fit_transform(df_ipip)
    logger.info(f"IPIP features: {ipip_raw_features.shape[1]} raw + {ipip_ocean_features.shape[1]} OCEAN")

    # OkCupid preprocessing
    okcupid_config = config.get("preprocessing", {}).get("okcupid", {})
    okcupid_config["tfidf"] = config.get("feature_engineering", {}).get("tfidf", {})
    okcupid_preprocessor = OkCupidPreprocessor(okcupid_config)
    df_ok_info, ok_numeric, ok_categorical, ok_text = okcupid_preprocessor.fit_transform(df_okcupid)
    logger.info(f"OkCupid features: {ok_numeric.shape[1]} numeric + "
               f"{ok_categorical.shape[1]} categorical + {ok_text.shape[1]} text")

    # Save preprocessors
    artifact_manager.save_preprocessor(ipip_preprocessor, "personality")
    artifact_manager.save_preprocessor(okcupid_preprocessor, "interests")

    # =========================================================================
    # 4. Run training for each seed (stability analysis)
    # =========================================================================
    all_results = {
        "personality": {"scores": [], "importances": []},
        "interests": {"scores": [], "importances": []}
    }

    feature_types = config.get("feature_engineering", {}).get("pairwise_features", [
        "absolute_difference", "element_wise_product", "mean", "cosine_similarity"
    ])

    for seed_idx, seed in enumerate(seeds):
        logger.info("\n" + "=" * 60)
        logger.info(f"TRAINING RUN {seed_idx + 1}/{len(seeds)} (seed={seed})")
        logger.info("=" * 60)

        # ---------------------------------------------------------------------
        # Generate pairs
        # ---------------------------------------------------------------------
        logger.info("\nGenerating pairs...")

        # IPIP pairs
        ipip_generator = PairGenerator(
            max_pairs=config.get("pair_generation", {}).get("max_pairs", 200000),
            small_dataset_multiplier=config.get("pair_generation", {}).get("small_dataset_multiplier", 50),
            random_seed=seed
        )
        ipip_pairs_a, ipip_pairs_b = ipip_generator.generate_pairs(len(df_ipip))
        ipip_train, ipip_val = ipip_generator.split_pairs(
            ipip_pairs_a, ipip_pairs_b,
            train_ratio=config.get("pair_generation", {}).get("train_ratio", 0.8)
        )

        # OkCupid pairs
        ok_generator = PairGenerator(
            max_pairs=config.get("pair_generation", {}).get("max_pairs", 200000),
            small_dataset_multiplier=config.get("pair_generation", {}).get("small_dataset_multiplier", 50),
            random_seed=seed + 1000  # Different seed to ensure different pairs
        )
        ok_pairs_a, ok_pairs_b = ok_generator.generate_pairs(len(df_okcupid))
        ok_train, ok_val = ok_generator.split_pairs(
            ok_pairs_a, ok_pairs_b,
            train_ratio=config.get("pair_generation", {}).get("train_ratio", 0.8)
        )

        # ---------------------------------------------------------------------
        # Generate pseudo-labels
        # ---------------------------------------------------------------------
        logger.info("\nGenerating pseudo-labels...")

        pseudo_config = PseudoLabelConfig.from_config(config, seed)

        # IPIP pseudo-labels
        ipip_train_labels = generate_pseudo_labels_ipip(
            ipip_raw_features, ipip_ocean_features,
            ipip_train[0], ipip_train[1], pseudo_config
        )
        ipip_val_labels = generate_pseudo_labels_ipip(
            ipip_raw_features, ipip_ocean_features,
            ipip_val[0], ipip_val[1], pseudo_config
        )

        # OkCupid pseudo-labels
        ok_train_labels = generate_pseudo_labels_okcupid(
            ok_numeric, ok_categorical, ok_text,
            ok_train[0], ok_train[1], pseudo_config
        )
        ok_val_labels = generate_pseudo_labels_okcupid(
            ok_numeric, ok_categorical, ok_text,
            ok_val[0], ok_val[1], pseudo_config
        )

        # ---------------------------------------------------------------------
        # Compute pairwise features
        # ---------------------------------------------------------------------
        logger.info("\nComputing pairwise features...")

        # IPIP pairwise features
        ipip_train_features = compute_pairwise_features_ipip(
            ipip_raw_features[ipip_train[0]], ipip_raw_features[ipip_train[1]],
            ipip_ocean_features[ipip_train[0]], ipip_ocean_features[ipip_train[1]],
            feature_types
        )
        ipip_val_features = compute_pairwise_features_ipip(
            ipip_raw_features[ipip_val[0]], ipip_raw_features[ipip_val[1]],
            ipip_ocean_features[ipip_val[0]], ipip_ocean_features[ipip_val[1]],
            feature_types
        )
        ipip_feature_names = get_pairwise_feature_names_ipip(
            ipip_preprocessor.get_raw_feature_names(),
            ipip_preprocessor.get_ocean_feature_names(),
            feature_types
        )
        logger.info(f"IPIP pairwise features: {ipip_train_features.shape[1]}")

        # OkCupid pairwise features
        ok_train_features = compute_pairwise_features_okcupid(
            ok_numeric[ok_train[0]], ok_numeric[ok_train[1]],
            ok_categorical[ok_train[0]], ok_categorical[ok_train[1]],
            ok_text[ok_train[0]], ok_text[ok_train[1]],
            feature_types
        )
        ok_val_features = compute_pairwise_features_okcupid(
            ok_numeric[ok_val[0]], ok_numeric[ok_val[1]],
            ok_categorical[ok_val[0]], ok_categorical[ok_val[1]],
            ok_text[ok_val[0]], ok_text[ok_val[1]],
            feature_types
        )
        ok_feature_names = get_pairwise_feature_names_okcupid(
            okcupid_preprocessor.get_numeric_feature_names(),
            okcupid_preprocessor.get_categorical_feature_names(),
            ok_text.shape[1] > 0,
            feature_types
        )
        logger.info(f"OkCupid pairwise features: {ok_train_features.shape[1]}")

        # ---------------------------------------------------------------------
        # Train models
        # ---------------------------------------------------------------------
        logger.info("\nTraining models...")

        # Personality model (IPIP)
        personality_trainer = CompatibilityModelTrainer(config, random_seed=seed)
        personality_trainer.fit(
            ipip_train_features, ipip_train_labels, ipip_feature_names,
            ipip_val_features, ipip_val_labels
        )

        # Interests model (OkCupid)
        interests_trainer = CompatibilityModelTrainer(config, random_seed=seed)
        interests_trainer.fit(
            ok_train_features, ok_train_labels, ok_feature_names,
            ok_val_features, ok_val_labels
        )

        # Get predictions on validation set
        personality_val_preds = personality_trainer.predict(ipip_val_features)
        interests_val_preds = interests_trainer.predict(ok_val_features)

        # Store results for stability analysis
        all_results["personality"]["scores"].append(personality_val_preds)
        all_results["personality"]["importances"].append(personality_trainer.get_feature_importance())

        all_results["interests"]["scores"].append(interests_val_preds)
        all_results["interests"]["importances"].append(interests_trainer.get_feature_importance())

        # Save models from last run
        if seed_idx == len(seeds) - 1:
            artifact_manager.save_model(personality_trainer, "personality")
            artifact_manager.save_model(interests_trainer, "interests")

            # Extract and save feature importance
            personality_importance = extract_feature_importance(
                personality_trainer, ipip_feature_names, "personality"
            )
            interests_importance = extract_feature_importance(
                interests_trainer, ok_feature_names, "interests"
            )
            artifact_manager.save_feature_importance(personality_importance, "personality")
            artifact_manager.save_feature_importance(interests_importance, "interests")

            # Save pseudo-label config
            pseudo_config.save(str(artifact_manager.output_dir / "configs" / "pseudo_label_config.json"))

    # =========================================================================
    # 5. Late fusion demonstration
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: Late Fusion")
    logger.info("=" * 60)

    fusion_config = FusionConfig.from_config(config)
    fusion = LateFusion(fusion_config)

    # For demonstration, create some sample fused scores
    # In real use, you'd have matched pairs across datasets
    sample_size = min(1000, len(all_results["personality"]["scores"][-1]),
                      len(all_results["interests"]["scores"][-1]))
    sample_personality = all_results["personality"]["scores"][-1][:sample_size]
    sample_interests = all_results["interests"]["scores"][-1][:sample_size]

    fused_results = fusion.fuse(sample_personality, sample_interests, return_components=True)
    logger.info(f"Fused score statistics:")
    logger.info(f"  Mean: {np.mean(fused_results['final_score']):.4f}")
    logger.info(f"  Std: {np.std(fused_results['final_score']):.4f}")

    fusion_config.save(str(artifact_manager.output_dir / "configs" / "fusion_config.json"))

    # =========================================================================
    # 6. Evaluation and stability analysis
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 6: Evaluation and Stability Analysis")
    logger.info("=" * 60)

    # Personality model evaluation
    personality_report = create_evaluation_report(
        model_name="personality",
        scores=all_results["personality"]["scores"][-1],
        similarity_scores=ipip_val_labels,  # Use pseudo-labels as similarity proxy
        run_scores=all_results["personality"]["scores"] if len(seeds) > 1 else None,
        run_importances=all_results["personality"]["importances"] if len(seeds) > 1 else None
    )
    artifact_manager.save_evaluation_report(personality_report, "personality")
    logger.info("\n" + personality_report.summary())

    # Interests model evaluation
    interests_report = create_evaluation_report(
        model_name="interests",
        scores=all_results["interests"]["scores"][-1],
        similarity_scores=ok_val_labels,
        run_scores=all_results["interests"]["scores"] if len(seeds) > 1 else None,
        run_importances=all_results["interests"]["importances"] if len(seeds) > 1 else None
    )
    artifact_manager.save_evaluation_report(interests_report, "interests")
    logger.info("\n" + interests_report.summary())

    # Stability report
    if len(seeds) > 1:
        stability_report = {
            "n_seeds": len(seeds),
            "seeds_used": seeds,
            "personality": {
                "score_stability": personality_report.stability_metrics.to_dict() if personality_report.stability_metrics else None
            },
            "interests": {
                "score_stability": interests_report.stability_metrics.to_dict() if interests_report.stability_metrics else None
            }
        }
        artifact_manager.save_stability_report(stability_report)

    # =========================================================================
    # 7. Save metadata
    # =========================================================================
    metadata = {
        "pipeline_version": "1.0.0",
        "run_timestamp": datetime.now().isoformat(),
        "config_path": config_path,
        "seeds_used": seeds,
        "ipip_samples": len(df_ipip),
        "okcupid_samples": len(df_okcupid),
        "ipip_pairs_generated": len(ipip_pairs_a),
        "okcupid_pairs_generated": len(ok_pairs_a)
    }
    artifact_manager.save_metadata(metadata)
    artifact_manager.save_yaml_config(config, "config_used")

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)

    artifacts = artifact_manager.list_artifacts()
    logger.info("\nArtifacts saved:")
    for category, files in artifacts.items():
        logger.info(f"  {category}/")
        for f in files:
            logger.info(f"    - {f}")

    return {
        "success": True,
        "output_dir": str(artifact_manager.output_dir),
        "artifacts": artifacts,
        "metadata": metadata
    }


def _create_synthetic_ipip_data() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Create synthetic IPIP data for demonstration when real data is unavailable."""
    import yaml

    n_samples = 1000
    rng = np.random.RandomState(42)

    # Create synthetic responses (1-5 Likert scale)
    columns = []
    for dim, prefix in [("extraversion", "EXT"), ("neuroticism", "EST"),
                        ("agreeableness", "AGR"), ("conscientiousness", "CSN"),
                        ("openness", "OPN")]:
        for i in range(1, 11):
            columns.append(f"{prefix}{i}")

    data = rng.randint(1, 6, size=(n_samples, 50))
    df = pd.DataFrame(data, columns=columns)

    # Load mapping from file if exists, otherwise create minimal mapping
    mapping = {
        "scale": {"min": 1, "max": 5},
        "dimensions": {
            "extraversion": {"items": [f"EXT{i}" for i in range(1, 11)],
                            "reverse_scored": ["EXT2", "EXT4", "EXT6", "EXT8", "EXT10"]},
            "neuroticism": {"items": [f"EST{i}" for i in range(1, 11)],
                          "reverse_scored": ["EST2", "EST4"]},
            "agreeableness": {"items": [f"AGR{i}" for i in range(1, 11)],
                             "reverse_scored": ["AGR1", "AGR3", "AGR5", "AGR7"]},
            "conscientiousness": {"items": [f"CSN{i}" for i in range(1, 11)],
                                  "reverse_scored": ["CSN2", "CSN4", "CSN6", "CSN8"]},
            "openness": {"items": [f"OPN{i}" for i in range(1, 11)],
                        "reverse_scored": ["OPN2", "OPN4", "OPN6"]}
        }
    }

    logger.info(f"Created synthetic IPIP data: {n_samples} samples")
    return df, mapping


def _create_synthetic_okcupid_data() -> pd.DataFrame:
    """Create synthetic OkCupid data for demonstration when real data is unavailable."""
    n_samples = 1000
    rng = np.random.RandomState(43)

    data = {
        "age": rng.randint(18, 60, n_samples),
        "height": rng.normal(170, 10, n_samples).astype(int),
        "income": rng.choice([-1, 20000, 40000, 60000, 80000, 100000], n_samples),
        "sex": rng.choice(["m", "f"], n_samples),
        "orientation": rng.choice(["straight", "gay", "bisexual"], n_samples),
        "status": rng.choice(["single", "seeing someone", "married"], n_samples),
        "drinks": rng.choice(["not at all", "socially", "often"], n_samples),
        "smokes": rng.choice(["no", "sometimes", "yes"], n_samples),
        "drugs": rng.choice(["never", "sometimes"], n_samples),
        "diet": rng.choice(["anything", "vegetarian", "vegan"], n_samples),
        "body_type": rng.choice(["average", "fit", "athletic", "thin"], n_samples),
        "education": rng.choice(["high school", "college", "masters", "phd"], n_samples),
        "job": rng.choice(["tech", "healthcare", "education", "finance", "other"], n_samples),
    }

    # Add essay columns with sample text
    interests = ["hiking", "reading", "music", "movies", "travel", "cooking",
                 "gaming", "art", "sports", "photography"]

    for i in range(10):
        essays = []
        for _ in range(n_samples):
            selected = rng.choice(interests, size=rng.randint(2, 5), replace=False)
            essays.append(f"I enjoy {', '.join(selected)}. Looking for someone who shares my interests.")
        data[f"essay{i}"] = essays

    df = pd.DataFrame(data)
    logger.info(f"Created synthetic OkCupid data: {n_samples} samples")
    return df


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the compatibility scoring pipeline (Iteration 1)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Run with a single seed (skip stability analysis)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for artifacts (overrides config)"
    )

    args = parser.parse_args()

    try:
        result = run_pipeline(args.config, single_seed=args.seed, output_dir=args.output_dir)
        if result["success"]:
            logger.info("\nPipeline completed successfully!")
            return 0
        else:
            logger.error("\nPipeline failed!")
            return 1
    except Exception as e:
        logger.exception(f"Pipeline failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
