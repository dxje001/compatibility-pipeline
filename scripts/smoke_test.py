"""
Smoke test for data loading and preprocessing.

This script validates that:
1. Both datasets load correctly
2. Column mappings are valid
3. Basic preprocessing works
4. No runtime errors in the data pipeline

Usage:
    python scripts/smoke_test.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_smoke_test():
    """Run smoke tests on data loading and preprocessing."""

    logger.info("=" * 60)
    logger.info("SMOKE TEST: Data Loading and Preprocessing")
    logger.info("=" * 60)

    # Import modules
    from pipeline.configs import load_config
    from pipeline.data_loading import load_ipip_data, load_okcupid_data, load_ipip_mapping
    from pipeline.data_loading.loaders import validate_ipip_columns
    from pipeline.preprocessing import IPIPPreprocessor, OkCupidPreprocessor

    # Load config
    config_path = project_root / "configs" / "config.yaml"
    logger.info(f"Loading config from {config_path}")
    config = load_config(str(config_path))

    results = {"ipip": {}, "okcupid": {}}

    # =========================================================================
    # Test IPIP Dataset
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("TEST 1: IPIP Big Five Dataset")
    logger.info("=" * 60)

    try:
        ipip_path = config["data"]["ipip"]["path"]
        ipip_delimiter = config["data"]["ipip"].get("delimiter", "\t")
        ipip_mapping_path = config["data"]["ipip"]["mapping_file"]

        # Load data
        logger.info(f"Loading IPIP data from {ipip_path}")
        df_ipip = load_ipip_data(str(project_root / ipip_path), delimiter=ipip_delimiter)

        results["ipip"]["rows"] = len(df_ipip)
        results["ipip"]["columns"] = len(df_ipip.columns)
        logger.info(f"  Rows: {len(df_ipip):,}")
        logger.info(f"  Columns: {len(df_ipip.columns)}")

        # Load mapping
        logger.info(f"Loading IPIP mapping from {ipip_mapping_path}")
        ipip_mapping = load_ipip_mapping(str(project_root / ipip_mapping_path))

        # Validate columns
        missing_cols = validate_ipip_columns(df_ipip, ipip_mapping)
        if missing_cols:
            logger.error(f"  MISSING COLUMNS: {missing_cols}")
            results["ipip"]["status"] = "FAILED - missing columns"
        else:
            logger.info("  All mapped columns present: OK")
            results["ipip"]["status"] = "PASSED"

        # Check question columns for valid values (1-5)
        question_cols = []
        for dim in ipip_mapping["dimensions"].values():
            question_cols.extend(dim["items"])

        df_questions = df_ipip[question_cols]
        min_val = df_questions.min().min()
        max_val = df_questions.max().max()
        logger.info(f"  Question value range: [{min_val}, {max_val}]")

        # Missing values
        missing_pct = df_questions.isna().sum().sum() / df_questions.size * 100
        logger.info(f"  Missing values: {missing_pct:.2f}%")
        results["ipip"]["missing_pct"] = missing_pct

        # Test preprocessing
        logger.info("  Testing preprocessing...")
        ipip_preprocessor = IPIPPreprocessor(ipip_mapping, config.get("preprocessing", {}).get("ipip", {}))

        # Use subset for quick test
        df_sample = df_ipip.head(1000)
        df_processed, raw_features, ocean_features = ipip_preprocessor.fit_transform(df_sample)

        logger.info(f"  Preprocessed features: {raw_features.shape[1]} raw + {ocean_features.shape[1]} OCEAN")
        logger.info(f"  Raw feature shape: {raw_features.shape}")
        logger.info(f"  OCEAN feature shape: {ocean_features.shape}")

        results["ipip"]["preprocessing"] = "PASSED"

    except Exception as e:
        logger.error(f"  IPIP TEST FAILED: {e}")
        results["ipip"]["status"] = f"FAILED - {e}"
        import traceback
        traceback.print_exc()

    # =========================================================================
    # Test OkCupid Dataset
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: OkCupid Profiles Dataset")
    logger.info("=" * 60)

    try:
        okcupid_path = config["data"]["okcupid"]["path"]
        okcupid_delimiter = config["data"]["okcupid"].get("delimiter", ",")

        # Load data
        logger.info(f"Loading OkCupid data from {okcupid_path}")
        df_okcupid = load_okcupid_data(str(project_root / okcupid_path), delimiter=okcupid_delimiter)

        results["okcupid"]["rows"] = len(df_okcupid)
        results["okcupid"]["columns"] = len(df_okcupid.columns)
        logger.info(f"  Rows: {len(df_okcupid):,}")
        logger.info(f"  Columns: {len(df_okcupid.columns)}")
        logger.info(f"  Column names: {list(df_okcupid.columns)}")

        # Check expected columns
        okcupid_config = config.get("preprocessing", {}).get("okcupid", {})
        expected_text = okcupid_config.get("text_columns", [])
        expected_cat = okcupid_config.get("categorical_columns", [])
        expected_num = okcupid_config.get("numeric_columns", [])

        found_text = [c for c in expected_text if c in df_okcupid.columns]
        found_cat = [c for c in expected_cat if c in df_okcupid.columns]
        found_num = [c for c in expected_num if c in df_okcupid.columns]

        logger.info(f"  Text columns found: {len(found_text)}/{len(expected_text)}")
        logger.info(f"  Categorical columns found: {len(found_cat)}/{len(expected_cat)}")
        logger.info(f"  Numeric columns found: {len(found_num)}/{len(expected_num)}")

        # Missing values per column type
        if found_text:
            text_missing = df_okcupid[found_text].isna().sum().sum() / (len(df_okcupid) * len(found_text)) * 100
            logger.info(f"  Text columns missing: {text_missing:.2f}%")

        if found_cat:
            cat_missing = df_okcupid[found_cat].isna().sum().sum() / (len(df_okcupid) * len(found_cat)) * 100
            logger.info(f"  Categorical columns missing: {cat_missing:.2f}%")

        if found_num:
            num_missing = df_okcupid[found_num].isna().sum().sum() / (len(df_okcupid) * len(found_num)) * 100
            logger.info(f"  Numeric columns missing: {num_missing:.2f}%")

        results["okcupid"]["status"] = "PASSED"

        # Test preprocessing
        logger.info("  Testing preprocessing...")
        okcupid_config["tfidf"] = config.get("feature_engineering", {}).get("tfidf", {})
        okcupid_preprocessor = OkCupidPreprocessor(okcupid_config)

        # Use subset for quick test
        df_sample = df_okcupid.head(1000)
        df_info, numeric_features, categorical_features, text_features = okcupid_preprocessor.fit_transform(df_sample)

        logger.info(f"  Numeric features: {numeric_features.shape}")
        logger.info(f"  Categorical features: {categorical_features.shape}")
        logger.info(f"  Text features (TF-IDF): {text_features.shape}")

        results["okcupid"]["preprocessing"] = "PASSED"

    except Exception as e:
        logger.error(f"  OkCupid TEST FAILED: {e}")
        results["okcupid"]["status"] = f"FAILED - {e}"
        import traceback
        traceback.print_exc()

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("SMOKE TEST SUMMARY")
    logger.info("=" * 60)

    all_passed = True
    for dataset, result in results.items():
        status = result.get("status", "UNKNOWN")
        preproc = result.get("preprocessing", "NOT RUN")
        logger.info(f"  {dataset.upper()}:")
        logger.info(f"    Data loading: {status}")
        logger.info(f"    Preprocessing: {preproc}")
        if "FAILED" in status or "FAILED" in preproc:
            all_passed = False

    if all_passed:
        logger.info("\n  ALL TESTS PASSED")
        return 0
    else:
        logger.error("\n  SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(run_smoke_test())
