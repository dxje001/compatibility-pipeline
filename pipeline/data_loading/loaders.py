"""
Data loading functions for the compatibility pipeline.

This module handles loading raw data from CSV files and configuration from YAML.
No preprocessing is done here - that's handled by the preprocessing module.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_ipip_data(filepath: str, delimiter: str = "\t") -> pd.DataFrame:
    """
    Load IPIP Big Five personality dataset from CSV/TSV.

    The IPIP dataset should contain:
    - 50 Likert-scale question responses (typically 1-5 scale)
    - Optional demographic columns
    - Each row represents one person

    Args:
        filepath: Path to the IPIP data file
        delimiter: Field delimiter (default: tab for standard IPIP datasets)

    Returns:
        DataFrame with raw IPIP data

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is empty or has no valid rows
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"IPIP data file not found: {filepath}")

    logger.info(f"Loading IPIP data from {filepath} (delimiter: {repr(delimiter)})")
    df = pd.read_csv(filepath, sep=delimiter)

    if df.empty:
        raise ValueError(f"IPIP data file is empty: {filepath}")

    logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    return df


def load_okcupid_data(filepath: str, delimiter: str = ",") -> pd.DataFrame:
    """
    Load OkCupid profiles dataset from CSV.

    The OkCupid dataset should contain:
    - Structured categorical attributes (sex, orientation, status, etc.)
    - Numeric attributes (age, height, income)
    - Free-text essay fields (essay0 through essay9)
    - Each row represents one person's profile

    Args:
        filepath: Path to the OkCupid CSV file
        delimiter: Field delimiter (default: comma)

    Returns:
        DataFrame with raw OkCupid data

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is empty or has no valid rows
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"OkCupid data file not found: {filepath}")

    logger.info(f"Loading OkCupid data from {filepath} (delimiter: {repr(delimiter)})")
    df = pd.read_csv(filepath, sep=delimiter)

    if df.empty:
        raise ValueError(f"OkCupid data file is empty: {filepath}")

    logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")

    # Log which essay columns are present
    essay_cols = [c for c in df.columns if c.startswith("essay")]
    logger.info(f"Found {len(essay_cols)} essay columns: {essay_cols}")

    return df


def load_ipip_mapping(filepath: str) -> Dict[str, Any]:
    """
    Load IPIP-50 question to OCEAN dimension mapping from YAML.

    The mapping file specifies:
    - Which columns belong to which OCEAN dimension
    - Which items are reverse-scored
    - Scale min/max for reverse scoring calculation

    Args:
        filepath: Path to the IPIP mapping YAML file

    Returns:
        Dictionary with mapping configuration:
        {
            "scale": {"min": 1, "max": 5},
            "dimensions": {
                "extraversion": {
                    "items": ["EXT1", "EXT2", ...],
                    "reverse_scored": ["EXT2", "EXT4", ...]
                },
                ...
            }
        }

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the mapping is invalid or incomplete
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"IPIP mapping file not found: {filepath}")

    logger.info(f"Loading IPIP mapping from {filepath}")
    with open(filepath, "r") as f:
        mapping = yaml.safe_load(f)

    # Validate mapping structure
    _validate_ipip_mapping(mapping)

    # Count items
    total_items = sum(
        len(dim_config["items"])
        for dim_config in mapping["dimensions"].values()
    )
    logger.info(f"Loaded mapping for {len(mapping['dimensions'])} dimensions, {total_items} items")

    return mapping


def _validate_ipip_mapping(mapping: Dict[str, Any]) -> None:
    """
    Validate the IPIP mapping configuration.

    Checks:
    - Required keys are present
    - All 5 OCEAN dimensions are defined
    - Each dimension has items list
    - Reverse-scored items are subset of items

    Args:
        mapping: The loaded mapping dictionary

    Raises:
        ValueError: If validation fails
    """
    if "scale" not in mapping:
        raise ValueError("IPIP mapping missing 'scale' configuration")

    if "dimensions" not in mapping:
        raise ValueError("IPIP mapping missing 'dimensions' configuration")

    required_dimensions = {
        "extraversion",
        "neuroticism",
        "agreeableness",
        "conscientiousness",
        "openness"
    }

    found_dimensions = set(mapping["dimensions"].keys())
    missing = required_dimensions - found_dimensions
    if missing:
        raise ValueError(f"IPIP mapping missing dimensions: {missing}")

    for dim_name, dim_config in mapping["dimensions"].items():
        if "items" not in dim_config:
            raise ValueError(f"Dimension '{dim_name}' missing 'items' list")

        if not isinstance(dim_config["items"], list):
            raise ValueError(f"Dimension '{dim_name}' items must be a list")

        if len(dim_config["items"]) == 0:
            raise ValueError(f"Dimension '{dim_name}' has no items")

        # Check reverse_scored is subset of items
        if "reverse_scored" in dim_config:
            items_set = set(dim_config["items"])
            reverse_set = set(dim_config["reverse_scored"])
            invalid = reverse_set - items_set
            if invalid:
                raise ValueError(
                    f"Dimension '{dim_name}' has reverse_scored items not in items list: {invalid}"
                )


def get_all_ipip_columns(mapping: Dict[str, Any]) -> List[str]:
    """
    Extract all IPIP question column names from the mapping.

    Args:
        mapping: IPIP mapping dictionary

    Returns:
        List of all question column names across all dimensions
    """
    columns = []
    for dim_config in mapping["dimensions"].values():
        columns.extend(dim_config["items"])
    return columns


def validate_ipip_columns(df: pd.DataFrame, mapping: Dict[str, Any]) -> List[str]:
    """
    Validate that all mapped IPIP columns exist in the DataFrame.

    Args:
        df: IPIP DataFrame
        mapping: IPIP mapping dictionary

    Returns:
        List of missing column names (empty if all present)
    """
    required_columns = get_all_ipip_columns(mapping)
    missing = [c for c in required_columns if c not in df.columns]
    return missing
