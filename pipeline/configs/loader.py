"""
Configuration loading and validation.

This module handles loading of YAML configuration files and
validates that all required fields are present.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import yaml

logger = logging.getLogger(__name__)


def load_config(filepath: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        filepath: Path to the YAML configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML is invalid
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {filepath}")

    logger.info(f"Loading configuration from {filepath}")
    with open(filepath, "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Configuration file is empty: {filepath}")

    return config


def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate configuration and return list of warnings/errors.

    Args:
        config: Configuration dictionary

    Returns:
        List of warning/error messages (empty if valid)
    """
    issues = []

    # Check required top-level sections
    required_sections = ["global", "data", "pair_generation", "labeling",
                        "preprocessing", "modeling", "fusion", "evaluation"]

    for section in required_sections:
        if section not in config:
            issues.append(f"Missing required section: {section}")

    # Check data paths
    if "data" in config:
        data = config["data"]
        if "ipip" not in data or "path" not in data.get("ipip", {}):
            issues.append("Missing data.ipip.path")
        if "okcupid" not in data or "path" not in data.get("okcupid", {}):
            issues.append("Missing data.okcupid.path")

    # Check labeling weights sum to 1
    if "labeling" in config:
        labeling = config["labeling"]
        if "personality" in labeling:
            p = labeling["personality"]
            w_ocean = p.get("weight_ocean", 0.7)
            w_raw = p.get("weight_raw", 0.3)
            if abs(w_ocean + w_raw - 1.0) > 0.01:
                issues.append(f"Personality weights don't sum to 1: {w_ocean} + {w_raw}")

        if "interests" in labeling and "weights" in labeling["interests"]:
            w = labeling["interests"]["weights"]
            total = sum(w.values())
            if abs(total - 1.0) > 0.01:
                issues.append(f"Interests weights don't sum to 1: {total}")

    # Check fusion alpha
    if "fusion" in config:
        alpha = config["fusion"].get("alpha", 0.5)
        if not 0 <= alpha <= 1:
            issues.append(f"Fusion alpha must be in [0, 1], got {alpha}")

    # Check random seed is set
    if "global" in config:
        if "random_seed" not in config["global"]:
            issues.append("Missing global.random_seed (required for reproducibility)")

    return issues


def get_config_value(config: Dict[str, Any], path: str, default: Any = None) -> Any:
    """
    Get a nested configuration value using dot notation.

    Args:
        config: Configuration dictionary
        path: Dot-separated path (e.g., "modeling.gradient_boosting.max_iter")
        default: Default value if path doesn't exist

    Returns:
        Configuration value or default
    """
    keys = path.split(".")
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value
