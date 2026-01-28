"""
IPIP Big Five data preprocessor.

Handles:
- Missing value imputation
- Reverse scoring of items
- OCEAN aggregate computation
- Feature scaling
"""

import logging
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

logger = logging.getLogger(__name__)


class IPIPPreprocessor:
    """
    Preprocessor for IPIP Big Five personality data.

    This class handles all preprocessing steps for the IPIP dataset:
    1. Missing value handling (mean/median imputation)
    2. Reverse scoring of designated items
    3. OCEAN dimension aggregate computation
    4. Feature scaling (standardization)

    The preprocessor can be fitted on training data and applied to new data,
    ensuring consistent preprocessing across train/validation/inference.

    Attributes:
        mapping: IPIP-50 question to OCEAN dimension mapping
        config: Preprocessing configuration
        scaler: Fitted StandardScaler (after fit is called)
        imputation_values: Dict of column -> imputation value
    """

    def __init__(self, mapping: Dict[str, Any], config: Dict[str, Any]):
        """
        Initialize the preprocessor.

        Args:
            mapping: IPIP-50 mapping from load_ipip_mapping()
            config: Preprocessing config from config.yaml
        """
        self.mapping = mapping
        self.config = config
        self.scaler: Optional[StandardScaler] = None
        self.imputation_values: Dict[str, float] = {}
        self._fitted = False

        # Extract column lists from mapping
        self.all_question_columns = self._get_all_question_columns()
        self.reverse_scored_columns = self._get_reverse_scored_columns()

        # Scale info for reverse scoring
        self.scale_min = mapping["scale"]["min"]
        self.scale_max = mapping["scale"]["max"]

    def _get_all_question_columns(self) -> List[str]:
        """Get list of all question column names."""
        columns = []
        for dim_config in self.mapping["dimensions"].values():
            columns.extend(dim_config["items"])
        return columns

    def _get_reverse_scored_columns(self) -> List[str]:
        """Get list of reverse-scored column names."""
        columns = []
        for dim_config in self.mapping["dimensions"].values():
            if "reverse_scored" in dim_config:
                columns.extend(dim_config["reverse_scored"])
        return columns

    def fit(self, df: pd.DataFrame) -> "IPIPPreprocessor":
        """
        Fit the preprocessor on training data.

        Computes:
        - Imputation values for missing data
        - Scaler parameters for standardization

        Args:
            df: Raw IPIP DataFrame

        Returns:
            self (for method chaining)
        """
        logger.info("Fitting IPIP preprocessor")

        # Validate columns exist
        missing_cols = [c for c in self.all_question_columns if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing IPIP columns in data: {missing_cols}")

        # Extract question columns only
        df_questions = df[self.all_question_columns].copy()

        # Compute imputation values
        strategy = self.config.get("missing_strategy", "mean")
        logger.info(f"Computing imputation values using strategy: {strategy}")

        for col in self.all_question_columns:
            if strategy == "mean":
                self.imputation_values[col] = df_questions[col].mean()
            elif strategy == "median":
                self.imputation_values[col] = df_questions[col].median()
            else:
                raise ValueError(f"Unknown missing strategy: {strategy}")

        # Apply imputation to compute scaler
        df_imputed = df_questions.fillna(self.imputation_values)

        # Apply reverse scoring
        df_reversed = self._apply_reverse_scoring(df_imputed)

        # Compute OCEAN aggregates
        df_with_ocean = self._compute_ocean_aggregates(df_reversed)

        # Fit scaler on all features (raw + OCEAN)
        if self.config.get("scale_features", True):
            logger.info("Fitting feature scaler")
            self.scaler = StandardScaler()
            self.scaler.fit(df_with_ocean)

        self._fitted = True
        logger.info("IPIP preprocessor fitting complete")
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Transform raw IPIP data into model-ready features.

        Args:
            df: Raw IPIP DataFrame

        Returns:
            Tuple of:
            - df_processed: DataFrame with all features (raw + OCEAN)
            - raw_features: numpy array of 50 raw question features (scaled)
            - ocean_features: numpy array of 5 OCEAN aggregates (scaled)

        Raises:
            RuntimeError: If preprocessor hasn't been fitted
        """
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")

        # Extract and validate question columns
        df_questions = df[self.all_question_columns].copy()

        # Apply imputation
        df_imputed = df_questions.fillna(self.imputation_values)

        # Apply reverse scoring
        df_reversed = self._apply_reverse_scoring(df_imputed)

        # Compute OCEAN aggregates
        df_with_ocean = self._compute_ocean_aggregates(df_reversed)

        # Apply scaling if enabled
        if self.scaler is not None:
            scaled_values = self.scaler.transform(df_with_ocean)
            df_processed = pd.DataFrame(
                scaled_values,
                columns=df_with_ocean.columns,
                index=df_with_ocean.index
            )
        else:
            df_processed = df_with_ocean

        # Extract raw and OCEAN features separately
        raw_features = df_processed[self.all_question_columns].values
        ocean_columns = ["ocean_extraversion", "ocean_neuroticism",
                        "ocean_agreeableness", "ocean_conscientiousness", "ocean_openness"]
        ocean_features = df_processed[ocean_columns].values

        return df_processed, raw_features, ocean_features

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Fit and transform in one step.

        Args:
            df: Raw IPIP DataFrame

        Returns:
            Same as transform()
        """
        self.fit(df)
        return self.transform(df)

    def _apply_reverse_scoring(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply reverse scoring to designated items.

        Reverse scoring formula: new_value = (scale_max + scale_min) - original_value
        For typical 1-5 scale: new_value = 6 - original_value

        Args:
            df: DataFrame with imputed values

        Returns:
            DataFrame with reverse-scored items corrected
        """
        df_out = df.copy()
        reverse_value = self.scale_max + self.scale_min

        for col in self.reverse_scored_columns:
            if col in df_out.columns:
                df_out[col] = reverse_value - df_out[col]

        return df_out

    def _compute_ocean_aggregates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute OCEAN dimension aggregates from individual items.

        Each dimension aggregate is the mean of its constituent items
        (after reverse scoring has been applied).

        Args:
            df: DataFrame with reverse-scored values

        Returns:
            DataFrame with original columns + 5 OCEAN aggregate columns
        """
        df_out = df.copy()

        dimension_names = {
            "extraversion": "ocean_extraversion",
            "neuroticism": "ocean_neuroticism",
            "agreeableness": "ocean_agreeableness",
            "conscientiousness": "ocean_conscientiousness",
            "openness": "ocean_openness"
        }

        for dim_key, col_name in dimension_names.items():
            items = self.mapping["dimensions"][dim_key]["items"]
            # Compute mean across items for each person
            df_out[col_name] = df[items].mean(axis=1)

        return df_out

    def get_feature_names(self) -> List[str]:
        """
        Get list of all feature names after preprocessing.

        Returns:
            List of feature names (50 raw + 5 OCEAN)
        """
        ocean_columns = ["ocean_extraversion", "ocean_neuroticism",
                        "ocean_agreeableness", "ocean_conscientiousness", "ocean_openness"]
        return self.all_question_columns + ocean_columns

    def get_raw_feature_names(self) -> List[str]:
        """Get list of raw question feature names."""
        return self.all_question_columns.copy()

    def get_ocean_feature_names(self) -> List[str]:
        """Get list of OCEAN aggregate feature names."""
        return ["ocean_extraversion", "ocean_neuroticism",
                "ocean_agreeableness", "ocean_conscientiousness", "ocean_openness"]

    def save(self, filepath: str) -> None:
        """
        Save preprocessor state to disk.

        Args:
            filepath: Path to save the preprocessor
        """
        state = {
            "mapping": self.mapping,
            "config": self.config,
            "scaler": self.scaler,
            "imputation_values": self.imputation_values,
            "fitted": self._fitted
        }
        joblib.dump(state, filepath)
        logger.info(f"Saved IPIP preprocessor to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "IPIPPreprocessor":
        """
        Load preprocessor state from disk.

        Args:
            filepath: Path to the saved preprocessor

        Returns:
            Loaded IPIPPreprocessor instance
        """
        state = joblib.load(filepath)
        instance = cls(state["mapping"], state["config"])
        instance.scaler = state["scaler"]
        instance.imputation_values = state["imputation_values"]
        instance._fitted = state["fitted"]
        logger.info(f"Loaded IPIP preprocessor from {filepath}")
        return instance
