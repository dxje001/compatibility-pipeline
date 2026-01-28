"""
OkCupid profiles data preprocessor.

Handles:
- Missing value imputation for numeric and categorical columns
- Categorical encoding (one-hot)
- Text concatenation and TF-IDF vectorization
- Feature scaling for numeric columns
"""

import logging
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import joblib

logger = logging.getLogger(__name__)


class OkCupidPreprocessor:
    """
    Preprocessor for OkCupid profile data.

    This class handles all preprocessing steps for the OkCupid dataset:
    1. Missing value handling (numeric: median, categorical: 'unknown')
    2. Categorical encoding (one-hot encoding)
    3. Text processing (concatenation + TF-IDF)
    4. Numeric feature scaling

    The preprocessor maintains separate representations for:
    - Numeric features (scaled)
    - Categorical features (one-hot encoded)
    - Text features (TF-IDF vectors)

    These are combined for pairwise feature engineering but kept
    distinct for interpretability.

    Attributes:
        config: Preprocessing configuration
        scaler: Fitted StandardScaler for numeric features
        encoder: Fitted OneHotEncoder for categorical features
        tfidf: Fitted TfidfVectorizer for text features
        numeric_imputation_values: Dict of column -> imputation value
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the preprocessor.

        Args:
            config: Preprocessing config from config.yaml
        """
        self.config = config
        self.scaler: Optional[StandardScaler] = None
        self.encoder: Optional[OneHotEncoder] = None
        self.tfidf: Optional[TfidfVectorizer] = None
        self.numeric_imputation_values: Dict[str, float] = {}
        self._fitted = False

        # Extract column lists from config
        self.text_columns = config.get("text_columns", [])
        self.categorical_columns = config.get("categorical_columns", [])
        self.numeric_columns = config.get("numeric_columns", [])

        # Track which columns actually exist in data
        self.available_text_columns: List[str] = []
        self.available_categorical_columns: List[str] = []
        self.available_numeric_columns: List[str] = []

    def fit(self, df: pd.DataFrame) -> "OkCupidPreprocessor":
        """
        Fit the preprocessor on training data.

        Computes:
        - Available columns (intersection of config and data)
        - Imputation values for numeric columns
        - OneHotEncoder for categorical columns
        - TF-IDF vectorizer for text columns
        - Scaler for numeric columns

        Args:
            df: Raw OkCupid DataFrame

        Returns:
            self (for method chaining)
        """
        logger.info("Fitting OkCupid preprocessor")

        # Determine available columns
        self.available_text_columns = [c for c in self.text_columns if c in df.columns]
        self.available_categorical_columns = [c for c in self.categorical_columns if c in df.columns]
        self.available_numeric_columns = [c for c in self.numeric_columns if c in df.columns]

        logger.info(f"Found {len(self.available_text_columns)} text columns")
        logger.info(f"Found {len(self.available_categorical_columns)} categorical columns")
        logger.info(f"Found {len(self.available_numeric_columns)} numeric columns")

        # Fit numeric preprocessing
        self._fit_numeric(df)

        # Fit categorical preprocessing
        self._fit_categorical(df)

        # Fit text preprocessing
        self._fit_text(df)

        self._fitted = True
        logger.info("OkCupid preprocessor fitting complete")
        return self

    def _fit_numeric(self, df: pd.DataFrame) -> None:
        """Fit numeric column preprocessing (imputation + scaling)."""
        if not self.available_numeric_columns:
            logger.info("No numeric columns to process")
            return

        df_numeric = df[self.available_numeric_columns].copy()

        # Compute imputation values
        strategy = self.config.get("missing_numeric_strategy", "median")
        logger.info(f"Computing numeric imputation values using strategy: {strategy}")

        for col in self.available_numeric_columns:
            if strategy == "mean":
                self.numeric_imputation_values[col] = df_numeric[col].mean()
            elif strategy == "median":
                self.numeric_imputation_values[col] = df_numeric[col].median()
            elif strategy == "-1":
                self.numeric_imputation_values[col] = -1
            else:
                raise ValueError(f"Unknown numeric missing strategy: {strategy}")

            # Handle case where all values are NaN
            if pd.isna(self.numeric_imputation_values[col]):
                self.numeric_imputation_values[col] = 0

        # Apply imputation and fit scaler
        df_imputed = df_numeric.fillna(self.numeric_imputation_values)
        self.scaler = StandardScaler()
        self.scaler.fit(df_imputed)

    def _fit_categorical(self, df: pd.DataFrame) -> None:
        """Fit categorical column preprocessing (imputation + one-hot encoding)."""
        if not self.available_categorical_columns:
            logger.info("No categorical columns to process")
            return

        df_cat = df[self.available_categorical_columns].copy()

        # Fill missing values
        missing_strategy = self.config.get("missing_categorical_strategy", "unknown")
        if missing_strategy == "unknown":
            df_cat = df_cat.fillna("unknown")
        elif missing_strategy == "mode":
            for col in self.available_categorical_columns:
                mode_val = df_cat[col].mode()
                fill_val = mode_val.iloc[0] if len(mode_val) > 0 else "unknown"
                df_cat[col] = df_cat[col].fillna(fill_val)

        # Convert to string (handles mixed types)
        df_cat = df_cat.astype(str)

        # Fit one-hot encoder
        self.encoder = OneHotEncoder(
            sparse_output=True,
            handle_unknown="ignore",
            drop=None  # Keep all categories for interpretability
        )
        self.encoder.fit(df_cat)

        n_features = len(self.encoder.get_feature_names_out())
        logger.info(f"Fitted one-hot encoder with {n_features} output features")

    def _fit_text(self, df: pd.DataFrame) -> None:
        """Fit text column preprocessing (concatenation + TF-IDF)."""
        if not self.available_text_columns:
            logger.info("No text columns to process")
            return

        # Concatenate all text columns
        combined_text = self._concatenate_text_columns(df)

        # Get TF-IDF config
        tfidf_config = self.config.get("tfidf", {})
        max_features = tfidf_config.get("max_features", 5000)
        ngram_range = tuple(tfidf_config.get("ngram_range", [1, 2]))
        min_df = tfidf_config.get("min_df", 5)
        stop_words = tfidf_config.get("stop_words", "english")

        logger.info(f"Fitting TF-IDF: max_features={max_features}, "
                   f"ngram_range={ngram_range}, min_df={min_df}")

        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            stop_words=stop_words,
            dtype=np.float32  # Save memory
        )
        self.tfidf.fit(combined_text)

        logger.info(f"TF-IDF vocabulary size: {len(self.tfidf.vocabulary_)}")

    def _concatenate_text_columns(self, df: pd.DataFrame) -> pd.Series:
        """
        Concatenate all available text columns into a single text per person.

        Args:
            df: DataFrame with text columns

        Returns:
            Series with concatenated text
        """
        text_parts = []
        for col in self.available_text_columns:
            # Convert to string and handle NaN
            text_parts.append(df[col].fillna("").astype(str))

        # Join with space separator
        combined = text_parts[0]
        for part in text_parts[1:]:
            combined = combined + " " + part

        # Clean up whitespace
        combined = combined.str.strip()
        combined = combined.replace("", " ")  # Avoid empty strings

        return combined

    def transform(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, csr_matrix]:
        """
        Transform raw OkCupid data into model-ready features.

        Args:
            df: Raw OkCupid DataFrame

        Returns:
            Tuple of:
            - df_info: DataFrame with person indices (for tracking)
            - numeric_features: numpy array of scaled numeric features
            - categorical_features: numpy array of one-hot encoded features
            - text_features: sparse matrix of TF-IDF features

        Raises:
            RuntimeError: If preprocessor hasn't been fitted
        """
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")

        # Create info DataFrame with original index
        df_info = pd.DataFrame({"original_index": df.index})

        # Transform numeric features
        numeric_features = self._transform_numeric(df)

        # Transform categorical features
        categorical_features = self._transform_categorical(df)

        # Transform text features
        text_features = self._transform_text(df)

        return df_info, numeric_features, categorical_features, text_features

    def _transform_numeric(self, df: pd.DataFrame) -> np.ndarray:
        """Transform numeric columns."""
        if not self.available_numeric_columns:
            return np.zeros((len(df), 0), dtype=np.float32)

        df_numeric = df[self.available_numeric_columns].copy()
        df_imputed = df_numeric.fillna(self.numeric_imputation_values)
        return self.scaler.transform(df_imputed).astype(np.float32)

    def _transform_categorical(self, df: pd.DataFrame) -> np.ndarray:
        """Transform categorical columns."""
        if not self.available_categorical_columns:
            return np.zeros((len(df), 0), dtype=np.float32)

        df_cat = df[self.available_categorical_columns].copy()

        # Fill missing values (same strategy as fit)
        missing_strategy = self.config.get("missing_categorical_strategy", "unknown")
        if missing_strategy == "unknown":
            df_cat = df_cat.fillna("unknown")
        elif missing_strategy == "mode":
            df_cat = df_cat.fillna("unknown")  # Use unknown at transform time

        df_cat = df_cat.astype(str)

        # One-hot encode (returns sparse, convert to dense)
        encoded = self.encoder.transform(df_cat)
        return encoded.toarray().astype(np.float32)

    def _transform_text(self, df: pd.DataFrame) -> csr_matrix:
        """Transform text columns."""
        if not self.available_text_columns:
            return csr_matrix((len(df), 0), dtype=np.float32)

        combined_text = self._concatenate_text_columns(df)
        return self.tfidf.transform(combined_text)

    def fit_transform(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, csr_matrix]:
        """
        Fit and transform in one step.

        Args:
            df: Raw OkCupid DataFrame

        Returns:
            Same as transform()
        """
        self.fit(df)
        return self.transform(df)

    def get_numeric_feature_names(self) -> List[str]:
        """Get list of numeric feature names."""
        return self.available_numeric_columns.copy()

    def get_categorical_feature_names(self) -> List[str]:
        """Get list of one-hot encoded feature names."""
        if self.encoder is None:
            return []
        return list(self.encoder.get_feature_names_out())

    def get_text_feature_names(self) -> List[str]:
        """Get list of TF-IDF feature names."""
        if self.tfidf is None:
            return []
        return list(self.tfidf.get_feature_names_out())

    def get_all_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        return (
            self.get_numeric_feature_names() +
            self.get_categorical_feature_names() +
            self.get_text_feature_names()
        )

    def save(self, filepath: str) -> None:
        """
        Save preprocessor state to disk.

        Args:
            filepath: Path to save the preprocessor
        """
        state = {
            "config": self.config,
            "scaler": self.scaler,
            "encoder": self.encoder,
            "tfidf": self.tfidf,
            "numeric_imputation_values": self.numeric_imputation_values,
            "available_text_columns": self.available_text_columns,
            "available_categorical_columns": self.available_categorical_columns,
            "available_numeric_columns": self.available_numeric_columns,
            "fitted": self._fitted
        }
        joblib.dump(state, filepath)
        logger.info(f"Saved OkCupid preprocessor to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "OkCupidPreprocessor":
        """
        Load preprocessor state from disk.

        Args:
            filepath: Path to the saved preprocessor

        Returns:
            Loaded OkCupidPreprocessor instance
        """
        state = joblib.load(filepath)
        instance = cls(state["config"])
        instance.scaler = state["scaler"]
        instance.encoder = state["encoder"]
        instance.tfidf = state["tfidf"]
        instance.numeric_imputation_values = state["numeric_imputation_values"]
        instance.available_text_columns = state["available_text_columns"]
        instance.available_categorical_columns = state["available_categorical_columns"]
        instance.available_numeric_columns = state["available_numeric_columns"]
        instance._fitted = state["fitted"]
        logger.info(f"Loaded OkCupid preprocessor from {filepath}")
        return instance
