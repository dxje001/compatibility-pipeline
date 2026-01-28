"""
Model training for compatibility scoring.

This module handles training of compatibility models that predict
pseudo-labels from pairwise features. The models serve as calibration
layers that stabilize and refine the similarity-based compatibility scores.

Supported Models:
- HistGradientBoostingRegressor (default): Fast, handles missing values
- LogisticRegression: Simple, interpretable (requires binarization)

Model Purpose:
- ML does NOT "discover ground truth" (there is none)
- ML calibrates and stabilizes similarity-based scoring
- ML learns nonlinear combinations of pairwise features
- ML provides feature importance for UI question derivation
"""

import logging
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator
import joblib

logger = logging.getLogger(__name__)


class CompatibilityModelTrainer:
    """
    Trainer for compatibility prediction models.

    This class handles model configuration, training, and persistence
    for both personality and interests compatibility models.

    The trainer supports multiple model types with configurable hyperparameters,
    all specified through the config dictionary for reproducibility.

    Attributes:
        model_type: Type of model ("gradient_boosting" or "logistic_regression")
        config: Model configuration dictionary
        model: Trained sklearn model (after fit is called)
        feature_names: List of feature names
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model_type: Optional[str] = None,
        random_seed: int = 42
    ):
        """
        Initialize the trainer.

        Args:
            config: Configuration dictionary with modeling settings
            model_type: Override model type from config
            random_seed: Random seed for reproducibility
        """
        modeling_config = config.get("modeling", {})

        self.model_type = model_type or modeling_config.get("model_type", "gradient_boosting")
        self.config = modeling_config
        self.random_seed = random_seed
        self.model: Optional[BaseEstimator] = None
        self.feature_names: List[str] = []
        self._fitted = False
        self._X_importance: Optional[np.ndarray] = None
        self._y_importance: Optional[np.ndarray] = None

        logger.info(f"Initialized trainer with model_type={self.model_type}")

    def _create_model(self) -> BaseEstimator:
        """
        Create the sklearn model based on configuration.

        Returns:
            Configured sklearn estimator

        Raises:
            ValueError: If model type is not supported
        """
        if self.model_type == "gradient_boosting":
            gb_config = self.config.get("gradient_boosting", {})
            return HistGradientBoostingRegressor(
                max_iter=gb_config.get("max_iter", 200),
                max_depth=gb_config.get("max_depth", 8),
                learning_rate=gb_config.get("learning_rate", 0.1),
                min_samples_leaf=gb_config.get("min_samples_leaf", 20),
                l2_regularization=gb_config.get("l2_regularization", 0.1),
                early_stopping=gb_config.get("early_stopping", True),
                validation_fraction=gb_config.get("validation_fraction", 0.1),
                n_iter_no_change=gb_config.get("n_iter_no_change", 10),
                random_state=self.random_seed,
                verbose=0
            )
        elif self.model_type == "logistic_regression":
            lr_config = self.config.get("logistic_regression", {})
            return LogisticRegression(
                C=lr_config.get("C", 1.0),
                max_iter=lr_config.get("max_iter", 1000),
                random_state=self.random_seed,
                solver="lbfgs",
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: List[str],
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> "CompatibilityModelTrainer":
        """
        Train the compatibility model.

        Args:
            X_train: Training features (N x D)
            y_train: Training pseudo-labels (N,)
            feature_names: List of feature names
            X_val: Optional validation features
            y_val: Optional validation labels

        Returns:
            self (for method chaining)
        """
        logger.info(f"Training {self.model_type} model")
        logger.info(f"Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")

        self.feature_names = feature_names
        self.model = self._create_model()

        # For logistic regression, binarize labels if needed
        if self.model_type == "logistic_regression":
            threshold = self.config.get("logistic_regression", {}).get(
                "binarization_threshold", 0.5
            )
            y_train_binary = (y_train >= threshold).astype(int)
            logger.info(f"Binarized labels at threshold {threshold}")
            logger.info(f"Class distribution: {np.mean(y_train_binary):.2%} positive")
            self.model.fit(X_train, y_train_binary)
        else:
            self.model.fit(X_train, y_train)

        self._fitted = True
        logger.info("Model training complete")

        # Store subset of data for permutation importance (used for HistGradientBoosting)
        if X_val is not None and y_val is not None:
            # Use validation set for importance computation
            max_samples = min(5000, len(X_val))
            self._X_importance = X_val[:max_samples]
            self._y_importance = y_val[:max_samples]

            val_preds = self.predict(X_val)
            mse = np.mean((val_preds - y_val) ** 2)
            mae = np.mean(np.abs(val_preds - y_val))
            correlation = np.corrcoef(val_preds, y_val)[0, 1]
            logger.info(f"Validation metrics: MSE={mse:.4f}, MAE={mae:.4f}, "
                       f"Correlation={correlation:.4f}")
        else:
            # Use subset of training data
            max_samples = min(5000, len(X_train))
            self._X_importance = X_train[:max_samples]
            self._y_importance = y_train[:max_samples]

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict compatibility scores.

        Args:
            X: Feature matrix (N x D)

        Returns:
            Predicted scores (N,) in [0, 1]
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before predict")

        if self.model_type == "logistic_regression":
            # Use probability of positive class
            probs = self.model.predict_proba(X)
            return probs[:, 1]
        else:
            # Regression predictions, clip to [0, 1]
            preds = self.model.predict(X)
            return np.clip(preds, 0, 1)

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Extract feature importance from the trained model.

        Returns:
            Dictionary mapping feature names to importance scores

        Note:
            For gradient boosting, uses permutation importance.
            For logistic regression, uses absolute coefficient values.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before getting feature importance")

        if self.model_type == "gradient_boosting":
            # Use permutation importance for HistGradientBoostingRegressor
            logger.info("Computing permutation importance (this may take a moment)...")
            result = permutation_importance(
                self.model,
                self._X_importance,
                self._y_importance,
                n_repeats=5,
                random_state=self.random_seed,
                n_jobs=-1
            )
            importances = result.importances_mean
            logger.info("Permutation importance computed")
        elif self.model_type == "logistic_regression":
            # Use absolute coefficient values (higher = more important)
            importances = np.abs(self.model.coef_[0])
        else:
            raise ValueError(f"Feature importance not supported for {self.model_type}")

        # Create name -> importance mapping
        importance_dict = {
            name: float(imp)
            for name, imp in zip(self.feature_names, importances)
        }

        return importance_dict

    def save(self, filepath: str) -> None:
        """
        Save the trained model to disk.

        Args:
            filepath: Path to save the model
        """
        if not self._fitted:
            raise RuntimeError("Cannot save unfitted model")

        state = {
            "model": self.model,
            "model_type": self.model_type,
            "config": self.config,
            "random_seed": self.random_seed,
            "feature_names": self.feature_names,
            "fitted": self._fitted,
            "X_importance": self._X_importance,
            "y_importance": self._y_importance
        }
        joblib.dump(state, filepath, compress=3)
        logger.info(f"Saved model to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "CompatibilityModelTrainer":
        """
        Load a trained model from disk.

        Args:
            filepath: Path to the saved model

        Returns:
            Loaded CompatibilityModelTrainer instance
        """
        state = joblib.load(filepath)

        instance = cls(
            config={"modeling": state["config"]},
            model_type=state["model_type"],
            random_seed=state["random_seed"]
        )
        instance.model = state["model"]
        instance.feature_names = state["feature_names"]
        instance._fitted = state["fitted"]
        instance.config = state["config"]
        instance._X_importance = state.get("X_importance")
        instance._y_importance = state.get("y_importance")

        logger.info(f"Loaded model from {filepath}")
        return instance
