"""
Compatibility prediction from UI questionnaire responses.

This module provides the inference pipeline that:
1. Accepts questionnaire responses for Person A and Person B
2. Transforms answers into model feature space
3. Computes personality and interests scores
4. Applies late fusion for final compatibility score

The predictor uses trained artifacts (models, preprocessors, configs)
from a specific seed run.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json

import numpy as np
import pandas as pd
import joblib
from scipy.sparse import csr_matrix

from .schema import (
    QuestionnaireResponse,
    PersonalityAnswers,
    InterestsAnswers,
    CompatibilityResult,
    ReligionChoice,
    LifestyleChoice,
    FamilyChoice,
    EducationChoice,
)

logger = logging.getLogger(__name__)


class CompatibilityPredictor:
    """
    Compatibility scoring predictor using trained models.

    Loads artifacts from a trained pipeline run and provides inference
    for new questionnaire responses.

    Attributes:
        artifacts_dir: Path to the artifacts directory
        personality_model: Trained personality model
        interests_model: Trained interests model
        personality_preprocessor: Fitted IPIP preprocessor
        interests_preprocessor: Fitted OkCupid preprocessor
        fusion_config: Late fusion configuration
    """

    def __init__(self, artifacts_dir: str):
        """
        Initialize predictor with trained artifacts.

        Args:
            artifacts_dir: Path to artifacts directory (e.g., artifacts/runs/seed_11)
        """
        self.artifacts_dir = Path(artifacts_dir)
        self._load_artifacts()
        logger.info(f"Initialized CompatibilityPredictor from {artifacts_dir}")

    def _load_artifacts(self) -> None:
        """Load all required artifacts from disk."""
        # Load models (stored as dicts with 'model' key)
        model_dir = self.artifacts_dir / "models"
        personality_artifact = joblib.load(model_dir / "model_personality.joblib")
        interests_artifact = joblib.load(model_dir / "model_interests.joblib")

        # Extract actual model objects from artifact dicts
        self.personality_model = personality_artifact["model"]
        self.interests_model = interests_artifact["model"]
        self.personality_feature_names = personality_artifact.get("feature_names", [])
        self.interests_feature_names = interests_artifact.get("feature_names", [])
        logger.info("Loaded trained models")

        # Load preprocessors
        preprocessor_dir = self.artifacts_dir / "preprocessors"
        self.personality_preprocessor = joblib.load(
            preprocessor_dir / "preprocessor_personality.joblib"
        )
        self.interests_preprocessor = joblib.load(
            preprocessor_dir / "preprocessor_interests.joblib"
        )
        logger.info("Loaded preprocessors")

        # Load fusion config
        config_dir = self.artifacts_dir / "configs"
        with open(config_dir / "fusion_config.json", "r") as f:
            self.fusion_config = json.load(f)
        logger.info(f"Loaded fusion config: alpha={self.fusion_config['alpha']}")

        # Load config used for training
        import yaml
        with open(config_dir / "config_used.yaml", "r") as f:
            self.training_config = yaml.safe_load(f)

        # Extract feature types
        self.feature_types = self.training_config["feature_engineering"]["pairwise_features"]

    def predict(
        self,
        person_a: QuestionnaireResponse,
        person_b: QuestionnaireResponse,
        return_breakdown: bool = False
    ) -> CompatibilityResult:
        """
        Compute compatibility score between two persons.

        Args:
            person_a: Questionnaire response for Person A
            person_b: Questionnaire response for Person B
            return_breakdown: Whether to include detailed breakdown

        Returns:
            CompatibilityResult with scores
        """
        # Compute personality score
        personality_score = self._compute_personality_score(
            person_a.personality, person_b.personality
        )

        # Compute interests score
        interests_score = self._compute_interests_score(
            person_a.interests, person_b.interests
        )

        # Apply late fusion
        alpha = self.fusion_config["alpha"]
        final_score = alpha * personality_score + (1 - alpha) * interests_score

        # Build result
        breakdown = None
        if return_breakdown:
            breakdown = {
                "fusion_alpha": alpha,
                "personality_contribution": alpha * personality_score,
                "interests_contribution": (1 - alpha) * interests_score,
                "dominant_model": "personality" if personality_score > interests_score else "interests"
            }

        return CompatibilityResult(
            personality_score=float(personality_score),
            interests_score=float(interests_score),
            final_score=float(final_score),
            breakdown=breakdown
        )

    def _compute_personality_score(
        self,
        answers_a: PersonalityAnswers,
        answers_b: PersonalityAnswers
    ) -> float:
        """
        Compute personality compatibility score.

        Strategy:
        - Use UI OCEAN answers as the OCEAN dimension scores
        - Fill raw 50-item features with neutral values (3.0)
        - Apply preprocessor scaling
        - Compute pairwise features
        - Apply model

        Args:
            answers_a: Personality answers for Person A
            answers_b: Personality answers for Person B

        Returns:
            Personality compatibility score [0, 1]
        """
        # Get preprocessor state for scaling
        prep = self.personality_preprocessor

        # Build feature DataFrames mimicking training data structure
        # We use neutral values (3.0) for all 50 raw items since we don't have them
        neutral_value = 3.0
        raw_columns = prep["mapping"]["dimensions"]

        # Build row for person A and B with all 50 items set to neutral
        row_data_a = {}
        row_data_b = {}
        for dim_key, dim_config in raw_columns.items():
            for item in dim_config["items"]:
                row_data_a[item] = neutral_value
                row_data_b[item] = neutral_value

        df_a = pd.DataFrame([row_data_a])
        df_b = pd.DataFrame([row_data_b])

        # Apply imputation (fills any missing)
        all_cols = list(row_data_a.keys())
        for col in all_cols:
            if col in prep["imputation_values"]:
                df_a[col] = df_a[col].fillna(prep["imputation_values"][col])
                df_b[col] = df_b[col].fillna(prep["imputation_values"][col])

        # Apply reverse scoring
        scale_min = prep["mapping"]["scale"]["min"]
        scale_max = prep["mapping"]["scale"]["max"]
        reverse_value = scale_max + scale_min

        for dim_config in prep["mapping"]["dimensions"].values():
            if "reverse_scored" in dim_config:
                for col in dim_config["reverse_scored"]:
                    if col in df_a.columns:
                        df_a[col] = reverse_value - df_a[col]
                        df_b[col] = reverse_value - df_b[col]

        # Get raw features (50 items, all neutral after reverse scoring)
        raw_features_a = df_a[all_cols].values
        raw_features_b = df_b[all_cols].values

        # Use UI OCEAN answers directly (these are the actual user inputs)
        # Order: extraversion, neuroticism, agreeableness, conscientiousness, openness
        ocean_a = np.array([[
            float(answers_a.extraversion),
            float(answers_a.neuroticism),
            float(answers_a.agreeableness),
            float(answers_a.conscientiousness),
            float(answers_a.openness)
        ]])
        ocean_b = np.array([[
            float(answers_b.extraversion),
            float(answers_b.neuroticism),
            float(answers_b.agreeableness),
            float(answers_b.conscientiousness),
            float(answers_b.openness)
        ]])

        # Apply scaling if scaler exists
        scaler = prep.get("scaler")
        if scaler is not None:
            # Combine raw + ocean for scaling (same as training)
            combined_a = np.hstack([raw_features_a, ocean_a])
            combined_b = np.hstack([raw_features_b, ocean_b])

            # Scale
            combined_a_scaled = scaler.transform(combined_a)
            combined_b_scaled = scaler.transform(combined_b)

            # Split back
            raw_features_a = combined_a_scaled[:, :50]
            raw_features_b = combined_b_scaled[:, :50]
            ocean_a = combined_a_scaled[:, 50:]
            ocean_b = combined_b_scaled[:, 50:]

        # Compute pairwise features
        pairwise_features = self._compute_personality_pairwise(
            raw_features_a, raw_features_b, ocean_a, ocean_b
        )

        # Apply model
        score = self.personality_model.predict(pairwise_features)[0]

        # Clip to [0, 1]
        score = np.clip(score, 0.0, 1.0)

        return score

    def _compute_personality_pairwise(
        self,
        raw_a: np.ndarray,
        raw_b: np.ndarray,
        ocean_a: np.ndarray,
        ocean_b: np.ndarray
    ) -> np.ndarray:
        """
        Compute pairwise features for personality model.

        Mirrors compute_pairwise_features_ipip from training.
        """
        features = []

        # Combine raw and ocean
        all_a = np.hstack([raw_a, ocean_a])
        all_b = np.hstack([raw_b, ocean_b])

        if "absolute_difference" in self.feature_types:
            features.append(np.abs(all_a - all_b))

        if "element_wise_product" in self.feature_types:
            features.append(all_a * all_b)

        if "mean" in self.feature_types:
            features.append((all_a + all_b) / 2)

        if "cosine_similarity" in self.feature_types:
            # Raw cosine similarity
            raw_cos = self._batch_cosine_similarity(raw_a, raw_b)
            # OCEAN cosine similarity
            ocean_cos = self._batch_cosine_similarity(ocean_a, ocean_b)
            features.append(np.column_stack([raw_cos, ocean_cos]))

        return np.hstack(features)

    def _compute_interests_score(
        self,
        answers_a: InterestsAnswers,
        answers_b: InterestsAnswers
    ) -> float:
        """
        Compute interests compatibility score.

        Strategy:
        - Map UI categorical answers to OkCupid encoding
        - Fill missing categoricals with 'unknown'
        - Use TF-IDF vectorizer for free-text
        - Set numeric features to median values
        - Compute pairwise features
        - Apply model

        Args:
            answers_a: Interests answers for Person A
            answers_b: Interests answers for Person B

        Returns:
            Interests compatibility score [0, 1]
        """
        prep = self.interests_preprocessor

        # Build categorical feature arrays
        categorical_a, categorical_b = self._encode_categoricals(answers_a, answers_b, prep)

        # Build numeric feature arrays (use medians from training)
        numeric_a, numeric_b = self._get_numeric_features(prep)

        # Build text features from free-text field
        text_a, text_b = self._encode_text(answers_a.about_me, answers_b.about_me, prep)

        # Compute pairwise features
        pairwise_features = self._compute_interests_pairwise(
            numeric_a, numeric_b, categorical_a, categorical_b, text_a, text_b
        )

        # Apply model
        score = self.interests_model.predict(pairwise_features)[0]

        # Clip to [0, 1]
        score = np.clip(score, 0.0, 1.0)

        return score

    def _encode_categoricals(
        self,
        answers_a: InterestsAnswers,
        answers_b: InterestsAnswers,
        prep: Dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode categorical answers to match training encoding.

        Maps UI answer enums to OkCupid categorical values.
        """
        encoder = prep["encoder"]
        available_cols = prep["available_categorical_columns"]

        # Create mapping from UI answers to OkCupid column values
        row_a = self._map_interests_to_okcupid(answers_a)
        row_b = self._map_interests_to_okcupid(answers_b)

        # Build DataFrames with all categorical columns
        df_a = pd.DataFrame([row_a])
        df_b = pd.DataFrame([row_b])

        # Ensure all expected columns exist (fill with 'unknown')
        for col in available_cols:
            if col not in df_a.columns:
                df_a[col] = "unknown"
                df_b[col] = "unknown"

        # Reorder to match training order
        df_a = df_a[available_cols].astype(str)
        df_b = df_b[available_cols].astype(str)

        # One-hot encode
        encoded_a = encoder.transform(df_a).toarray().astype(np.float32)
        encoded_b = encoder.transform(df_b).toarray().astype(np.float32)

        return encoded_a, encoded_b

    def _map_interests_to_okcupid(self, answers: InterestsAnswers) -> Dict[str, str]:
        """
        Map UI interests answers to OkCupid column format.

        This handles the mapping between our UI enum values and
        the original OkCupid categorical values.
        """
        row = {}

        # Religion mapping
        religion_map = {
            ReligionChoice.CHRISTIANITY_SERIOUS: "christianity and very serious about it",
            ReligionChoice.CHRISTIANITY_CASUAL: "christianity but not too serious about it",
            ReligionChoice.CATHOLICISM_SERIOUS: "catholicism and very serious about it",
            ReligionChoice.CATHOLICISM_CASUAL: "catholicism and laughing about it",
            ReligionChoice.JUDAISM_SERIOUS: "judaism and very serious about it",
            ReligionChoice.JUDAISM_CASUAL: "judaism and laughing about it",
            ReligionChoice.ISLAM_SERIOUS: "islam and very serious about it",
            ReligionChoice.ISLAM_CASUAL: "islam and laughing about it",
            ReligionChoice.HINDUISM_SERIOUS: "hinduism and very serious about it",
            ReligionChoice.HINDUISM_CASUAL: "hinduism and laughing about it",
            ReligionChoice.BUDDHISM_SERIOUS: "buddhism and very serious about it",
            ReligionChoice.BUDDHISM_CASUAL: "buddhism and laughing about it",
            ReligionChoice.ATHEISM: "atheism",
            ReligionChoice.AGNOSTICISM: "agnosticism",
            ReligionChoice.SPIRITUAL: "other and very serious about it",
            ReligionChoice.OTHER: "other",
            ReligionChoice.UNKNOWN: "unknown",
        }
        row["religion"] = religion_map.get(answers.religion, "unknown")

        # Education mapping
        education_map = {
            EducationChoice.HIGH_SCHOOL: "graduated from high school",
            EducationChoice.SOME_COLLEGE: "dropped out of college/university",
            EducationChoice.TWO_YEAR_COLLEGE: "graduated from two-year college",
            EducationChoice.BACHELORS: "graduated from college/university",
            EducationChoice.MASTERS: "graduated from masters program",
            EducationChoice.PHD_LAW_MD: "graduated from ph.d program",
            EducationChoice.TRADE_SCHOOL: "graduated from two-year college",
            EducationChoice.UNKNOWN: "unknown",
        }
        row["education"] = education_map.get(answers.education, "unknown")

        # Lifestyle mapping (merged drinks/smokes/drugs)
        # We need to set individual columns
        lifestyle_drinks_map = {
            LifestyleChoice.VERY_HEALTHY: "not at all",
            LifestyleChoice.MOSTLY_HEALTHY: "socially",
            LifestyleChoice.MODERATE: "socially",
            LifestyleChoice.RELAXED: "often",
            LifestyleChoice.UNKNOWN: "unknown",
        }
        lifestyle_smokes_map = {
            LifestyleChoice.VERY_HEALTHY: "no",
            LifestyleChoice.MOSTLY_HEALTHY: "no",
            LifestyleChoice.MODERATE: "sometimes",
            LifestyleChoice.RELAXED: "yes",
            LifestyleChoice.UNKNOWN: "unknown",
        }
        lifestyle_drugs_map = {
            LifestyleChoice.VERY_HEALTHY: "never",
            LifestyleChoice.MOSTLY_HEALTHY: "never",
            LifestyleChoice.MODERATE: "sometimes",
            LifestyleChoice.RELAXED: "often",
            LifestyleChoice.UNKNOWN: "unknown",
        }
        row["drinks"] = lifestyle_drinks_map.get(answers.lifestyle, "unknown")
        row["smokes"] = lifestyle_smokes_map.get(answers.lifestyle, "unknown")
        row["drugs"] = lifestyle_drugs_map.get(answers.lifestyle, "unknown")

        # Family mapping (merged status/offspring)
        family_status_map = {
            FamilyChoice.SINGLE_NO_KIDS_WANTS: "single",
            FamilyChoice.SINGLE_NO_KIDS_DOESNT_WANT: "single",
            FamilyChoice.SINGLE_HAS_KIDS: "single",
            FamilyChoice.SINGLE_UNDECIDED: "single",
            FamilyChoice.RELATIONSHIP_NO_KIDS: "seeing someone",
            FamilyChoice.RELATIONSHIP_HAS_KIDS: "seeing someone",
            FamilyChoice.UNKNOWN: "unknown",
        }
        family_offspring_map = {
            FamilyChoice.SINGLE_NO_KIDS_WANTS: "doesn't have kids, but wants them",
            FamilyChoice.SINGLE_NO_KIDS_DOESNT_WANT: "doesn't have kids, and doesn't want any",
            FamilyChoice.SINGLE_HAS_KIDS: "has a kid",
            FamilyChoice.SINGLE_UNDECIDED: "doesn't have kids, but might want them",
            FamilyChoice.RELATIONSHIP_NO_KIDS: "doesn't have kids, but might want them",
            FamilyChoice.RELATIONSHIP_HAS_KIDS: "has kids",
            FamilyChoice.UNKNOWN: "unknown",
        }
        row["status"] = family_status_map.get(answers.family, "unknown")
        row["offspring"] = family_offspring_map.get(answers.family, "unknown")

        # Fill remaining categorical columns with unknown
        # These are not captured by UI: sex, orientation, diet, body_type, job, sign, pets
        for col in ["sex", "orientation", "diet", "body_type", "job", "sign", "pets"]:
            row[col] = "unknown"

        return row

    def _get_numeric_features(self, prep: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get numeric features using median values from training.

        Since UI doesn't capture age/height/income, we use neutral values.
        """
        available_cols = prep["available_numeric_columns"]
        scaler = prep["scaler"]
        imputation_values = prep["numeric_imputation_values"]

        if not available_cols:
            return np.zeros((1, 0), dtype=np.float32), np.zeros((1, 0), dtype=np.float32)

        # Use imputation values (medians from training) for all numeric features
        row = {col: imputation_values.get(col, 0) for col in available_cols}
        df = pd.DataFrame([row])[available_cols]

        # Scale using fitted scaler
        if scaler is not None:
            scaled = scaler.transform(df).astype(np.float32)
        else:
            scaled = df.values.astype(np.float32)

        # Return same values for A and B (since we don't have this info)
        return scaled, scaled

    def _encode_text(
        self,
        text_a: str,
        text_b: str,
        prep: Dict
    ) -> Tuple[csr_matrix, csr_matrix]:
        """
        Encode free-text using TF-IDF vectorizer from training.
        """
        tfidf = prep.get("tfidf")

        if tfidf is None:
            return csr_matrix((1, 0), dtype=np.float32), csr_matrix((1, 0), dtype=np.float32)

        # Transform text
        vec_a = tfidf.transform([text_a])
        vec_b = tfidf.transform([text_b])

        return vec_a, vec_b

    def _compute_interests_pairwise(
        self,
        numeric_a: np.ndarray,
        numeric_b: np.ndarray,
        categorical_a: np.ndarray,
        categorical_b: np.ndarray,
        text_a: csr_matrix,
        text_b: csr_matrix
    ) -> np.ndarray:
        """
        Compute pairwise features for interests model.

        Mirrors compute_pairwise_features_okcupid from training.
        """
        features = []

        # Numeric features
        if numeric_a.shape[1] > 0:
            if "absolute_difference" in self.feature_types:
                features.append(np.abs(numeric_a - numeric_b))
            if "element_wise_product" in self.feature_types:
                features.append(numeric_a * numeric_b)
            if "mean" in self.feature_types:
                features.append((numeric_a + numeric_b) / 2)
            if "cosine_similarity" in self.feature_types:
                num_cos = self._batch_cosine_similarity(numeric_a, numeric_b)
                features.append(num_cos.reshape(-1, 1))

        # Categorical features
        if categorical_a.shape[1] > 0:
            if "absolute_difference" in self.feature_types:
                features.append(np.abs(categorical_a - categorical_b))
            if "element_wise_product" in self.feature_types:
                features.append(categorical_a * categorical_b)
            if "cosine_similarity" in self.feature_types:
                cat_cos = self._batch_cosine_similarity(categorical_a, categorical_b)
                features.append(cat_cos.reshape(-1, 1))

        # Text features
        if text_a.shape[1] > 0:
            if "cosine_similarity" in self.feature_types:
                text_cos = self._batch_cosine_similarity_sparse(text_a, text_b)
                features.append(text_cos.reshape(-1, 1))

        return np.hstack(features)

    @staticmethod
    def _batch_cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute row-wise cosine similarity between two matrices."""
        norm_a = np.linalg.norm(a, axis=1, keepdims=True)
        norm_b = np.linalg.norm(b, axis=1, keepdims=True)

        norm_a = np.where(norm_a == 0, 1, norm_a)
        norm_b = np.where(norm_b == 0, 1, norm_b)

        a_normalized = a / norm_a
        b_normalized = b / norm_b

        similarity = np.sum(a_normalized * b_normalized, axis=1)
        return similarity

    @staticmethod
    def _batch_cosine_similarity_sparse(a: csr_matrix, b: csr_matrix) -> np.ndarray:
        """Compute row-wise cosine similarity between two sparse matrices."""
        norm_a = np.sqrt(a.multiply(a).sum(axis=1)).A1
        norm_b = np.sqrt(b.multiply(b).sum(axis=1)).A1

        norm_a = np.where(norm_a == 0, 1, norm_a)
        norm_b = np.where(norm_b == 0, 1, norm_b)

        dot_products = np.array(a.multiply(b).sum(axis=1)).flatten()
        similarity = dot_products / (norm_a * norm_b)

        return similarity

    def predict_batch(
        self,
        pairs: list,
        return_breakdown: bool = False
    ) -> list:
        """
        Compute compatibility scores for multiple pairs.

        Args:
            pairs: List of (person_a, person_b) tuples
            return_breakdown: Whether to include detailed breakdown

        Returns:
            List of CompatibilityResult objects
        """
        results = []
        for person_a, person_b in pairs:
            result = self.predict(person_a, person_b, return_breakdown)
            results.append(result)
        return results


def create_predictor(artifacts_dir: str) -> CompatibilityPredictor:
    """
    Factory function to create a CompatibilityPredictor.

    Args:
        artifacts_dir: Path to artifacts directory

    Returns:
        Configured CompatibilityPredictor instance
    """
    return CompatibilityPredictor(artifacts_dir)
