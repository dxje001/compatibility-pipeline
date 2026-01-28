"""
Pairwise feature engineering for compatibility modeling.

This module computes features that represent the relationship between
two persons (Person A and Person B), rather than individual persons.

Pairwise Feature Types:
- Absolute difference: |A - B| (captures dissimilarity)
- Element-wise product: A * B (captures interaction)
- Mean: (A + B) / 2 (captures joint level)
- Cosine similarity: cos(A, B) (captures directional similarity)

These features transform the compatibility problem from a
latent-variable problem into a supervised learning problem.
"""

import logging
from typing import List, Tuple, Optional

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


def compute_pairwise_features_ipip(
    raw_features_a: np.ndarray,
    raw_features_b: np.ndarray,
    ocean_features_a: np.ndarray,
    ocean_features_b: np.ndarray,
    feature_types: List[str]
) -> np.ndarray:
    """
    Compute pairwise features for IPIP personality model.

    Creates features that capture the relationship between Person A and Person B
    based on their personality profiles (raw 50 questions + OCEAN aggregates).

    Args:
        raw_features_a: Raw question features for persons A (N x 50)
        raw_features_b: Raw question features for persons B (N x 50)
        ocean_features_a: OCEAN features for persons A (N x 5)
        ocean_features_b: OCEAN features for persons B (N x 5)
        feature_types: List of feature types to compute

    Returns:
        numpy array of pairwise features (N x num_features)
    """
    features = []

    # Combine raw and ocean for convenience
    all_features_a = np.hstack([raw_features_a, ocean_features_a])
    all_features_b = np.hstack([raw_features_b, ocean_features_b])

    if "absolute_difference" in feature_types:
        diff = np.abs(all_features_a - all_features_b)
        features.append(diff)

    if "element_wise_product" in feature_types:
        product = all_features_a * all_features_b
        features.append(product)

    if "mean" in feature_types:
        mean = (all_features_a + all_features_b) / 2
        features.append(mean)

    if "cosine_similarity" in feature_types:
        # Compute cosine similarity for raw features
        raw_cos = _batch_cosine_similarity(raw_features_a, raw_features_b)
        # Compute cosine similarity for OCEAN features
        ocean_cos = _batch_cosine_similarity(ocean_features_a, ocean_features_b)
        # Stack as features
        cos_features = np.column_stack([raw_cos, ocean_cos])
        features.append(cos_features)

    if not features:
        raise ValueError("No valid feature types specified")

    return np.hstack(features)


def compute_pairwise_features_okcupid(
    numeric_a: np.ndarray,
    numeric_b: np.ndarray,
    categorical_a: np.ndarray,
    categorical_b: np.ndarray,
    text_a: csr_matrix,
    text_b: csr_matrix,
    feature_types: List[str]
) -> np.ndarray:
    """
    Compute pairwise features for OkCupid interests model.

    Creates features that capture the relationship between Person A and Person B
    based on their profile attributes (numeric, categorical, text).

    Args:
        numeric_a: Numeric features for persons A (N x num_numeric)
        numeric_b: Numeric features for persons B (N x num_numeric)
        categorical_a: Categorical features for persons A (N x num_categorical)
        categorical_b: Categorical features for persons B (N x num_categorical)
        text_a: TF-IDF features for persons A (N x vocab_size)
        text_b: TF-IDF features for persons B (N x vocab_size)
        feature_types: List of feature types to compute

    Returns:
        numpy array of pairwise features (N x num_features)
    """
    features = []

    # Process numeric features
    if numeric_a.shape[1] > 0:
        if "absolute_difference" in feature_types:
            features.append(np.abs(numeric_a - numeric_b))
        if "element_wise_product" in feature_types:
            features.append(numeric_a * numeric_b)
        if "mean" in feature_types:
            features.append((numeric_a + numeric_b) / 2)
        if "cosine_similarity" in feature_types:
            num_cos = _batch_cosine_similarity(numeric_a, numeric_b)
            features.append(num_cos.reshape(-1, 1))

    # Process categorical features
    if categorical_a.shape[1] > 0:
        if "absolute_difference" in feature_types:
            features.append(np.abs(categorical_a - categorical_b))
        if "element_wise_product" in feature_types:
            # For one-hot encoded features, product gives exact match indicator
            features.append(categorical_a * categorical_b)
        if "cosine_similarity" in feature_types:
            cat_cos = _batch_cosine_similarity(categorical_a, categorical_b)
            features.append(cat_cos.reshape(-1, 1))

    # Process text features (TF-IDF)
    if text_a.shape[1] > 0:
        if "cosine_similarity" in feature_types:
            # Compute text cosine similarity
            text_cos = _batch_cosine_similarity_sparse(text_a, text_b)
            features.append(text_cos.reshape(-1, 1))

        # Note: We don't compute element-wise operations on TF-IDF
        # as the vocabulary is too large and sparse operations are inefficient.
        # Instead, we use aggregate similarity measures.

    if not features:
        raise ValueError("No valid feature types specified or no features available")

    return np.hstack(features)


def _batch_cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute row-wise cosine similarity between two matrices.

    Args:
        a: First matrix (N x D)
        b: Second matrix (N x D)

    Returns:
        Array of cosine similarities (N,)
    """
    # Normalize rows
    norm_a = np.linalg.norm(a, axis=1, keepdims=True)
    norm_b = np.linalg.norm(b, axis=1, keepdims=True)

    # Avoid division by zero
    norm_a = np.where(norm_a == 0, 1, norm_a)
    norm_b = np.where(norm_b == 0, 1, norm_b)

    a_normalized = a / norm_a
    b_normalized = b / norm_b

    # Row-wise dot product
    similarity = np.sum(a_normalized * b_normalized, axis=1)

    return similarity


def _batch_cosine_similarity_sparse(a: csr_matrix, b: csr_matrix) -> np.ndarray:
    """
    Compute row-wise cosine similarity between two sparse matrices.

    Args:
        a: First sparse matrix (N x D)
        b: Second sparse matrix (N x D)

    Returns:
        Array of cosine similarities (N,)
    """
    # Compute norms
    norm_a = np.sqrt(a.multiply(a).sum(axis=1)).A1
    norm_b = np.sqrt(b.multiply(b).sum(axis=1)).A1

    # Avoid division by zero
    norm_a = np.where(norm_a == 0, 1, norm_a)
    norm_b = np.where(norm_b == 0, 1, norm_b)

    # Compute dot products row by row
    dot_products = np.array(a.multiply(b).sum(axis=1)).flatten()

    # Compute cosine similarity
    similarity = dot_products / (norm_a * norm_b)

    return similarity


def get_pairwise_feature_names_ipip(
    raw_feature_names: List[str],
    ocean_feature_names: List[str],
    feature_types: List[str]
) -> List[str]:
    """
    Generate feature names for IPIP pairwise features.

    Args:
        raw_feature_names: Names of raw question features
        ocean_feature_names: Names of OCEAN features
        feature_types: List of feature types used

    Returns:
        List of pairwise feature names
    """
    names = []
    all_names = raw_feature_names + ocean_feature_names

    if "absolute_difference" in feature_types:
        names.extend([f"{n}_diff" for n in all_names])

    if "element_wise_product" in feature_types:
        names.extend([f"{n}_product" for n in all_names])

    if "mean" in feature_types:
        names.extend([f"{n}_mean" for n in all_names])

    if "cosine_similarity" in feature_types:
        names.extend(["raw_cosine_sim", "ocean_cosine_sim"])

    return names


def get_pairwise_feature_names_okcupid(
    numeric_feature_names: List[str],
    categorical_feature_names: List[str],
    has_text_features: bool,
    feature_types: List[str]
) -> List[str]:
    """
    Generate feature names for OkCupid pairwise features.

    Args:
        numeric_feature_names: Names of numeric features
        categorical_feature_names: Names of categorical features
        has_text_features: Whether text features are present
        feature_types: List of feature types used

    Returns:
        List of pairwise feature names
    """
    names = []

    # Numeric feature names
    if numeric_feature_names:
        if "absolute_difference" in feature_types:
            names.extend([f"{n}_diff" for n in numeric_feature_names])
        if "element_wise_product" in feature_types:
            names.extend([f"{n}_product" for n in numeric_feature_names])
        if "mean" in feature_types:
            names.extend([f"{n}_mean" for n in numeric_feature_names])
        if "cosine_similarity" in feature_types:
            names.append("numeric_cosine_sim")

    # Categorical feature names
    if categorical_feature_names:
        if "absolute_difference" in feature_types:
            names.extend([f"{n}_diff" for n in categorical_feature_names])
        if "element_wise_product" in feature_types:
            names.extend([f"{n}_match" for n in categorical_feature_names])
        if "cosine_similarity" in feature_types:
            names.append("categorical_cosine_sim")

    # Text feature name
    if has_text_features and "cosine_similarity" in feature_types:
        names.append("text_cosine_sim")

    return names


def compute_similarity_scores_ipip(
    raw_features_a: np.ndarray,
    raw_features_b: np.ndarray,
    ocean_features_a: np.ndarray,
    ocean_features_b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute raw and OCEAN similarity scores for pseudo-labeling.

    Similarity is computed as (cosine_similarity + 1) / 2 to map to [0, 1].

    Args:
        raw_features_a: Raw question features for persons A
        raw_features_b: Raw question features for persons B
        ocean_features_a: OCEAN features for persons A
        ocean_features_b: OCEAN features for persons B

    Returns:
        Tuple of (sim_raw, sim_ocean) arrays in [0, 1]
    """
    # Compute cosine similarities
    raw_cos = _batch_cosine_similarity(raw_features_a, raw_features_b)
    ocean_cos = _batch_cosine_similarity(ocean_features_a, ocean_features_b)

    # Map from [-1, 1] to [0, 1]
    sim_raw = (raw_cos + 1) / 2
    sim_ocean = (ocean_cos + 1) / 2

    return sim_raw, sim_ocean


def compute_similarity_scores_okcupid(
    numeric_a: np.ndarray,
    numeric_b: np.ndarray,
    categorical_a: np.ndarray,
    categorical_b: np.ndarray,
    text_a: csr_matrix,
    text_b: csr_matrix
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute similarity scores for OkCupid pseudo-labeling.

    Args:
        numeric_a, numeric_b: Numeric features
        categorical_a, categorical_b: Categorical features
        text_a, text_b: Text (TF-IDF) features

    Returns:
        Tuple of (sim_numeric, sim_categorical, sim_text) arrays in [0, 1]
    """
    # Numeric similarity
    if numeric_a.shape[1] > 0:
        num_cos = _batch_cosine_similarity(numeric_a, numeric_b)
        sim_numeric = (num_cos + 1) / 2
    else:
        sim_numeric = np.ones(len(numeric_a)) * 0.5  # Neutral if no features

    # Categorical similarity
    if categorical_a.shape[1] > 0:
        cat_cos = _batch_cosine_similarity(categorical_a, categorical_b)
        sim_categorical = (cat_cos + 1) / 2
    else:
        sim_categorical = np.ones(len(categorical_a)) * 0.5

    # Text similarity (TF-IDF cosine is already in [0, 1] for non-negative vectors)
    if text_a.shape[1] > 0:
        text_cos = _batch_cosine_similarity_sparse(text_a, text_b)
        # TF-IDF vectors are non-negative, so cosine is already in [0, 1]
        sim_text = text_cos
    else:
        sim_text = np.ones(len(numeric_a)) * 0.5

    return sim_numeric, sim_categorical, sim_text
