"""
Pair generation for compatibility modeling.

This module handles random generation of (Person A, Person B) pairs
for pairwise compatibility modeling.

Key Design Decisions:
- Pairs are unordered: (A, B) and (B, A) are considered the same pair
- Self-pairs are excluded: (A, A) is never generated
- Random uniform sampling without stratification
- Reproducible given a random seed
"""

import logging
from typing import Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


class PairGenerator:
    """
    Generator for random person pairs.

    This class generates random pairs of person indices for compatibility
    modeling. It ensures:
    - No self-pairs (i != j)
    - No duplicate pairs (considering order doesn't matter)
    - Reproducible results given a random seed

    Attributes:
        max_pairs: Maximum number of pairs to generate
        small_dataset_multiplier: For small datasets, cap at n * multiplier
        random_state: Numpy RandomState for reproducibility
    """

    def __init__(
        self,
        max_pairs: int = 200000,
        small_dataset_multiplier: int = 50,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the pair generator.

        Args:
            max_pairs: Maximum number of pairs to generate
            small_dataset_multiplier: Multiplier for small dataset cap
            random_seed: Random seed for reproducibility
        """
        self.max_pairs = max_pairs
        self.small_dataset_multiplier = small_dataset_multiplier
        self.random_state = np.random.RandomState(random_seed)

    def generate_pairs(self, n_persons: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate random pairs of person indices.

        The number of pairs generated is:
        min(max_pairs, n_persons * small_dataset_multiplier, n_persons * (n_persons - 1) / 2)

        Args:
            n_persons: Total number of persons in the dataset

        Returns:
            Tuple of (indices_a, indices_b) arrays where each (indices_a[i], indices_b[i])
            represents a pair. Indices are guaranteed to satisfy indices_a[i] < indices_b[i].
        """
        # Calculate maximum possible unique pairs
        max_possible = n_persons * (n_persons - 1) // 2

        # Determine target number of pairs
        target_pairs = min(
            self.max_pairs,
            n_persons * self.small_dataset_multiplier,
            max_possible
        )

        logger.info(f"Generating {target_pairs} pairs from {n_persons} persons")
        logger.info(f"Max possible pairs: {max_possible}")

        # For small datasets where we can generate all pairs, do so directly
        if target_pairs >= max_possible * 0.5:
            # Generate all pairs and sample
            return self._generate_by_enumeration(n_persons, target_pairs)
        else:
            # Use rejection sampling for efficiency
            return self._generate_by_sampling(n_persons, target_pairs)

    def _generate_by_enumeration(
        self, n_persons: int, target_pairs: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate pairs by enumerating all and sampling.

        More efficient when we need a large fraction of all possible pairs.

        Args:
            n_persons: Number of persons
            target_pairs: Number of pairs to generate

        Returns:
            Tuple of (indices_a, indices_b) arrays
        """
        # Generate all pairs using triangular indices
        indices_a = []
        indices_b = []

        for i in range(n_persons):
            for j in range(i + 1, n_persons):
                indices_a.append(i)
                indices_b.append(j)

        indices_a = np.array(indices_a)
        indices_b = np.array(indices_b)

        # Sample if needed
        if target_pairs < len(indices_a):
            sample_idx = self.random_state.choice(
                len(indices_a), size=target_pairs, replace=False
            )
            indices_a = indices_a[sample_idx]
            indices_b = indices_b[sample_idx]

        logger.info(f"Generated {len(indices_a)} pairs by enumeration")
        return indices_a, indices_b

    def _generate_by_sampling(
        self, n_persons: int, target_pairs: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate pairs by rejection sampling.

        More efficient when we need a small fraction of all possible pairs.

        Args:
            n_persons: Number of persons
            target_pairs: Number of pairs to generate

        Returns:
            Tuple of (indices_a, indices_b) arrays
        """
        # Use a set to track generated pairs
        pairs_set = set()

        # Generate with some oversampling to reduce iterations
        batch_size = min(target_pairs * 2, 1000000)

        while len(pairs_set) < target_pairs:
            # Generate random pairs
            a = self.random_state.randint(0, n_persons, size=batch_size)
            b = self.random_state.randint(0, n_persons, size=batch_size)

            # Process each pair
            for i in range(batch_size):
                if a[i] != b[i]:  # No self-pairs
                    # Canonical ordering: smaller index first
                    pair = (min(a[i], b[i]), max(a[i], b[i]))
                    pairs_set.add(pair)

                    if len(pairs_set) >= target_pairs:
                        break

        # Convert to arrays
        pairs_list = list(pairs_set)[:target_pairs]
        indices_a = np.array([p[0] for p in pairs_list])
        indices_b = np.array([p[1] for p in pairs_list])

        logger.info(f"Generated {len(indices_a)} pairs by sampling")
        return indices_a, indices_b

    def split_pairs(
        self,
        indices_a: np.ndarray,
        indices_b: np.ndarray,
        train_ratio: float = 0.8
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Split pairs into training and validation sets.

        Args:
            indices_a: First person indices
            indices_b: Second person indices
            train_ratio: Fraction of pairs to use for training

        Returns:
            Tuple of ((train_a, train_b), (val_a, val_b))
        """
        n_pairs = len(indices_a)
        n_train = int(n_pairs * train_ratio)

        # Shuffle indices
        shuffle_idx = self.random_state.permutation(n_pairs)

        train_idx = shuffle_idx[:n_train]
        val_idx = shuffle_idx[n_train:]

        train_pairs = (indices_a[train_idx], indices_b[train_idx])
        val_pairs = (indices_a[val_idx], indices_b[val_idx])

        logger.info(f"Split into {len(train_idx)} train and {len(val_idx)} validation pairs")
        return train_pairs, val_pairs


def generate_pairs_for_dataset(
    n_persons: int,
    config: dict,
    random_seed: int
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Convenience function to generate and split pairs.

    Args:
        n_persons: Number of persons in the dataset
        config: Configuration dictionary with pair_generation settings
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of ((train_a, train_b), (val_a, val_b))
    """
    pair_config = config.get("pair_generation", {})

    generator = PairGenerator(
        max_pairs=pair_config.get("max_pairs", 200000),
        small_dataset_multiplier=pair_config.get("small_dataset_multiplier", 50),
        random_seed=random_seed
    )

    indices_a, indices_b = generator.generate_pairs(n_persons)

    train_pairs, val_pairs = generator.split_pairs(
        indices_a, indices_b,
        train_ratio=pair_config.get("train_ratio", 0.8)
    )

    return train_pairs, val_pairs
