"""Pseudo-labeling module for compatibility scores."""

from .pseudo_labels import (
    generate_pseudo_labels_ipip,
    generate_pseudo_labels_okcupid,
    PseudoLabelConfig
)

__all__ = [
    "generate_pseudo_labels_ipip",
    "generate_pseudo_labels_okcupid",
    "PseudoLabelConfig"
]
