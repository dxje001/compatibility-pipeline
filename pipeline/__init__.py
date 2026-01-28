"""
Compatibility Scoring Pipeline - Iteration 1

This package implements an offline training system for compatibility scoring
based on personality traits (IPIP Big Five) and interests/beliefs (OkCupid profiles).

Key Design Decisions:
- Two independent models trained on separate datasets (no person-level join)
- Late fusion combines model predictions at score level
- Pseudo-labels derived from theoretically-motivated similarity functions
- ML serves as calibration layer, not ground-truth discovery
"""

__version__ = "1.0.0"
