"""Depth-matched detector aggregation helpers.

Plain z-normalisation + weighted z-mean + logistic squash. Deliberately the simple
z-weighted mean (NOT the rank-consensus used by the consensus detector), so the only
material difference from the baseline gradient-boosted ensemble is the depth-matched
fleet feature extraction implemented in detector.py.
"""
from __future__ import annotations

import numpy as np


def zscore(p, mean: float, std: float) -> np.ndarray:
    """Component-level z-normalisation against snapshot reference stats."""
    return (np.asarray(p, dtype=float) - float(mean)) / max(float(std), 1e-9)


def weighted_zmean(matrix: np.ndarray, weights) -> np.ndarray:
    """Weighted mean across the component axis (rank-preserving within a query)."""
    w = np.asarray(weights, dtype=float)
    denom = float(w.sum()) if w.sum() != 0 else 1.0
    return (np.asarray(matrix, dtype=float) @ w) / denom


def squash(z: float) -> float:
    """Monotone logistic map to [0, 1]; monotone => ranking (and AP) unchanged."""
    z = max(-40.0, min(40.0, float(z)))
    return 1.0 / (1.0 + float(np.exp(-z)))
