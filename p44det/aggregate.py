"""Rank-consensus aggregation for the p44det detector line.

This is an independent aggregation strategy, deliberately distinct from the
z-mean ensemble: each model's probability is turned into a robust component
score, the single most-extreme component is trimmed, the remainder are combined
with a consensus weighting that favours the models which transfer best to the
live distribution, and the result is squashed with a tuned logistic. The design
targets rank-stability on the fixed daily snapshot rather than absolute value.
"""
from __future__ import annotations

import math
from typing import Sequence


def robust_z(p: float, mean: float, std: float) -> float:
    """Standardise a probability against its snapshot-level reference stats."""
    s = std if std and std > 1e-9 else 1e-9
    return (p - mean) / s


def trimmed_consensus(vals: Sequence[float], weights: Sequence[float]) -> float:
    """Weighted mean after dropping the single most-outlying component.

    Trimming the largest |z| suppresses a single disagreeing model (the usual
    cause of rank noise on the snapshot) without discarding the consensus.
    """
    n = len(vals)
    if n == 0:
        return 0.0
    if n <= 2:
        den = sum(weights) or 1.0
        return sum(v * w for v, w in zip(vals, weights)) / den
    drop = max(range(n), key=lambda i: abs(vals[i]))
    num = den = 0.0
    for i, (v, w) in enumerate(zip(vals, weights)):
        if i == drop:
            continue
        num += v * w
        den += w
    return num / (den or 1.0)


def logistic_squash(z: float, gain: float = 1.15, bias: float = 0.0) -> float:
    """Monotone map of the consensus z to (0,1). Rank-preserving by construction."""
    x = gain * z + bias
    x = 40.0 if x > 40.0 else (-40.0 if x < -40.0 else x)
    return 1.0 / (1.0 + math.exp(-x))
