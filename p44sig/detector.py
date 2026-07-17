"""p44sig.Detector — drift-robust signature-uniqueness scorer.

Independent implementation. Where the other detectors feed all ~353 engineered features into a
gradient-boosted fleet, this one deliberately keeps only the small subset whose meaning survives
the benchmark->live shift: signature-uniqueness, entropy and share ratios. Rationale: a bot runs a
mechanical policy, so it repeats street/amount signatures (low uniqueness), and because these are
ratios they are invariant to table structure -- which is exactly what moves between the data we can
train on and the data we are scored on. Count/stack/player features are excluded: they are constant
in training and vary live, so a tree can only extrapolate noise from them.

Scores are recentred per window (monotone) so the 0.5 decision boundary sits where the reward wants
it: the validator reads human-safety and calibration off the hard 0.5 threshold, while AP and
recall are rank-based and therefore untouched by the shift.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

import numpy as np

_ENS = os.environ.get("POKER44_ENSEMBLE_DIR", "/root/poker44-heroprofiler/ensemble")

# Pin the flag rate: hard_fpr stays under the 0.10 cliff even if every flagged chunk were human,
# while still flagging enough that a true positive clears 0.5 (none => q=0 => reward 0).
_TARGET_PPR = 0.05
_MIN_CAL_N = 20


class Detector:
    """Bot detector over the drift-robust feature core."""

    def __init__(self, base=None):
        import joblib

        from p44bot.features import extract_features

        self._extract = extract_features
        self.base = base  # unused; interface parity

        path = os.environ.get(
            "POKER44_SIG_MODEL",
            os.path.join(os.path.dirname(__file__), "sig_model.joblib"),
        )
        obj = joblib.load(path)
        self.model, self.keys = obj["model"], obj["keys"]

    def score_chunks(self, chunks: List[List[Dict[str, Any]]]) -> List[float]:
        if not chunks:
            return []
        feats = [self._extract(c) for c in chunks]
        X = np.array([[float(f.get(k, 0.0)) for k in self.keys] for f in feats], dtype=float)
        p = np.clip(self.model.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
        z = np.log(p / (1.0 - p))
        if z.size >= _MIN_CAL_N:
            z = z - np.quantile(z, 1.0 - _TARGET_PPR)
        z = np.clip(z, -40.0, 40.0)
        out = 1.0 / (1.0 + np.exp(-z))
        return [float(min(1.0, max(0.0, v))) for v in out]
