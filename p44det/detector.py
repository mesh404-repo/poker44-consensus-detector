"""p44det.Detector — the challenger scorer line (served on the second hotkey).

Independent implementation of a bot-detection ensemble. It loads the refreshed
gradient-boosted fleet plus two public reference models, converts each model's
output into a robust component score against snapshot-level reference stats, and
combines them with the rank-consensus aggregation in aggregate.py. The consensus
weighting favours the components with the highest measured transfer to the live
distribution. Low-level per-chunk feature extraction is shared infrastructure.
"""
from __future__ import annotations

import os
import json
import sys
from typing import Any, Dict, List

import numpy as np

from p44det.aggregate import robust_z, trimmed_consensus, logistic_squash

_ENS = os.environ.get("POKER44_ENSEMBLE_DIR", "/root/poker44-heroprofiler/ensemble")
_TRAVIS = os.environ.get("POKER44_TRAVIS_DIR", "/root/poker44-heroprofiler/travis")

# Gradient-boosted fleet trained on the recent-regime releases.
FLEET = ("lgb_new", "xgb_new", "cat_new", "et_new", "rf_new")

# Consensus weights: favour the components that rank the live snapshot best
# (measured transfer / consensus-corr). Deliberately non-uniform, unlike the
# equal-weight z-mean ensemble.
WEIGHTS = {
    "lgb_new": 1.10,
    "xgb_new": 1.25,
    "cat_new": 1.00,
    "et_new": 0.85,
    "rf_new": 0.65,
    "uid174": 1.00,
    "uid208": 1.35,
    "blend": 0.80,
}

# Reference stats for the two public models (snapshot-level).
_M174, _S174 = 0.494, 0.197
_M208, _S208 = 0.311, 0.344
_MBLEND, _SBLEND = 0.274, 0.020


class Detector:
    """Rank-consensus detector over the refreshed fleet + public references."""

    def __init__(self, base=None):
        import joblib

        for p in (os.path.join(_ENS, "uid174"), _TRAVIS):
            if p not in sys.path:
                sys.path.insert(0, p)
        from p44bot.features import extract_features
        from poker44_model.features import chunk_features as cf174, FEATURE_NAMES as fn174
        from poker44_ml.inference import Poker44Model

        self._extract = extract_features
        self._cf174, self._fn174 = cf174, fn174
        self.base = base  # optional blend detector (shares the base scorer)

        sub = os.environ.get("POKER44_DET_SUBDIR", "v4")
        d = os.path.join(_ENS, sub)
        self.fleet = {}
        for nm in FLEET:
            obj = joblib.load(os.path.join(d, "v4_%s.joblib" % nm))
            self.fleet[nm] = (obj["model"], obj["keys"])
        self.ref = json.load(open(os.path.join(d, "v4_ref_stats.json")))
        self.m174 = joblib.load(os.path.join(_ENS, "uid174", "poker44_model", "model.joblib"))
        self.m208 = Poker44Model(os.path.join(_ENS, "uid208_v112.joblib"))

    def _component_scores(self, chunks: List[List[Dict[str, Any]]]):
        """Return an (n_chunks, n_components) matrix of robust component scores
        plus the aligned component-name list."""
        n = len(chunks)
        feats = [self._extract(c) for c in chunks]
        names: List[str] = []
        cols: List[np.ndarray] = []

        # fleet
        base_keys = self.fleet[FLEET[0]][1]
        X0 = np.array([[float(f.get(k, 0.0)) for k in base_keys] for f in feats], dtype=float)
        for nm in FLEET:
            model, keys = self.fleet[nm]
            X = X0 if keys == base_keys else np.array(
                [[float(f.get(k, 0.0)) for k in keys] for f in feats], dtype=float)
            p = model.predict_proba(X)[:, 1]
            r = self.ref[nm]
            cols.append(np.array([robust_z(v, r["mean"], r["std"]) for v in p]))
            names.append(nm)

        # uid174
        X174 = np.array([[float(self._cf174(c).get(k, 0.0)) for k in self._fn174]
                         for c in chunks], dtype=float)
        p174 = self.m174.predict_proba(X174)[:, 1]
        cols.append(np.array([robust_z(v, _M174, _S174) for v in p174]))
        names.append("uid174")

        # uid208
        p208 = np.asarray(self.m208.predict_chunk_scores(chunks), dtype=float)
        cols.append(np.array([robust_z(v, _M208, _S208) for v in p208]))
        names.append("uid208")

        # blend (optional; only if a base scorer is wired in)
        if self.base is not None:
            pb = np.asarray(self.base.score_chunks(chunks), dtype=float)
            cols.append(np.array([robust_z(v, _MBLEND, _SBLEND) for v in pb]))
            names.append("blend")

        M = np.vstack(cols).T if cols else np.zeros((n, 0))
        return M, names

    def score_chunks(self, chunks: List[List[Dict[str, Any]]]) -> List[float]:
        if not chunks:
            return []
        M, names = self._component_scores(chunks)
        w = [WEIGHTS.get(nm, 1.0) for nm in names]
        out = []
        for row in M:
            z = trimmed_consensus(list(row), w)
            out.append(logistic_squash(z))
        return [float(min(1.0, max(0.0, v))) for v in out]
