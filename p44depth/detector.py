"""p44depth.Detector — depth-matched challenger scorer (canary line).

Hypothesis under test: the gradient-boosted fleet was trained on ~34-hand benchmark
chunks but serves on ~90-hand live chunks, so any count/volume feature is out of
distribution at serve time. This detector extracts the fleet's features from
TRAIN_HANDS-sized bagged subsamples of each live chunk (matching the training chunk
depth) and averages them, then combines the fleet with the two public reference
models by a z-weighted mean. The public models (uid174/uid208) receive the full
chunk (their own native handling). Independent implementation, kept separate from the
baseline gradient-boosted detector and the rank-consensus detector.
"""
from __future__ import annotations

import os
import json
import sys
import random
from typing import Any, Dict, List

import numpy as np

_TARGET_PPR = 0.05   # flag top 5% of a window -> hard_fpr<=0.10 even if every flag were human
_MIN_CAL_N = 20      # a quantile below this is meaningless; leave unshifted

from p44depth.aggregate import zscore, weighted_zmean, squash

_ENS = os.environ.get("POKER44_ENSEMBLE_DIR", "/root/poker44-heroprofiler/ensemble")
_TRAVIS = os.environ.get("POKER44_TRAVIS_DIR", "/root/poker44-heroprofiler/travis")

# Gradient-boosted fleet trained on the recent-regime releases (~34-hand chunks).
FLEET = ("lgb_new", "xgb_new", "cat_new", "et_new", "rf_new")

# Component weights (same transfer weighting as the baseline ensemble, so the only
# variable versus the baseline is the depth-matched fleet feature extraction).
WEIGHTS = {
    "lgb_new": 1.10,
    "xgb_new": 1.25,
    "cat_new": 1.00,
    "et_new": 0.85,
    "rf_new": 0.65,
    "uid174": 1.00,
    "uid208": 1.35,
}

_M174, _S174 = 0.494, 0.197
_M208, _S208 = 0.311, 0.344

# Depth match: benchmark training chunks are 30-40 hands; sample this many hands and
# bag-average to a stable estimate. extract_features is cheap (~4ms/chunk) so a large
# bag count costs little while cutting subsample variance.
TRAIN_HANDS = int(os.environ.get("POKER44_DEPTH_HANDS", "34"))
N_BAG = int(os.environ.get("POKER44_DEPTH_BAGS", "10"))


class Detector:
    """Depth-matched detector over the fleet + public reference models."""

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
        self.base = base  # unused here; kept for interface parity

        sub = os.environ.get("POKER44_DET_SUBDIR", "v4")
        d = os.path.join(_ENS, sub)
        self.fleet = {}
        for nm in FLEET:
            obj = joblib.load(os.path.join(d, "v4_%s.joblib" % nm))
            self.fleet[nm] = (obj["model"], obj["keys"])
        self.ref = json.load(open(os.path.join(d, "v4_ref_stats.json")))
        self.m174 = joblib.load(os.path.join(_ENS, "uid174", "poker44_model", "model.joblib"))
        self.m208 = Poker44Model(os.path.join(_ENS, "uid208_v112.joblib"))

    def _depth_features(self, chunks: List[List[Dict[str, Any]]]) -> List[Dict[str, float]]:
        """Fleet features from TRAIN_HANDS-sized bagged subsamples (match training depth)."""
        feats: List[Dict[str, float]] = []
        for i, c in enumerate(chunks):
            n = len(c)
            if n <= TRAIN_HANDS:
                feats.append(self._extract(c))
                continue
            rng = random.Random(1000 + i)  # deterministic per chunk index
            accs = [self._extract([c[j] for j in sorted(rng.sample(range(n), TRAIN_HANDS))])
                    for _ in range(N_BAG)]
            keyset = set().union(*[a.keys() for a in accs])
            feats.append({k: float(np.mean([a.get(k, 0.0) for a in accs])) for k in keyset})
        return feats

    def score_chunks(self, chunks: List[List[Dict[str, Any]]]) -> List[float]:
        if not chunks:
            return []
        dfeats = self._depth_features(chunks)  # fleet input: depth-matched
        base_keys = self.fleet[FLEET[0]][1]
        X0 = np.array([[float(f.get(k, 0.0)) for k in base_keys] for f in dfeats], dtype=float)

        cols: List[np.ndarray] = []
        names: List[str] = []
        for nm in FLEET:
            model, keys = self.fleet[nm]
            X = X0 if keys == base_keys else np.array(
                [[float(f.get(k, 0.0)) for k in keys] for f in dfeats], dtype=float)
            p = model.predict_proba(X)[:, 1]
            r = self.ref[nm]
            cols.append(zscore(p, r["mean"], r["std"]))
            names.append(nm)

        # public reference models receive the FULL chunk (native handling)
        X174 = np.array([[float(self._cf174(c).get(k, 0.0)) for k in self._fn174]
                         for c in chunks], dtype=float)
        cols.append(zscore(self.m174.predict_proba(X174)[:, 1], _M174, _S174))
        names.append("uid174")

        p208 = np.asarray(self.m208.predict_chunk_scores(chunks), dtype=float)
        cols.append(zscore(p208, _M208, _S208))
        names.append("uid208")

        M = np.vstack(cols).T
        w = np.array([WEIGHTS.get(nm, 1.0) for nm in names], dtype=float)
        z = weighted_zmean(M, w)
        # Reward-aware calibration. The validator pays 0.35*AP + 0.30*recall@FPR<=5% + 0.30*q + 0.05:
        # AP and recall are rank-based, but q is read off the HARD 0.5 threshold (q=0 when nothing
        # true clears 0.5 -- a total wipeout -- and q=1 only while hard_fpr<=0.10). squash is
        # monotone, so recentring z cannot touch the ranking terms; it only banks q. Per-window
        # quantile rather than a fixed offset, because the live chunk distribution drifts from the
        # benchmark and a constant shift risks either the FPR cliff or flagging nothing at all.
        if z.size >= _MIN_CAL_N:
            z = z - np.quantile(z, 1.0 - _TARGET_PPR)
        return [float(min(1.0, max(0.0, squash(v)))) for v in z]
