# poker44-heroprofiler — SN126 poker bot detector

Bot-detection miner for Poker44 (Bittensor **SN126**). Serves one bot-risk score per
chunk of poker hands; the score is `P(focus player is a bot)`.

The validator metric is rank-based — `0.75 * AveragePrecision + 0.25 * Recall@FPR<=5%`
— so the goal is to rank bot chunks above human chunks. This model does that with a
gradient-boosted tree model over behavioural features.

## Approach

Each chunk is ~30–40 hands featuring the **same focus player** (`metadata.hero_seat`).
We build ~324 features from **miner-visible fields only** (no hole cards, no showdown/
outcome, no board, no button — those are stripped by the validator's payload view):

- **Per-hand metrics → 7-stat aggregation** (mean/std/min/max/q10/q50/q90): action-type
  shares, aggression/passivity, action/actor/street entropy, bet sizing (bb) stats,
  pot-growth dynamics, starting-stack stats, street progression.
- **Sequence-repetition signatures**: top-share and unique-share of per-hand action /
  actor / street / bet-size-bucket sequences (bots repeat exact lines → a strong tell).
- **Hero-centric behaviour** (the label is about the focus player): VPIP, PFR, fold/
  call/raise-to-aggression, per-hero action rates, bet-size regularity, stack tells.

Features that are benchmark artifacts or unreliable in the live payload (aliased seat
ids, raw subsample-sensitive action counts, raw noised absolute pot sizes) are excluded.

## Layout

- `neurons/miner.py` — miner neuron; serves per-chunk risk scores.
- `p44bot/features.py` — chunk feature extraction (shared train/serve).
- `p44bot/inference.py` — model loader + chunk scorer.
- `p44bot/model/` — trained LightGBM booster (`lgbm.txt`) + feature list (`feat_keys.json`).

## Training data

Trained **only** on the public Poker44 benchmark releases
(`https://api.poker44.net/api/v1/benchmark`). No validator-only evaluation data is used.

## Serving

```bash
scripts/setup_host.sh                                   # build venv (bittensor 10.2.1 + lightgbm)
EXTERNAL_IP=<public-ip> AXON_PORT=8091 scripts/run_miner.sh
```

The miner requires a reachable public IP + open axon port (validators query it directly).
