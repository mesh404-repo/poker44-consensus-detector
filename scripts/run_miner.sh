#!/usr/bin/env bash
# Poker44 SN126 miner run script.
# Required env: EXTERNAL_IP (this host's public IP).
# Optional env: WALLET_NAME, HOTKEY, AXON_PORT, NETUID, VENV, POKER44_MODEL_REPO_URL.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${WALLET_NAME:=sn96}"
: "${HOTKEY:=vera1}"
: "${AXON_PORT:=8091}"
: "${NETUID:=126}"
: "${VENV:=$REPO/.venv}"
: "${EXTERNAL_IP:?set EXTERNAL_IP to this hosts public IP so validators can reach it}"

export POKER44_ENABLE_REMOTE_VERSION_CHECK=0
export POKER44_MODEL_NAME="${POKER44_MODEL_NAME:-poker44-heroprofiler}"
export POKER44_USE_TRAVIS="0"
export POKER44_USE_V4="1"
export POKER44_V4_SUBDIR="v4"
export POKER44_CAPTURE_FULL="1"
export POKER44_MODEL_VERSION="${POKER44_MODEL_VERSION:-1}"
export POKER44_MODEL_REPO_URL="${POKER44_MODEL_REPO_URL:-https://github.com/hisorhikaneko92-create/poker44-heroprofiler}"
export POKER44_MODEL_REPO_COMMIT="${POKER44_MODEL_REPO_COMMIT:-a31a7600f70f19c55d127eae8076dc11c5e0c266}"
export POKER44_MODEL_OPEN_SOURCE="true"
export POKER44_MODEL_LICENSE="${POKER44_MODEL_LICENSE:-MIT}"
export POKER44_MODEL_FRAMEWORK="${POKER44_MODEL_FRAMEWORK:-lightgbm}"
export POKER44_MODEL_TRAINING_DATA_STATEMENT="${POKER44_MODEL_TRAINING_DATA_STATEMENT:-Trained only on the public Poker44 training benchmark (schema shadow-training-v1) fetched via api.poker44.net/api/v1/benchmark across all released source dates. No private, scraped, or live-validator data is used.}"
export POKER44_MODEL_PRIVATE_DATA_ATTESTATION="${POKER44_MODEL_PRIVATE_DATA_ATTESTATION:-This model does not train on validator-only live evaluation data; only the public Poker44 benchmark releases are used.}"
export POKER44_MODEL_TRAINING_DATA_SOURCES="${POKER44_MODEL_TRAINING_DATA_SOURCES:-poker44-public-benchmark}"
# POKER44_MODEL_REPO_URL should point at the PUBLIC model repo once published.

cd "$REPO"
exec "$VENV/bin/python" neurons/miner.py \
  --netuid "$NETUID" \
  --wallet.name "$WALLET_NAME" \
  --wallet.hotkey "$HOTKEY" \
  --axon.port "$AXON_PORT" \
  --axon.external_ip "$EXTERNAL_IP" \
  --blacklist.force_validator_permit \
  --logging.info
