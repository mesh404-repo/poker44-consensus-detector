#!/usr/bin/env bash
# One-time setup for a serving host: builds an isolated venv with the serving deps.
# Serving needs only bittensor + numpy + lightgbm (model inference is CPU-only).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
python3 -m venv .venv
./.venv/bin/pip install -q --upgrade pip
./.venv/bin/pip install -q "bittensor==10.2.1" "numpy>=1.24" "lightgbm>=4.0"
./.venv/bin/python - <<'PY'
import bittensor, lightgbm, numpy
print("serving deps OK: bittensor", bittensor.__version__, "| lightgbm", lightgbm.__version__, "| numpy", numpy.__version__)
PY
echo "Setup complete."
echo "Next: EXTERNAL_IP=<public-ip> AXON_PORT=8091 scripts/run_miner.sh   (or via pm2)"
