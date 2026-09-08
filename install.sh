#!/usr/bin/env bash
# Alpharidge V2 — one-shot environment setup for miners & validators.
#
#   ./install.sh
#
# Defaults to CUDA 12.8 torch wheels. Override for a different driver, e.g.:
#   TORCH_INDEX=https://download.pytorch.org/whl/cu121 ./install.sh
#   TORCH_INDEX=https://download.pytorch.org/whl/cpu   ./install.sh   # (not recommended; see hardware reqs)
#
# Creates ./.venv, installs the full ML stack incl. ReFinED, and the spaCy model.
# Determinism / CLI launch env vars are baked into the entrypoints — nothing to set there.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3.12}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

[ -d .venv ] || "$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel

echo ">> Installing PyTorch ($TORCH_INDEX)"
pip install "torch>=2" --index-url "$TORCH_INDEX"

echo ">> Installing requirements + package (+ spaCy model)"
pip install -r requirements.txt
pip install -e .

echo ">> Installing ReFinED (GitHub @V1, --no-deps so it doesn't downgrade torch/transformers)"
pip install --no-deps "git+https://github.com/amazon-science/ReFinED.git@V1"
pip install ujson nltk Unidecode lmdb prettyprint

echo
echo "✅ Environment ready (./.venv)."
echo "Next:"
echo "  1. cp miner_env.example .miner_env      # validators: cp vali_env.example .vali_env"
echo "  2. add your OpenRouter API key to it  (API_KEY=sk-or-...)"
echo "  3. start your neuron with ./.venv/bin/python  (e.g. via pm2)"
echo "     First boot downloads ~44 GB of models (one-time) — let it finish before it serves."
