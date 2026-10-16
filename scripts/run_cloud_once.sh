#!/usr/bin/env bash
set -euo pipefail

export DISPLAY=""
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="/root/SimplerEnv:/root/octo:${PYTHONPATH:-}"

cd /root/vlalora
python scripts/evaluate_octo_simpler.py \
  --checkpoint /root/models/octo-small-1.5 \
  --checkpoint-step 300000 \
  --task widowx_spoon_on_towel \
  --episodes 1 \
  --seed 0 \
  --output-dir /root/vlalora/results/smoke_spoon
