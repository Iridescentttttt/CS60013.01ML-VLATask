#!/usr/bin/env bash
set -euo pipefail

export DISPLAY=""
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM=false
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/lvp_icd.x86_64.json}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/xdg-runtime-root}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="/root/SimplerEnv:/root/octo:${PYTHONPATH:-}"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

cd /root/vlalora
python scripts/evaluate_octo_simpler.py \
  --checkpoint /root/models/octo-small-1.5 \
  --checkpoint-step 300000 \
  --task widowx_spoon_on_towel \
  --episodes 1 \
  --seed 0 \
  --output-dir /root/vlalora/results/smoke_spoon
