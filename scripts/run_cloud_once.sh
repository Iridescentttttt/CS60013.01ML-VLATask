#!/usr/bin/env bash
set -euo pipefail

export DISPLAY=""
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM=false
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/root/vlalora/config/nvidia_icd.json}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/xdg-runtime-root}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
NVIDIA_PYTHON_LIB="/root/miniconda3/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="$NVIDIA_PYTHON_LIB/cudnn/lib:$NVIDIA_PYTHON_LIB/cublas/lib:$NVIDIA_PYTHON_LIB/cuda_nvrtc/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
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
