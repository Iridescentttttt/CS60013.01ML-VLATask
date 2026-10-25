#!/usr/bin/env bash
# 全量数据探究：保留关键词轨迹的全部时间步，然后在正式种子上评测。不替换 v3 Final。
set -euo pipefail

cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TF_CPP_MIN_LOG_LEVEL=2

OPENVLA="/home/lizhenkai/anaconda3/envs/openvla/bin/python"
SIMPLER="/home/lizhenkai/anaconda3/envs/simpler_env/bin/python"
CONFIG="config/eval_states_b_formal.json"
SERVER_PID=""

stop_server() {
  if [[ -n "${SERVER_PID}" ]]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
    SERVER_PID=""
  fi
}
trap stop_server EXIT

echo "train v4 $(date '+%F %T')"
"$OPENVLA" scripts/finetune_openvla_bridge.py \
  --output-dir checkpoints/openvla_bridge_lora_v4 \
  --merged-dir models/openvla-7b-bridge-lora-v4 \
  --log-path logs/openvla_bridge_lora_v4.jsonl \
  --max-steps 1200 \
  --val-interval 50 \
  --keep-all-steps

echo "eval v4 $(date '+%F %T')"
"$OPENVLA" scripts/serve_openvla.py --model models/openvla-7b-bridge-lora-v4 --port 8000 &
SERVER_PID=$!
ready=0
for _ in $(seq 1 120); do
  if curl -sf http://127.0.0.1:8000 >/dev/null; then
    ready=1
    break
  fi
  if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    echo "server exited before becoming ready" >&2
    exit 1
  fi
  sleep 2
done
if [[ "${ready}" -ne 1 ]]; then
  echo "server failed to start" >&2
  exit 1
fi
"$SIMPLER" scripts/evaluate_openvla_widowx.py \
  --episodes 10 \
  --model-label v4 \
  --eval-config "${CONFIG}" \
  --output-dir results/b_unified/v4
stop_server
echo "V4_DONE $(date '+%F %T')"
