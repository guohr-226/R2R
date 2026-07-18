#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

export R2R_HTTP_REQUEST_TIMEOUT_SEC="${R2R_HTTP_REQUEST_TIMEOUT_SEC:-45}"
export QUICK_MEM_FRACTION="${QUICK_MEM_FRACTION:-0.50}"
export REF_MEM_FRACTION="${REF_MEM_FRACTION:-0.80}"
export MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-512}"
export MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-4096}"
export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-512}"
R2R_THRESHOLD="${R2R_THRESHOLD:-0.9}"

./script/inference/launch_r2r_qwen1_7b_sft_qwen8b.sh \
  --gpus 0,1 \
  --threshold "${R2R_THRESHOLD}" \
  --small-model /home/guohurui/workspace/model/Qwen-1.7B-sft/Qwen3-1.7B-Log-SFT-balanced-r32-best\
  > run.log 2>&1
