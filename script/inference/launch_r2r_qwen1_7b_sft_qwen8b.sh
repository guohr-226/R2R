#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  launch_r2r_qwen1_7b_sft_qwen8b.sh [GPU_IDS] [extra launch_r2r_server.py args...]
  launch_r2r_qwen1_7b_sft_qwen8b.sh --gpus 0,1

Defaults:
  GPU_IDS=0,1
  PORT=10001
  SMALL_MODEL_PATH=/home/guohurui/workspace/model/Qwen-1.7B-sft/Qwen3-1.7B-Log-SFT
  LARGE_MODEL_PATH=/home/guohurui/workspace/model/Qwen3-8B
  ROUTER_PATH=/home/guohurui/workspace/R2R_router_collections/Qwen3-1.7B+Qwen3-8B/default_router.pt

Examples:
  ./script/inference/launch_r2r_qwen1_7b_sft_qwen8b.sh --gpus 0,1
  GPU_IDS=2,3 PORT=10001 ./script/inference/launch_r2r_qwen1_7b_sft_qwen8b.sh
  ./script/inference/launch_r2r_qwen1_7b_sft_qwen8b.sh 0,1 --threshold 0.45

GPU layout:
  The value passed to --gpus becomes CUDA_VISIBLE_DEVICES.
  If two or more GPUs are visible, the quick model uses visible GPU 0 and
  the reference model uses visible GPU 1 by default.
  If one GPU is visible, both models use visible GPU 0.

Advanced overrides:
  QUICK_BASE_GPU_ID, REF_BASE_GPU_ID, QUICK_TP_SIZE, REF_TP_SIZE
  QUICK_MEM_FRACTION, REF_MEM_FRACTION, OVERLAP_TP_SCHEDULE
  HOST, PORT, LOG_DIR, LOG_FILE, CONFIG_PATH, PYTHON_BIN
  COMPAT_MODEL_DIR, SANITIZE_SMALL_TOKENIZER
EOF
}

is_gpu_list() {
  [[ "$1" =~ ^[[:space:]]*[0-9]+([[:space:]]*,[[:space:]]*[0-9]+)*[[:space:]]*$ ]]
}

is_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd -P)"

GPU_IDS="${GPU_IDS:-}"
PORT="${PORT:-10001}"
HOST="${HOST:-0.0.0.0}"
SMALL_MODEL_PATH="${SMALL_MODEL_PATH:-/home/guohurui/workspace/model/Qwen-1.7B-sft/Qwen3-1.7B-Log-SFT}"
LARGE_MODEL_PATH="${LARGE_MODEL_PATH:-/home/guohurui/workspace/model/Qwen3-8B}"
ROUTER_PATH="${ROUTER_PATH:-/home/guohurui/workspace/R2R_router_collections/Qwen3-1.7B+Qwen3-8B/default_router.pt}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --gpus|--gpu-ids)
      if [[ $# -lt 2 ]]; then
        echo "error: $1 requires a comma-separated GPU list" >&2
        exit 2
      fi
      GPU_IDS="$2"
      shift 2
      ;;
    --port)
      if [[ $# -lt 2 ]]; then
        echo "error: --port requires a value" >&2
        exit 2
      fi
      PORT="$2"
      shift 2
      ;;
    --host)
      if [[ $# -lt 2 ]]; then
        echo "error: --host requires a value" >&2
        exit 2
      fi
      HOST="$2"
      shift 2
      ;;
    --small-model)
      if [[ $# -lt 2 ]]; then
        echo "error: --small-model requires a path" >&2
        exit 2
      fi
      SMALL_MODEL_PATH="$2"
      shift 2
      ;;
    --large-model)
      if [[ $# -lt 2 ]]; then
        echo "error: --large-model requires a path" >&2
        exit 2
      fi
      LARGE_MODEL_PATH="$2"
      shift 2
      ;;
    --router)
      if [[ $# -lt 2 ]]; then
        echo "error: --router requires a path" >&2
        exit 2
      fi
      ROUTER_PATH="$2"
      shift 2
      ;;
    --config-path)
      if [[ $# -lt 2 ]]; then
        echo "error: --config-path requires a path" >&2
        exit 2
      fi
      CONFIG_PATH="$2"
      shift 2
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      if [[ -z "${GPU_IDS}" ]] && is_gpu_list "$1"; then
        GPU_IDS="$1"
      else
        EXTRA_ARGS+=("$1")
      fi
      shift
      ;;
  esac
done

GPU_IDS="${GPU_IDS:-0,1}"
IFS=',' read -r -a RAW_GPU_ARRAY <<< "${GPU_IDS}"
GPU_ARRAY=()
NORMALIZED_GPU_IDS=""
for raw_gpu in "${RAW_GPU_ARRAY[@]}"; do
  gpu="${raw_gpu//[[:space:]]/}"
  if [[ -z "${gpu}" || ! "${gpu}" =~ ^[0-9]+$ ]]; then
    echo "error: invalid GPU id '${raw_gpu}' in GPU_IDS='${GPU_IDS}'" >&2
    exit 2
  fi
  GPU_ARRAY+=("${gpu}")
  if [[ -z "${NORMALIZED_GPU_IDS}" ]]; then
    NORMALIZED_GPU_IDS="${gpu}"
  else
    NORMALIZED_GPU_IDS="${NORMALIZED_GPU_IDS},${gpu}"
  fi
done
GPU_IDS="${NORMALIZED_GPU_IDS}"
VISIBLE_GPU_COUNT="${#GPU_ARRAY[@]}"
if (( VISIBLE_GPU_COUNT < 1 )); then
  echo "error: at least one GPU id is required" >&2
  exit 2
fi

QUICK_TP_SIZE="${QUICK_TP_SIZE:-1}"
if [[ -z "${REF_TP_SIZE:-}" ]]; then
  if (( VISIBLE_GPU_COUNT > 1 )); then
    REF_TP_SIZE="$((VISIBLE_GPU_COUNT - 1))"
  else
    REF_TP_SIZE="1"
  fi
fi
QUICK_BASE_GPU_ID="${QUICK_BASE_GPU_ID:-0}"
if [[ -z "${REF_BASE_GPU_ID:-}" ]]; then
  if (( VISIBLE_GPU_COUNT > 1 )); then
    REF_BASE_GPU_ID="1"
  else
    REF_BASE_GPU_ID="0"
  fi
fi

for numeric_var in QUICK_TP_SIZE REF_TP_SIZE QUICK_BASE_GPU_ID REF_BASE_GPU_ID PORT; do
  if [[ ! "${!numeric_var}" =~ ^[0-9]+$ ]]; then
    echo "error: ${numeric_var} must be a non-negative integer, got '${!numeric_var}'" >&2
    exit 2
  fi
done
if (( QUICK_TP_SIZE < 1 || REF_TP_SIZE < 1 )); then
  echo "error: QUICK_TP_SIZE and REF_TP_SIZE must be >= 1" >&2
  exit 2
fi
if (( QUICK_BASE_GPU_ID + QUICK_TP_SIZE > VISIBLE_GPU_COUNT )); then
  echo "error: quick model GPU range exceeds visible GPU count ${VISIBLE_GPU_COUNT}" >&2
  exit 2
fi
if (( REF_BASE_GPU_ID + REF_TP_SIZE > VISIBLE_GPU_COUNT )); then
  echo "error: reference model GPU range exceeds visible GPU count ${VISIBLE_GPU_COUNT}" >&2
  exit 2
fi

if [[ ! -d "${SMALL_MODEL_PATH}" ]]; then
  echo "error: small model path does not exist: ${SMALL_MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -d "${LARGE_MODEL_PATH}" ]]; then
  echo "error: large model path does not exist: ${LARGE_MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${ROUTER_PATH}" ]]; then
  echo "error: router checkpoint does not exist: ${ROUTER_PATH}" >&2
  exit 1
fi

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/log}"
mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/r2r_qwen1_7b_sft_qwen8b_${TIMESTAMP}.log}"
CONFIG_PATH="${CONFIG_PATH:-${TMPDIR:-/tmp}/r2r_qwen1_7b_sft_qwen8b_${PORT}.json}"
mkdir -p "$(dirname -- "${CONFIG_PATH}")"
COMPAT_MODEL_DIR="${COMPAT_MODEL_DIR:-${TMPDIR:-/tmp}/r2r_qwen1_7b_sft_qwen8b_compat_model}"
SANITIZE_SMALL_TOKENIZER="${SANITIZE_SMALL_TOKENIZER:-1}"

QUICK_MEM_FRACTION="${QUICK_MEM_FRACTION:-0.15}"
REF_MEM_FRACTION="${REF_MEM_FRACTION:-0.80}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-4096}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-8192}"

"${PYTHON_BIN}" - "${CONFIG_PATH}" "${COMPAT_MODEL_DIR}" "${SANITIZE_SMALL_TOKENIZER}" \
  "${SMALL_MODEL_PATH}" "${LARGE_MODEL_PATH}" "${ROUTER_PATH}" \
  "${QUICK_BASE_GPU_ID}" "${REF_BASE_GPU_ID}" \
  "${QUICK_TP_SIZE}" "${REF_TP_SIZE}" \
  "${QUICK_MEM_FRACTION}" "${REF_MEM_FRACTION}" \
  "${MAX_PREFILL_TOKENS}" "${MAX_TOTAL_TOKENS}" <<'PY'
import json
import os
import shutil
import sys
from pathlib import Path

(
    config_path,
    compat_model_dir,
    sanitize_small_tokenizer,
    small_model_path,
    large_model_path,
    router_path,
    quick_base_gpu_id,
    ref_base_gpu_id,
    quick_tp_size,
    ref_tp_size,
    quick_mem_fraction,
    ref_mem_fraction,
    max_prefill_tokens,
    max_total_tokens,
) = sys.argv[1:]

def truthy(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "y", "on"}

def prepare_small_model_path(source: str, compat_dir: str, sanitize: bool) -> str:
    source_path = Path(source).resolve()
    if not sanitize:
        return str(source_path)

    tokenizer_config_path = source_path / "tokenizer_config.json"
    if not tokenizer_config_path.exists():
        return str(source_path)

    with tokenizer_config_path.open("r", encoding="utf-8") as f:
        tokenizer_config = json.load(f)

    extra_special_tokens = tokenizer_config.get("extra_special_tokens")
    if not isinstance(extra_special_tokens, list):
        return str(source_path)

    compat_path = Path(compat_dir).resolve()
    compat_path.mkdir(parents=True, exist_ok=True)

    for item in source_path.iterdir():
        target = compat_path / item.name
        if item.name == "tokenizer_config.json":
            continue
        if target.exists() or target.is_symlink():
            if target.is_symlink() or target.is_file():
                target.unlink()
            elif target.is_dir():
                shutil.rmtree(target)
        os.symlink(item, target, target_is_directory=item.is_dir())

    tokenizer_config.pop("extra_special_tokens", None)
    tokenizer_config.setdefault("additional_special_tokens", extra_special_tokens)
    with (compat_path / "tokenizer_config.json").open("w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return str(compat_path)

effective_small_model_path = prepare_small_model_path(
    small_model_path,
    compat_model_dir,
    truthy(sanitize_small_tokenizer),
)

config = {
    "special_tokens": {
        "think_start": 151667,
        "think_end": 151668,
    },
    "quick": {
        "model_name": "Qwen3-1.7B-Log-SFT",
        "model_path": effective_small_model_path,
        "param": "1.7",
        "tp_size": int(quick_tp_size),
        "base_gpu_id": int(quick_base_gpu_id),
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "mem_fraction_static": float(quick_mem_fraction),
        "disable_cuda_graph": True,
        "max_prefill_tokens": int(max_prefill_tokens),
        "max_total_tokens": int(max_total_tokens)
    },
    "reference": {
        "model_name": "Qwen3-8B",
        "model_path": large_model_path,
        "param": "8",
        "tp_size": int(ref_tp_size),
        "base_gpu_id": int(ref_base_gpu_id),
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "mem_fraction_static": float(ref_mem_fraction),
        "disable_cuda_graph": True,
        "max_prefill_tokens": int(max_prefill_tokens),
        "max_total_tokens": int(max_total_tokens),
        "kv_cache_dtype": "auto",
    },
    "router": {
        "router_name": "Qwen3-1.7B-Log-SFT+Qwen3-8B",
        "router_path": router_path,
        "override_init_args": {
            "pretrained_model_name": effective_small_model_path,
        },
    },
}

with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)
    f.write("\n")
PY

if [[ -z "${OVERLAP_TP_SCHEDULE:-}" ]]; then
  if (( VISIBLE_GPU_COUNT == 1 )); then
    OVERLAP_TP_SCHEDULE="1"
  else
    OVERLAP_TP_SCHEDULE="0"
  fi
fi

OVERLAP_ARGS=()
if is_true "${OVERLAP_TP_SCHEDULE}"; then
  OVERLAP_ARGS+=(--overlap-tp-schedule)
fi

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-ALL}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

echo "REPO_ROOT=${REPO_ROOT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "PORT=${PORT}"
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "LOG_FILE=${LOG_FILE}"
echo "Quick model: ${SMALL_MODEL_PATH} (visible GPU base ${QUICK_BASE_GPU_ID}, tp ${QUICK_TP_SIZE})"
if [[ "${SANITIZE_SMALL_TOKENIZER}" != "0" && -d "${COMPAT_MODEL_DIR}" ]]; then
  echo "Quick model compat path: ${COMPAT_MODEL_DIR}"
fi
echo "Reference model: ${LARGE_MODEL_PATH} (visible GPU base ${REF_BASE_GPU_ID}, tp ${REF_TP_SIZE})"
echo "Router: ${ROUTER_PATH}"
echo "OVERLAP_TP_SCHEDULE=${OVERLAP_TP_SCHEDULE}"

if is_true "${DRY_RUN:-0}"; then
  echo "DRY_RUN=1, not launching server."
  exit 0
fi

cd "${REPO_ROOT}"
"${PYTHON_BIN}" "${REPO_ROOT}/script/inference/launch_r2r_server.py" \
  --config-path "${CONFIG_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tp-size-quick "${QUICK_TP_SIZE}" \
  --tp-size-ref "${REF_TP_SIZE}" \
  "${OVERLAP_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
