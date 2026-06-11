#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/checkpoints/Qwen3-1.7B}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-$REPO_ROOT/outputs}"
ROPE_WAVEFORMS="${ROPE_WAVEFORMS:-sinusoid,triangular,square,sawtooth}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
DEFAULT_DATASET_NAME="${DEFAULT_DATASET_NAME:-}"
DEFAULT_DATASET_CONFIG_NAME="${DEFAULT_DATASET_CONFIG_NAME:-}"
DEFAULT_DATASET_DIR="${DEFAULT_DATASET_DIR:-}"

if [[ -z "${HF_HOME:-}" ]]; then
  if [[ -e "$HOME/fscratch" ]]; then
    HF_HOME="$(readlink -f "$HOME/fscratch")/qwen3/hf_cache"
  else
    HF_HOME="$REPO_ROOT/.hf_cache"
  fi
fi
export HF_HOME
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
mkdir -p "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$HF_HUB_CACHE"

slugify() {
  local value="$1"
  value="${value##*/}"
  value="${value%.*}"
  value="${value//[^A-Za-z0-9._-]/-}"
  value="${value##-}"
  value="${value%%-}"
  if [[ -z "$value" ]]; then
    value="unknown"
  fi
  printf '%s' "$value"
}

infer_dataset_slug() {
  local dataset=""
  local dataset_config=""
  local dataset_dir=""
  local train_file=""
  local validation_file=""
  local args=("$@")

  for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[$i]}" in
      --dataset_name | --dataset-name)
        dataset="${args[$((i + 1))]:-}"
        ;;
      --dataset_name=* | --dataset-name=*)
        dataset="${args[$i]#*=}"
        ;;
      --dataset_config_name | --dataset-config-name)
        dataset_config="${args[$((i + 1))]:-}"
        ;;
      --dataset_config_name=* | --dataset-config-name=*)
        dataset_config="${args[$i]#*=}"
        ;;
      --dataset_dir | --dataset-dir)
        dataset_dir="${args[$((i + 1))]:-}"
        ;;
      --dataset_dir=* | --dataset-dir=*)
        dataset_dir="${args[$i]#*=}"
        ;;
      --train_file | --train-file)
        train_file="${args[$((i + 1))]:-}"
        ;;
      --train_file=* | --train-file=*)
        train_file="${args[$i]#*=}"
        ;;
      --validation_file | --validation-file)
        validation_file="${args[$((i + 1))]:-}"
        ;;
      --validation_file=* | --validation-file=*)
        validation_file="${args[$i]#*=}"
        ;;
      --output_dir | --output-dir | --output_dir=* | --output-dir=*)
        echo "Do not pass --output_dir to this launcher; set OUTPUT_BASE_DIR instead." >&2
        exit 2
        ;;
    esac
  done

  if [[ -n "$dataset" && -n "$dataset_config" ]]; then
    slugify "${dataset}-${dataset_config}"
  elif [[ -n "$dataset" ]]; then
    slugify "$dataset"
  elif [[ -n "$dataset_dir" ]]; then
    slugify "$dataset_dir"
  elif [[ -n "$train_file" ]]; then
    slugify "$train_file"
  elif [[ -n "$validation_file" ]]; then
    slugify "$validation_file"
  else
    printf '%s' "unspecified-dataset"
  fi
}

DATASET_SLUG="$(infer_dataset_slug "$@")"
EXTRA_ARGS=("$@")
if [[ "$DATASET_SLUG" == "unspecified-dataset" ]]; then
  if [[ -n "$DEFAULT_DATASET_DIR" ]]; then
    if [[ ! -d "$DEFAULT_DATASET_DIR" ]]; then
      echo "DEFAULT_DATASET_DIR does not exist: $DEFAULT_DATASET_DIR" >&2
      exit 2
    fi
    export HF_DATASETS_OFFLINE=1
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    DATASET_SLUG="$(slugify "$DEFAULT_DATASET_DIR")"
    EXTRA_ARGS=(--dataset_dir "$DEFAULT_DATASET_DIR" "${EXTRA_ARGS[@]}")
  elif [[ -n "$DEFAULT_DATASET_NAME" ]]; then
    DATASET_SLUG="$(slugify "${DEFAULT_DATASET_NAME}-${DEFAULT_DATASET_CONFIG_NAME}")"
    EXTRA_ARGS=(--dataset_name "$DEFAULT_DATASET_NAME" "${EXTRA_ARGS[@]}")
    if [[ -n "$DEFAULT_DATASET_CONFIG_NAME" ]]; then
      EXTRA_ARGS=(--dataset_config_name "$DEFAULT_DATASET_CONFIG_NAME" "${EXTRA_ARGS[@]}")
    fi
  else
    DEFAULT_DATASET_DIR="$HF_HOME/../datasets/Salesforce_wikitext__wikitext-103-raw-v1"
    DEFAULT_DATASET_DIR="$(readlink -m "$DEFAULT_DATASET_DIR")"
    if [[ ! -d "$DEFAULT_DATASET_DIR" ]]; then
      echo "Exact local dataset is missing: $DEFAULT_DATASET_DIR" >&2
      echo "Run: $PYTHON $REPO_ROOT/scripts/prefetch_qwen3_dataset.py" >&2
      exit 2
    fi
    export HF_DATASETS_OFFLINE=1
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    DATASET_SLUG="$(slugify "$DEFAULT_DATASET_DIR")"
    EXTRA_ARGS=(--dataset_dir "$DEFAULT_DATASET_DIR" "${EXTRA_ARGS[@]}")
  fi
fi

IFS=',' read -ra WAVES <<< "$ROPE_WAVEFORMS"
for wave in "${WAVES[@]}"; do
  output_dir="$OUTPUT_BASE_DIR/qwen3_$wave/${TIMESTAMP}_${DATASET_SLUG}"
  echo "Training Qwen3 with rope_waveform=$wave -> $output_dir"
  QWEN3_OUTPUT_BASE_DIR="$OUTPUT_BASE_DIR" QWEN3_OUTPUT_TIMESTAMP="$TIMESTAMP" \
    "$PYTHON" "$REPO_ROOT/examples/pytorch/language-modeling/run_clm.py" \
    --model_name_or_path "$MODEL_PATH" \
    --rope_waveform "$wave" \
    "${EXTRA_ARGS[@]}"
done
