#!/bin/bash
set -e

# ROCm ONNX Runtime library path
ORT_ROCM_DIR=/usr/local/lib/python3.10/dist-packages/onnxruntime/capi
export LD_LIBRARY_PATH="${ORT_ROCM_DIR}:${LD_LIBRARY_PATH}"

echo "=== Model Accuracy and Speedup Evaluation Mode ==="

PROJ_ROOT="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="$PROJ_ROOT/../model"
PERF_SCRIPT="$PROJ_ROOT/benchmark.py"
BATCHSIZE=256
WARMUP=5
RUNNUM=2
MODELS=("batch_normalization" "conv" "group_normalization" "leakyrelu" "attention")

SNR_THRESH=1e-7
COSINE_THRESH=1e-7

BASELINE_FILE="$PROJ_ROOT/baseline_latency.json"
declare -A baseline_latency
declare -A current_latency

# Step 1: Load or generate baseline latency
echo -e "\n>>> Step 1: Load baseline latency"

if [[ -f "$BASELINE_FILE" ]]; then
  echo " Loading baseline latency from file: $BASELINE_FILE"
  for model in "${MODELS[@]}"; do
    latency=$(grep "\"$model\"" "$BASELINE_FILE" | sed -E 's/.*: ([0-9.]+),?/\1/')
    if [[ "$latency" == "null" || -z "$latency" ]]; then
      echo " $model baseline latency not recorded, will be re-measured"
      unset baseline_latency["$model"]
    else
      baseline_latency["$model"]=$latency
      echo "$model baseline latency: ${latency} ms (loaded)"
    fi
  done
else
  echo " baseline_latency.json not found, will perform initial latency measurement..."
fi

baseline_updated=false

for model in "${MODELS[@]}"; do
  if [[ -z "${baseline_latency[$model]}" ]]; then
    MODEL_PATH="$MODEL_DIR/${model}.onnx"
    DATA_PATH="$MODEL_DIR/$model"
    mkdir -p "$DATA_PATH"

    echo "[Baseline] Measuring $model"
    timeout 45s python3 "$PERF_SCRIPT" -i "$MODEL_PATH" -d "$DATA_PATH" -b $BATCHSIZE -w $WARMUP -n $RUNNUM -t 1 > "$DATA_PATH/baseline_log.txt" 2>&1 || continue

    baseline=$(grep "Inference cost per sample" "$DATA_PATH/baseline_log.txt" | sed -n 's/.*sample: \([0-9.]*\) ms.*/\1/p')
    baseline_latency["$model"]=$baseline
    echo "$model baseline latency: $baseline ms"
    baseline_updated=true
  fi
done

# Write updated baseline latency to JSON file
if $baseline_updated; then
  echo " Writing to baseline_latency.json"
  {
    echo "{"
    for i in "${!baseline_latency[@]}"; do
      printf '  "%s": %.3f,\n' "$i" "${baseline_latency[$i]}"
    done | sed '$s/,$//'
    echo "}"
  } > "$BASELINE_FILE"
fi

# Step 2: Evaluate optimized versions for accuracy and performance
echo -e "\n>>> Step 2: Evaluate optimized models"

for model in "${MODELS[@]}"; do
  MODEL_PATH="$MODEL_DIR/${model}.onnx"
  DATA_PATH="$MODEL_DIR/$model"
  LOG_FILE="$DATA_PATH/log.txt"

  echo "[Evaluate] $model"
  timeout 60s python3 "$PERF_SCRIPT" -i "$MODEL_PATH" -d "$DATA_PATH" -b $BATCHSIZE -c True -w $WARMUP -n $RUNNUM -t 0 > "$LOG_FILE" 2>&1 || {
    echo " $model inference failed or timed out"
    continue
  }

  snr=$(grep "SNR IS" "$LOG_FILE" | awk '{print $4}')
  cosine=$(grep "COSINE IS" "$LOG_FILE" | awk '{print $4}')
  curr_latency=$(grep "Inference cost per sample" "$LOG_FILE" | sed -n 's/.*sample: \([0-9.]*\) ms.*/\1/p')

  [[ -z "$curr_latency" ]] && echo "⚠️ $model latency not found, skipping" && continue

  current_latency["$model"]=$curr_latency
  base=${baseline_latency[$model]}
  speedup=$(python3 -c "print($base / $curr_latency)")

  echo " $model inference results:"
  echo "  - Baseline latency    : $base ms"
  echo "  - Current latency     : $curr_latency ms"
  echo "  - Speedup             : ${speedup}x"
  echo "  - Accuracy SNR        : $snr"
  echo "  - Accuracy Cosine     : $cosine"

  if [[ $(python3 -c "print(1 if float($snr) <= $SNR_THRESH and float($cosine) <= $COSINE_THRESH else 0)") -eq 1 ]]; then
    echo " Accuracy check passed"
  else
    echo " Accuracy check failed"
  fi
  echo
done
echo -e "\n Cleaning up ONNX Runtime profiling files..."
find "$PROJ_ROOT" -name "*profile*.json" -delete
