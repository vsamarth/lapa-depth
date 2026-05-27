#!/usr/bin/env bash
# ============================================================
# Stage-25 Model4 fine-tuning
# Run setup.sh first. This script can auto-prepare feature manifests.
# ============================================================
set -eu
cd "$(dirname "$0")"

if [ -f .venv/bin/activate ] && [ -z "${VIRTUAL_ENV:-}" ]; then
    source .venv/bin/activate
fi

if [ -f .lapa_data_root ]; then
    # shellcheck disable=SC1091
    source .lapa_data_root
fi

NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-65001}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-1e-4}"
SAVE_MODEL_EVERY="${SAVE_MODEL_EVERY:-5000}"
LOG_EVERY="${LOG_EVERY:-100}"
NUM_WORKERS="${NUM_WORKERS:-12}"
PIN_MEMORY="${PIN_MEMORY:-1}"

if [ -n "${DATA_ROOT:-}" ]; then
    mkdir -p "$DATA_ROOT"
    DATA_ROOT="$(python3 -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$DATA_ROOT")"
else
    for candidate in "/data" "$HOME/lapa-data" "$PWD/.lapa-data"; do
        if mkdir -p "$candidate" 2>/dev/null && [ -w "$candidate" ]; then
            DATA_ROOT="$(python3 -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$candidate")"
            break
        fi
    done
    if [ -z "${DATA_ROOT:-}" ]; then
        mkdir -p "$PWD/.lapa-data"
        DATA_ROOT="$(python3 -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$PWD/.lapa-data")"
    fi
fi

OUT_DIR="${OUT_DIR:-$DATA_ROOT/libero_finetune}"
FEATURE_ROOT="${FEATURE_ROOT:-$DATA_ROOT/model4_features}"
Z_DEPTH_PATH="${Z_DEPTH_PATH:-$FEATURE_ROOT/z_depth_train_mixed.jsonl}"
Z_RGB_FEATURE_MANIFEST="${Z_RGB_FEATURE_MANIFEST:-$FEATURE_ROOT/z_rgb_features/z_rgb_train_mixed_manifest.json}"
Z_DEPTH_FEATURE_MANIFEST="${Z_DEPTH_FEATURE_MANIFEST:-$FEATURE_ROOT/z_depth_features/z_depth_train_mixed_manifest.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PWD/outputs}"
RESULTS_FOLDER="${RESULTS_FOLDER:-$OUTPUT_ROOT/model4_finetune}"
PREPARE_MODEL4_FEATURES_AUTO="${PREPARE_MODEL4_FEATURES_AUTO:-1}"

mkdir -p "$OUTPUT_ROOT" "$RESULTS_FOLDER"

if [ -z "${PRETRAINED_CHECKPOINT:-}" ]; then
    echo "ERROR: set PRETRAINED_CHECKPOINT to the pretrained stage-25 model4 checkpoint."
    exit 1
fi

echo "=== Stage-25 Model4 Fine-tuning ==="
echo "DATA_ROOT:               $DATA_ROOT"
echo "Feature root:            $FEATURE_ROOT"
echo "Results folder:          $RESULTS_FOLDER"
echo "Pretrained checkpoint:   $PRETRAINED_CHECKPOINT"
echo ""

missing_inputs=0
for path in "$Z_DEPTH_PATH" "$Z_RGB_FEATURE_MANIFEST" "$Z_DEPTH_FEATURE_MANIFEST"; do
    if [ ! -f "$path" ]; then
        missing_inputs=1
    fi
done

if [ "$missing_inputs" -eq 1 ]; then
    if [ "$PREPARE_MODEL4_FEATURES_AUTO" = "1" ]; then
        echo ">>> Model4 feature inputs missing; preparing them now..."
        bash ./prepare_model4_features.sh
    else
        echo "ERROR: model4 feature inputs missing."
        echo "Expected:"
        echo "  $Z_DEPTH_PATH"
        echo "  $Z_RGB_FEATURE_MANIFEST"
        echo "  $Z_DEPTH_FEATURE_MANIFEST"
        echo "Run 'bash prepare_model4_features.sh' first."
        exit 1
    fi
fi

for path in "$Z_DEPTH_PATH" "$Z_RGB_FEATURE_MANIFEST" "$Z_DEPTH_FEATURE_MANIFEST"; do
    if [ ! -f "$path" ]; then
        echo "ERROR: required input still missing after preparation: $path"
        exit 1
    fi
done

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true
echo ""

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export PRETRAINED_CHECKPOINT
export RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
export Z_DEPTH_PATH
export Z_RGB_FEATURE_MANIFEST
export Z_DEPTH_FEATURE_MANIFEST
export RESULTS_FOLDER
export NUM_TRAIN_STEPS
export BATCH_SIZE
export LR
export SAVE_MODEL_EVERY
export LOG_EVERY
export NUM_WORKERS

if [ "$PIN_MEMORY" = "1" ]; then
    export PIN_MEMORY=1
fi

echo ">>> Launching model4 fine-tune..."
bash ./laq/train_stage25_sthv2_feature_model4_corl_da.sh

echo ""
echo "=== TRAINING COMPLETE ==="
echo "Results: $RESULTS_FOLDER"
