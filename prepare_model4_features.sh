#!/usr/bin/env bash
# ============================================================
# Prepare stage-25 model4 features from setup.sh outputs
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
RAW_JSONL="${RAW_JSONL:-$OUT_DIR/libero_finetune.jsonl}"
FEATURE_ROOT="${FEATURE_ROOT:-$DATA_ROOT/model4_features}"
RGB_SOURCE_JSONL="${RGB_SOURCE_JSONL:-$FEATURE_ROOT/z_rgb_train_mixed.jsonl}"
DEPTH_SOURCE_JSONL="${DEPTH_SOURCE_JSONL:-$FEATURE_ROOT/z_depth_train_mixed.jsonl}"
RGB_UNSHUFFLED_JSONL="${RGB_UNSHUFFLED_JSONL:-$FEATURE_ROOT/z_rgb_unshuffled.jsonl}"
DEPTH_UNSHUFFLED_JSONL="${DEPTH_UNSHUFFLED_JSONL:-$FEATURE_ROOT/z_depth_unshuffled.jsonl}"
RGB_FEATURE_DIR="${RGB_FEATURE_DIR:-$FEATURE_ROOT/z_rgb_features}"
DEPTH_FEATURE_DIR="${DEPTH_FEATURE_DIR:-$FEATURE_ROOT/z_depth_features}"
RGB_FEATURE_PREFIX="${RGB_FEATURE_PREFIX:-z_rgb_train_mixed}"
DEPTH_FEATURE_PREFIX="${DEPTH_FEATURE_PREFIX:-z_depth_train_mixed}"
RGB_FEATURE_MANIFEST="${RGB_FEATURE_MANIFEST:-$RGB_FEATURE_DIR/${RGB_FEATURE_PREFIX}_manifest.json}"
DEPTH_FEATURE_MANIFEST="${DEPTH_FEATURE_MANIFEST:-$DEPTH_FEATURE_DIR/${DEPTH_FEATURE_PREFIX}_manifest.json}"

RGB_LAQ_CHECKPOINT="${RGB_LAQ_CHECKPOINT:-}"
DEPTH_LAQ_CHECKPOINT="${DEPTH_LAQ_CHECKPOINT:-}"
CODEBOOK_SIZE="${CODEBOOK_SIZE:-8}"
WINDOW_SIZE="${WINDOW_SIZE:-30}"
CODE_SEQ_LEN="${CODE_SEQ_LEN:-4}"
MODEL_LAYER="${MODEL_LAYER:-8}"
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-64}"
FEATURE_NUM_WORKERS="${FEATURE_NUM_WORKERS:-4}"
FEATURE_PART_SIZE="${FEATURE_PART_SIZE:-8192}"
REPEAT_DEPTH_TO_3CH="${REPEAT_DEPTH_TO_3CH:-1}"

echo "=== Prepare Model4 Features ==="
echo "DATA_ROOT:             $DATA_ROOT"
echo "RAW_JSONL:             $RAW_JSONL"
echo "FEATURE_ROOT:          $FEATURE_ROOT"
echo "RGB_LAQ_CHECKPOINT:    ${RGB_LAQ_CHECKPOINT:-<missing>}"
echo "DEPTH_LAQ_CHECKPOINT:  ${DEPTH_LAQ_CHECKPOINT:-<missing>}"
echo ""

if [ ! -f "$RAW_JSONL" ]; then
    echo "ERROR: setup output not found."
    echo "Expected raw JSONL at: $RAW_JSONL"
    echo "Run 'bash setup.sh' first."
    exit 1
fi

if [ -z "$RGB_LAQ_CHECKPOINT" ]; then
    echo "ERROR: set RGB_LAQ_CHECKPOINT to the pretrained RGB LAQ checkpoint."
    exit 1
fi

if [ -z "$DEPTH_LAQ_CHECKPOINT" ]; then
    echo "ERROR: set DEPTH_LAQ_CHECKPOINT to the pretrained depth LAQ checkpoint."
    exit 1
fi

mkdir -p "$FEATURE_ROOT" "$RGB_FEATURE_DIR" "$DEPTH_FEATURE_DIR"

echo ">>> Building aligned RGB/depth source JSONLs..."
python3 data/build_model4_source_jsonl.py \
    --raw_jsonl "$RAW_JSONL" \
    --data_root "$OUT_DIR" \
    --rgb_output "$RGB_SOURCE_JSONL" \
    --depth_output "$DEPTH_SOURCE_JSONL"

echo ">>> Exporting RGB features..."
(
    cd laq
    python3 export_model4_rgb_features.py \
        --input_file "$RGB_SOURCE_JSONL" \
        --dist_number 1 \
        --divider 1 \
        --codebook_size "$CODEBOOK_SIZE" \
        --laq_checkpoint "$RGB_LAQ_CHECKPOINT" \
        --window_size "$WINDOW_SIZE" \
        --code_seq_len "$CODE_SEQ_LEN" \
        --layer "$MODEL_LAYER" \
        --unshuffled_jsonl "$RGB_UNSHUFFLED_JSONL" \
        --feature_dir "$RGB_FEATURE_DIR" \
        --feature_prefix "$RGB_FEATURE_PREFIX" \
        --feature_part_size "$FEATURE_PART_SIZE" \
        --batch_size "$FEATURE_BATCH_SIZE" \
        --num_workers "$FEATURE_NUM_WORKERS"
)

echo ">>> Exporting depth features..."
(
    cd laq
    python3 export_model4_depth_features.py \
        --input_file "$DEPTH_SOURCE_JSONL" \
        --dist_number 1 \
        --divider 1 \
        --codebook_size "$CODEBOOK_SIZE" \
        --laq_checkpoint "$DEPTH_LAQ_CHECKPOINT" \
        --window_size "$WINDOW_SIZE" \
        --code_seq_len "$CODE_SEQ_LEN" \
        --layer "$MODEL_LAYER" \
        --unshuffled_jsonl "$DEPTH_UNSHUFFLED_JSONL" \
        --feature_dir "$DEPTH_FEATURE_DIR" \
        --feature_prefix "$DEPTH_FEATURE_PREFIX" \
        --feature_part_size "$FEATURE_PART_SIZE" \
        --batch_size "$FEATURE_BATCH_SIZE" \
        --num_workers "$FEATURE_NUM_WORKERS" \
        --repeat_depth_to_3ch "$REPEAT_DEPTH_TO_3CH"
)

echo ">>> Verifying feature alignment..."
python3 - "$DEPTH_SOURCE_JSONL" "$RGB_FEATURE_MANIFEST" "$DEPTH_FEATURE_MANIFEST" <<'PYEOF'
import json
import sys
from pathlib import Path

depth_jsonl = Path(sys.argv[1])
rgb_manifest = Path(sys.argv[2])
depth_manifest = Path(sys.argv[3])

with depth_jsonl.open("r", encoding="utf-8") as handle:
    depth_count = sum(1 for line in handle if line.strip())

with rgb_manifest.open("r", encoding="utf-8") as handle:
    rgb_total = int(json.load(handle)["total_samples"])

with depth_manifest.open("r", encoding="utf-8") as handle:
    depth_total = int(json.load(handle)["total_samples"])

if not (depth_count == rgb_total == depth_total):
    raise RuntimeError(
        f"Alignment mismatch: depth_jsonl={depth_count}, "
        f"rgb_manifest={rgb_total}, depth_manifest={depth_total}"
    )

print(
    f"OK: aligned samples = {depth_count} "
    f"(depth_jsonl={depth_count}, rgb_manifest={rgb_total}, depth_manifest={depth_total})"
)
PYEOF

echo ""
echo "=== MODEL4 FEATURES READY ==="
echo "z_depth_path:            $DEPTH_SOURCE_JSONL"
echo "z_rgb_feature_manifest:  $RGB_FEATURE_MANIFEST"
echo "z_depth_feature_manifest:$DEPTH_FEATURE_MANIFEST"
