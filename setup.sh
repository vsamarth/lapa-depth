#!/usr/bin/env bash
# ============================================================
# LAPA Depth: Setup Script
# ============================================================
set -eu
cd "$(dirname "$0")"

HF_TOKEN="${HF_TOKEN:-}"
SUITE="${SUITE:-libero_spatial,libero_object,libero_goal}"
MODALITY="${MODALITY:-vision,depth,action}"
FULL_RUN_SUITE="libero_90"
HF_DOWNLOAD_WORKERS="${HF_DOWNLOAD_WORKERS:-16}"
DEPTH_BATCH_SIZE="${DEPTH_BATCH_SIZE:-128}"
DEPTH_WRITE_WORKERS="${DEPTH_WRITE_WORKERS:-8}"
MAX_HDF5_FILES="${MAX_HDF5_FILES:-0}"
MAX_DEMOS="${MAX_DEMOS:-0}"

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

LIBERO_DIR="${LIBERO_DIR:-$DATA_ROOT/libero}"
OUT_DIR="${OUT_DIR:-$DATA_ROOT/libero_finetune}"
export DATA_ROOT LIBERO_DIR OUT_DIR

{
    printf 'DATA_ROOT=%q\n' "$DATA_ROOT"
    printf 'LIBERO_DIR=%q\n' "$LIBERO_DIR"
    printf 'OUT_DIR=%q\n' "$OUT_DIR"
} > .lapa_data_root

echo "=== LAPA Setup ==="
echo "Suites: $SUITE | Modality: $MODALITY"
echo "DATA_ROOT: $DATA_ROOT"
if [ "$SUITE" != "$FULL_RUN_SUITE" ]; then
    echo "REMINDER: switch to SUITE=$FULL_RUN_SUITE for the full run"
fi
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true
echo ""

# ---- venv + deps ----
echo ">>> Installing dependencies..."
uv sync
uv pip install "jax[cuda12]==0.4.23" nvidia-cudnn-cu12==8.9.7.29 \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
source .venv/bin/activate

# ---- checkpoints ----
echo ">>> Downloading checkpoints..."
mkdir -p checkpoints
download_checkpoints() {
    for f in params tokenizer.model vqgan; do
        [ -f "checkpoints/$f" ] || wget -c -q --show-progress "https://huggingface.co/latent-action-pretraining/LAPA-7B-openx/resolve/main/$f" -O "checkpoints/$f"
    done
}

# ---- LIBERO dataset ----
download_libero() {
    echo ">>> Downloading LIBERO..."
    python3 - "$SUITE" "$HF_DOWNLOAD_WORKERS" "$LIBERO_DIR" <<'PYEOF'
import os
import sys
from huggingface_hub import snapshot_download

suites = [s.strip() for s in sys.argv[1].split(",") if s.strip()]
max_workers = int(sys.argv[2])
libero_dir = sys.argv[3]
os.makedirs(libero_dir, exist_ok=True)

for suite in suites:
    snapshot_download(
        repo_id="yifengzhu-hf/LIBERO-datasets",
        repo_type="dataset",
        allow_patterns=f"{suite}/**/*.hdf5",
        local_dir=libero_dir,
        local_dir_use_symlinks=False,
        max_workers=max_workers,
    )

print(f"OK: {len(suites)} suite(s) downloaded with up to {max_workers} workers")
PYEOF
}

download_checkpoints &
ckpt_pid=$!
download_libero &
libero_pid=$!
wait "$ckpt_pid"
wait "$libero_pid"

echo ">>> Preprocessing LIBERO data..."
DEPTH_BATCH_SIZE="$DEPTH_BATCH_SIZE" \
DEPTH_WRITE_WORKERS="$DEPTH_WRITE_WORKERS" \
MAX_HDF5_FILES="$MAX_HDF5_FILES" \
MAX_DEMOS="$MAX_DEMOS" \
DATA_ROOT="$DATA_ROOT" \
LIBERO_DIR="$LIBERO_DIR" \
OUT_DIR="$OUT_DIR" \
bash ./preprocess_libero_data.sh

echo ""
echo "=== SETUP COMPLETE ==="
echo "RGB frames:     $OUT_DIR/frames/"
echo "Depth frames:   $OUT_DIR/depth_frames/"
echo "Processed data: $OUT_DIR/processed.jsonl"
echo "Action bins:    $OUT_DIR/action_scale.csv"
echo "Checkpoints:    $(pwd)/checkpoints/"
echo ""
echo "Full run:       SUITE=$FULL_RUN_SUITE ./setup.sh"
echo "Small test:     MAX_HDF5_FILES=1 MAX_DEMOS=5 ./setup.sh"
echo "Next: run './train.sh' to start fine-tuning"
