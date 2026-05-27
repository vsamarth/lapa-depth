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

echo "=== LAPA Setup ==="
echo "Suites: $SUITE | Modality: $MODALITY"
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
    python3 - "$SUITE" "$HF_DOWNLOAD_WORKERS" <<'PYEOF'
import os
import sys
from huggingface_hub import snapshot_download

suites = [s.strip() for s in sys.argv[1].split(",") if s.strip()]
max_workers = int(sys.argv[2])
os.makedirs("/datasets/libero", exist_ok=True)

for suite in suites:
    snapshot_download(
        repo_id="yifengzhu-hf/LIBERO-datasets",
        repo_type="dataset",
        allow_patterns=f"{suite}/**/*.hdf5",
        local_dir="/datasets/libero",
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
DEPTH_BATCH_SIZE="$DEPTH_BATCH_SIZE" DEPTH_WRITE_WORKERS="$DEPTH_WRITE_WORKERS" bash ./preprocess_libero_data.sh

echo ""
echo "=== SETUP COMPLETE ==="
echo "RGB frames:     /data/libero_finetune/frames/"
echo "Depth frames:   /data/libero_finetune/depth_frames/"
echo "Processed data: /data/libero_finetune/processed.jsonl"
echo "Action bins:    /data/libero_finetune/action_scale.csv"
echo "Checkpoints:    $(pwd)/checkpoints/"
echo ""
echo "Full run:       SUITE=$FULL_RUN_SUITE ./setup.sh"
echo "Next: run './train.sh' to start fine-tuning"
