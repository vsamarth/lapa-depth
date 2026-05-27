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
for f in params tokenizer.model; do
    [ -f "checkpoints/$f" ] || wget -c -q --show-progress "https://huggingface.co/latent-action-pretraining/LAPA-7B-openx/resolve/main/$f" -O "checkpoints/$f"
done
if [ ! -f "checkpoints/vqgan" ]; then
    wget -c -q --show-progress "https://huggingface.co/latent-action-pretraining/LAPA-7B-openx/resolve/main/vqgan" -O "checkpoints/vqgan"
fi

# ---- LIBERO dataset ----
echo ">>> Downloading LIBERO..."
python3 - "$SUITE" "$HF_TOKEN" <<'PYEOF'
import sys, os
from huggingface_hub import list_repo_files, hf_hub_download
suites = sys.argv[1].split(",")
token = sys.argv[2]
os.makedirs("/datasets/libero", exist_ok=True)
total = 0
for suite in suites:
    suite = suite.strip()
    files = sorted([f for f in list_repo_files("yifengzhu-hf/LIBERO-datasets", repo_type="dataset") if f.startswith(f"{suite}/") and f.endswith(".hdf5")])
    for f in files:
        dest = os.path.join("/datasets/libero", os.path.basename(f))
        if os.path.exists(dest): continue
        hf_hub_download("yifengzhu-hf/LIBERO-datasets", f, repo_type="dataset", local_dir="/datasets/libero", local_dir_use_symlinks=False)
        total += 1
print(f"OK: {total} files from {len(suites)} suite(s)")
PYEOF

# ---- convert HDF5 to RGB frames ----
echo ">>> Converting HDF5 to frames..."
python3 <<'PYEOF'
import sys, os, json, h5py, tensorflow as tf, numpy as np
from pathlib import Path
from tqdm import tqdm

hdf5_files = sorted(Path("/datasets/libero").rglob("*.hdf5"))
os.makedirs("/data/libero_finetune/frames", exist_ok=True)
os.makedirs("/data/libero_finetune/depth_frames", exist_ok=True)

data = []; ep_id = 0
for hf in tqdm(hdf5_files, desc="Converting"):
    name = hf.stem
    task = name.split("_", 1)[1].replace("_demo", "").replace("_", " ")
    with h5py.File(hf, "r") as f:
        demos = sorted([k for k in f["data"].keys() if k.startswith("demo_")], key=lambda k: int(k.split("_")[1]))
        for dk in demos:
            demo = f["data"][dk]
            actions = demo["actions"][:]
            images = demo["obs/agentview_rgb"][:]
            edir = Path(f"/data/libero_finetune/frames/ep_{ep_id}")
            edir.mkdir(parents=True, exist_ok=True)
            ddir = Path(f"/data/libero_finetune/depth_frames/ep_{ep_id}")
            ddir.mkdir(parents=True, exist_ok=True)
            for i in range(len(images)):
                img_path = edir / f"{i:04d}.jpg"
                if not img_path.exists():
                    tf.io.write_file(str(img_path), tf.image.encode_jpeg(images[i], quality=95))
                data.append({"id": f"ep_{ep_id}/step_{i}", "image": str(img_path),
                    "conversations": [{"from": "human", "value": f"<image>\nWhat action should the robot take to `{task}`"},
                                      {"from": "gpt", "raw_actions": [float(v) for v in actions[i]]}]})
            ep_id += 1

with open("/data/libero_finetune/libero_finetune.json", "w") as f:
    json.dump(data, f)
print(f"OK: {len(data)} steps from {ep_id} demos")
PYEOF

# ---- depth gen (Depth Anything V2) ----
echo ">>> Generating depth frames..."
python3 gen_depth.py

# ---- preprocess ----
echo ">>> Preprocessing actions..."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
FIELDS="[instruction],[vision],[depth],action"
if [ "$MODALITY" != "vision,depth,action" ]; then
    FIELDS="[instruction],[vision],action"
fi
python3 data/finetune_preprocess.py \
    --input_path /data/libero_finetune/libero_finetune.json \
    --output_filename /data/libero_finetune/processed.jsonl \
    --csv_filename /data/libero_finetune/action_scale.csv \
    --fields "$FIELDS"

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
