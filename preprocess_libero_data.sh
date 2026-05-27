#!/usr/bin/env bash
# ============================================================
# LAPA Depth: LIBERO preprocessing
# Assumes the LIBERO hdf5 files have already been downloaded.
# ============================================================
set -eu
cd "$(dirname "$0")"

if [ -f .venv/bin/activate ] && [ -z "${VIRTUAL_ENV:-}" ]; then
    source .venv/bin/activate
fi

LIBERO_DIR="${LIBERO_DIR:-/datasets/libero}"
OUT_DIR="${OUT_DIR:-/data/libero_finetune}"
FRAMES_DIR="${FRAMES_DIR:-$OUT_DIR/frames}"
DEPTH_DIR="${DEPTH_DIR:-$OUT_DIR/depth_frames}"
RAW_JSONL="${RAW_JSONL:-$OUT_DIR/libero_finetune.jsonl}"
PROCESSED_JSONL="${PROCESSED_JSONL:-$OUT_DIR/processed.jsonl}"
ACTION_SCALE_CSV="${ACTION_SCALE_CSV:-$OUT_DIR/action_scale.csv}"
MODALITY="${MODALITY:-vision,depth,action}"
DISCRETIZE_BINS="${DISCRETIZE_BINS:-256}"

echo "=== LIBERO Preprocessing ==="
echo "LIBERO_DIR:        $LIBERO_DIR"
echo "OUT_DIR:           $OUT_DIR"
echo "MODALITY:          $MODALITY"
echo ""

if [ ! -d "$LIBERO_DIR" ]; then
    echo "ERROR: LIBERO directory not found: $LIBERO_DIR"
    echo "Run setup.sh first or point LIBERO_DIR at the downloaded hdf5 files."
    exit 1
fi

mkdir -p "$FRAMES_DIR" "$DEPTH_DIR"

echo ">>> Converting HDF5 to RGB frames..."
python3 - "$LIBERO_DIR" "$FRAMES_DIR" "$DEPTH_DIR" "$RAW_JSONL" <<'PYEOF'
import json
import sys
from pathlib import Path

import h5py
import tensorflow as tf
from tqdm import tqdm

libero_dir = Path(sys.argv[1])
frames_dir = Path(sys.argv[2])
depth_dir = Path(sys.argv[3])
raw_jsonl = Path(sys.argv[4])

hdf5_files = sorted(libero_dir.rglob("*.hdf5"))
if not hdf5_files:
    print(f"ERROR: no .hdf5 files found under {libero_dir}")
    sys.exit(1)

ep_id = 0
total_steps = 0

raw_jsonl.parent.mkdir(parents=True, exist_ok=True)
raw_fp = raw_jsonl.open("w", encoding="utf-8")

try:
    for hf in tqdm(hdf5_files, desc="HDF5 files"):
        name = hf.stem
        task = name.split("_", 1)[1].replace("_demo", "").replace("_", " ")
        with h5py.File(hf, "r") as f:
            demos = sorted(
                [k for k in f["data"].keys() if k.startswith("demo_")],
                key=lambda k: int(k.split("_")[1]),
            )
            for dk in demos:
                demo = f["data"][dk]
                actions = demo["actions"][:]
                images = demo["obs/agentview_rgb"][:]

                ep_dir = frames_dir / f"ep_{ep_id}"
                dep_dir = depth_dir / f"ep_{ep_id}"
                ep_dir.mkdir(parents=True, exist_ok=True)
                dep_dir.mkdir(parents=True, exist_ok=True)

                for i in range(len(images)):
                    img_path = ep_dir / f"{i:04d}.jpg"
                    if not img_path.exists():
                        tf.io.write_file(
                            str(img_path),
                            tf.image.encode_jpeg(images[i], quality=95),
                        )

                    raw_fp.write(
                        json.dumps(
                            {
                                "id": f"ep_{ep_id}/step_{i}",
                                "image": str(img_path),
                                "depth": str(dep_dir / f"{i:04d}.png"),
                                "conversations": [
                                    {
                                        "from": "human",
                                        "value": f"<image>\nWhat action should the robot take to `{task}`",
                                    },
                                    {
                                        "from": "gpt",
                                        "raw_actions": [float(v) for v in actions[i]],
                                    },
                                ],
                            }
                        )
                        + "\n"
                    )

                total_steps += len(images)
                ep_id += 1
finally:
    raw_fp.close()

print(f"OK: {total_steps} steps from {ep_id} demos")
print(f"Raw JSONL: {raw_jsonl}")
PYEOF

echo ">>> Generating depth frames..."
python3 gen_depth.py

echo ">>> Preprocessing actions..."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

if [ "$MODALITY" = "vision,depth,action" ]; then
    FIELDS="[instruction],[vision],[depth],action"
else
    FIELDS="[instruction],[vision],action"
fi

python3 data/finetune_preprocess.py \
    --input_path "$RAW_JSONL" \
    --output_filename "$PROCESSED_JSONL" \
    --csv_filename "$ACTION_SCALE_CSV" \
    --discretize_bins "$DISCRETIZE_BINS" \
    --fields "$FIELDS"

echo ""
echo "=== PREPROCESS COMPLETE ==="
echo "RGB frames:     $FRAMES_DIR"
echo "Depth frames:   $DEPTH_DIR"
echo "Raw JSONL:      $RAW_JSONL"
echo "Processed data: $PROCESSED_JSONL"
echo "Action bins:    $ACTION_SCALE_CSV"
