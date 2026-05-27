#!/usr/bin/env python3
"""
Optimized depth generation using Depth Anything V2.
- FP16 inference with torch.cuda.amp
- Large batch size (64-128)
- Parallel PNG I/O
- No unnecessary disk scans
"""
import os, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from tqdm import tqdm

FRAMES_DIR = "/data/libero_finetune/frames"
DEPTH_DIR = "/data/libero_finetune/depth_frames"
MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"
BATCH_SIZE = int(os.environ.get("DEPTH_BATCH_SIZE", "128"))
WRITE_WORKERS = int(os.environ.get("DEPTH_WRITE_WORKERS", "8"))


def save_depth(args):
    """Save single depth PNG (called from thread pool)"""
    path, depth_np = args
    d_min, d_max = depth_np.min(), depth_np.max()
    if d_max - d_min > 1e-8:
        depth_norm = ((depth_np - d_min) / (d_max - d_min) * 65535.0).astype(np.uint16)
    else:
        depth_norm = np.zeros(depth_np.shape, dtype=np.uint16)
    cv2.imwrite(str(path), depth_norm)


def main():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    print(f"Device: {device}")
    print(f"Batch size: {BATCH_SIZE}, Write workers: {WRITE_WORKERS}")

    # Load model
    print(f"Loading {MODEL_NAME}...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_NAME).to(device)
    model = model.half()  # FP16 for speed
    model.eval()
    print("Model loaded (FP16)")

    # Find episodes
    ep_dirs = sorted(
        Path(FRAMES_DIR).glob("ep_*"),
        key=lambda p: int(p.name.split("_")[1]),
    )
    if not ep_dirs:
        print(f"No episode dirs found in {FRAMES_DIR}")
        return

    total_frames = sum(len(sorted(d.glob("*.jpg"))) for d in ep_dirs)
    print(f"Episodes: {len(ep_dirs)} | Total frames: {total_frames}")

    write_pool = ThreadPoolExecutor(max_workers=WRITE_WORKERS)

    with torch.inference_mode():
        for ep_dir in tqdm(ep_dirs, desc="Episodes"):
            ep_name = ep_dir.name
            depth_ep_dir = Path(DEPTH_DIR) / ep_name
            depth_ep_dir.mkdir(parents=True, exist_ok=True)

            rgb_files = sorted(ep_dir.glob("*.jpg"))

            for i in range(0, len(rgb_files), BATCH_SIZE):
                batch_files = rgb_files[i : i + BATCH_SIZE]

                # Filter: skip files that already have depth
                todo = []
                for rf in batch_files:
                    df = depth_ep_dir / f"{rf.stem}.png"
                    if not df.exists():
                        todo.append((rf, df))

                if not todo:
                    continue

                # Load images
                images = []
                for rf, _ in todo:
                    with Image.open(rf) as img:
                        images.append(img.convert("RGB"))

                # FP16 inference
                with torch.cuda.amp.autocast():
                    inputs = processor(images=images, return_tensors="pt").to(device)
                    outputs = model(**inputs)
                    depth_maps = outputs.predicted_depth  # (B, H, W)

                # Submit saves to thread pool
                for (_, df_path), depth_tensor in zip(todo, depth_maps):
                    depth_np = depth_tensor.float().cpu().numpy()
                    write_pool.submit(save_depth, (df_path, depth_np))

    # Wait for all writes to finish
    write_pool.shutdown(wait=True)

    generated = sum(1 for _ in Path(DEPTH_DIR).rglob("*.png"))
    print(f"\nDepth frames: {generated} | Missing: {total_frames - generated}")
    print("Done!")


if __name__ == "__main__":
    main()
