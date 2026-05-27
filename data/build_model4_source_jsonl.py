#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build aligned RGB/depth JSONL sources for stage-25 model4 feature export.",
    )
    parser.add_argument("--raw_jsonl", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--rgb_output", required=True)
    parser.add_argument("--depth_output", required=True)
    args = parser.parse_args()

    raw_jsonl = Path(args.raw_jsonl).resolve()
    data_root = Path(args.data_root).resolve()
    rgb_output = Path(args.rgb_output).resolve()
    depth_output = Path(args.depth_output).resolve()

    if not raw_jsonl.exists():
        raise FileNotFoundError(f"raw_jsonl not found: {raw_jsonl}")
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    rgb_items = []
    depth_items = []

    for item in load_jsonl(raw_jsonl):
        sample_id = str(item["id"])
        video_id = sample_id.split("/", 1)[0]

        rgb_rel = item.get("image")
        depth_rel = item.get("depth") or item.get("depth_path")
        if not rgb_rel or not depth_rel:
            raise KeyError(
                f"Sample {sample_id} must contain both image and depth/depth_path keys"
            )

        rgb_abs = (data_root / rgb_rel).resolve()
        depth_abs = (data_root / depth_rel).resolve()

        if not rgb_abs.exists():
            raise FileNotFoundError(f"RGB image missing for {sample_id}: {rgb_abs}")
        if not depth_abs.exists():
            raise FileNotFoundError(f"Depth image missing for {sample_id}: {depth_abs}")

        rgb_items.append(
            {
                "id": sample_id,
                "video_id": video_id,
                "image": str(rgb_abs),
            }
        )

        depth_items.append(
            {
                "id": sample_id,
                "video_id": video_id,
                "image": str(depth_abs),
                "depth1_path": str(depth_abs),
            }
        )

    write_jsonl(rgb_output, rgb_items)
    write_jsonl(depth_output, depth_items)

    print(f"Wrote {len(rgb_items)} RGB source records to {rgb_output}")
    print(f"Wrote {len(depth_items)} depth source records to {depth_output}")


if __name__ == "__main__":
    main()
