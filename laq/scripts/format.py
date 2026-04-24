from pathlib import Path
import json
import re

# ===== CONFIG =====
root_dir = Path("/datasets/something-something-v2/nips/depth_train")
output_jsonl = Path("/datasets/something-something-v2/nips/depth_train.jsonl")
label_path = Path("/datasets/something-something-v2/nips/labels/train.json")

valid_exts = {".png", ".jpg", ".jpeg"}


def natural_key(path: Path):
    m = re.search(r"(\d+)", path.stem)
    if m:
        return int(m.group(1))
    return path.stem


def folder_key(path: Path):
    return int(path.name) if path.name.isdigit() else path.name


# ===== LOAD LABELS =====
with label_path.open("r", encoding="utf-8") as f:
    label_data = json.load(f)

# map: video_id (string) -> instruction text
id_to_instruction = {}
for item in label_data:
    video_id = str(item["id"])
    instruction = item.get("label", "")
    id_to_instruction[video_id] = instruction

print(f"Loaded {len(id_to_instruction)} labels from {label_path}")


# ===== GENERATE JSONL =====
num_written = 0
num_missing_label = 0

with output_jsonl.open("w", encoding="utf-8") as f:
    for video_dir in sorted([p for p in root_dir.iterdir() if p.is_dir()], key=folder_key):
        video_id = str(video_dir.name)
        instruction = id_to_instruction.get(video_id, "")

        if instruction == "":
            num_missing_label += 1
            print(f"Warning: missing label for video_id={video_id}")

        frame_files = sorted(
            [p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts],
            key=natural_key
        )

        for frame_path in frame_files:
            m = re.search(r"(\d+)", frame_path.stem)
            if m is None:
                print(f"Skip file with no frame index: {frame_path}")
                continue

            step = int(m.group(1))

            item = {
                "id": f"{video_id}_{step}",              # sample id used later by your inference code
                "video_id": video_id,                    # added
                "image": str(frame_path.resolve()),
                "instruction": instruction,              # loaded from train.json
                "vision": []
            }

            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            num_written += 1

print(f"Done. Wrote {num_written} lines to: {output_jsonl}")
print(f"Videos missing label: {num_missing_label}")