import argparse
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

from laq_model.latent_action_quantization_feature_export import LatentActionQuantization

# =========================
# Argument parsing
# =========================
parser = argparse.ArgumentParser(
    description="LAQ inference for depth images, saving z_depth_indices and z_depth_feature"
)

parser.add_argument("--input_file", type=str, required=True, help="Path to input JSONL file")
parser.add_argument("--dist_number", type=int, required=True, help="Shard index, starting from 1")
parser.add_argument("--codebook_size", type=int, required=True, help="Codebook size")
parser.add_argument("--laq_checkpoint", type=str, required=True, help="Path to LAQ checkpoint")
parser.add_argument("--divider", type=int, default=1, help="Number of shards")
parser.add_argument("--window_size", type=int, required=True, help="Temporal offset")
parser.add_argument("--code_seq_len", type=int, required=True, help="Code sequence length")
parser.add_argument("--layer", type=int, required=True, help="Spatial and temporal depth")
parser.add_argument("--unshuffled_jsonl", type=str, required=True, help="Output JSONL path")

# feature output
parser.add_argument(
    "--feature_dir",
    type=str,
    required=True,
    help="Output directory for z_depth_feature .pt parts and manifest",
)
parser.add_argument(
    "--feature_prefix",
    type=str,
    default="z_depth_stage1",
    help="Prefix for saved .pt feature parts and manifest",
)
parser.add_argument(
    "--feature_part_size",
    type=int,
    default=8192,
    help="Number of samples per saved feature .pt part",
)
parser.add_argument(
    "--save_z_depth_tokens",
    action="store_true",
    help="Also save token-level z_depth_tokens if returned by LAQ inference",
)

# runtime
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
parser.add_argument(
    "--repeat_depth_to_3ch",
    type=int,
    default=1,
    choices=[0, 1],
    help="Repeat depth from 1 channel to 3 channels. Use 1 if train did the same."
)

# debug
parser.add_argument(
    "--debug_save_dir",
    type=str,
    default="",
    help="Optional folder to save preprocessed depth previews and tensors"
)
parser.add_argument(
    "--debug_num_samples",
    type=int,
    default=5,
    help="How many samples to save for debug"
)

args = parser.parse_args()


# =========================
# Constants
# =========================
dist_number = args.dist_number
batch_size = args.batch_size
repeat_depth_to_3ch = bool(args.repeat_depth_to_3ch)

print("input_file:", args.input_file)
print("laq_checkpoint:", args.laq_checkpoint)
print("repeat_depth_to_3ch:", repeat_depth_to_3ch)
print("feature_dir:", args.feature_dir)
print("feature_prefix:", args.feature_prefix)
print("feature_part_size:", args.feature_part_size)
print("save_z_depth_tokens:", args.save_z_depth_tokens)
print("debug_save_dir:", args.debug_save_dir)
print("debug_num_samples:", args.debug_num_samples)


# =========================
# Helpers
# =========================
def extract_video_id(elem: dict) -> str:
    if "video_id" in elem and elem["video_id"] is not None:
        return str(elem["video_id"])
    sample_id = str(elem.get("id", ""))
    if "_" in sample_id:
        return sample_id.split("_")[0]
    return sample_id


def make_sample_id(video_id: str, image_path: str) -> str:
    frame_name = os.path.basename(image_path)
    frame_stem = os.path.splitext(frame_name)[0]
    return f"{video_id}_{frame_stem}"


def natural_key(p: Path):
    m = re.search(r"(\d+)", p.stem)
    return int(m.group(1)) if m else p.stem


def tensor_to_cpu(x):
    if torch.is_tensor(x):
        return x.detach().cpu()
    return torch.tensor(x).detach().cpu()


def parse_laq_feature_output(out):
    """
    Expected output from updated LatentActionQuantization.inference(..., return_features=True):

        {
            "z_depth_indices": Tensor [B, L],
            "z_depth_feature": Tensor [B, D],
            "z_depth_tokens": Tensor [B, L, D] optional
        }

    Also supports tuple/list:
        (indices, feature)
        (indices, feature, tokens)
    """
    if isinstance(out, dict):
        index_keys = ["z_depth_indices", "indices", "codebook_ids", "codebook_indices", "ids", "delta"]
        feature_keys = ["z_depth_feature", "z_depth_features", "feature", "features", "z_features"]
        token_keys = ["z_depth_tokens", "tokens", "z_tokens"]

        indices = None
        feature = None
        tokens = None

        for k in index_keys:
            if k in out:
                indices = out[k]
                break

        for k in feature_keys:
            if k in out:
                feature = out[k]
                break

        for k in token_keys:
            if k in out:
                tokens = out[k]
                break

        if indices is None or feature is None:
            raise KeyError(
                "Cannot parse LAQ output. Need indices and z_depth_feature. "
                f"Available keys: {list(out.keys())}"
            )

        return indices, feature, tokens

    if isinstance(out, (tuple, list)):
        if len(out) < 2:
            raise RuntimeError(f"Expected tuple/list (indices, feature), got length {len(out)}")

        indices = out[0]
        feature = out[1]
        tokens = out[2] if len(out) >= 3 else None
        return indices, feature, tokens

    raise TypeError(f"Unsupported LAQ output type: {type(out)}")


# =========================
# Load input JSONL
# =========================
processed_jsonl_data = []
with open(args.input_file, "r") as file:
    for line in file:
        processed_jsonl_data.append(json.loads(line))

print(f"processed_jsonl_data: {len(processed_jsonl_data)}")


# =========================
# Build current image + future image pairs
# =========================
window_size = args.window_size
image_paths = []

folder_frames = {}
folder_name_to_idx = {}

for elem in processed_jsonl_data:
    image_path = Path(elem["image"]).resolve()
    folder = image_path.parent

    if folder not in folder_frames:
        frames = sorted(
            [p for p in folder.iterdir() if p.is_file()],
            key=natural_key
        )
        folder_frames[folder] = frames
        folder_name_to_idx[folder] = {p.name: i for i, p in enumerate(frames)}

    frames = folder_frames[folder]
    name_to_idx = folder_name_to_idx[folder]

    if image_path.name not in name_to_idx:
        raise ValueError(f"{image_path} not found in cached frame list")

    cur_idx = name_to_idx[image_path.name]
    next_idx = min(cur_idx + window_size, len(frames) - 1)
    next_image = str(frames[next_idx])

    image_paths.append([str(image_path), next_image])

print(f"image_paths: {len(image_paths)}")


# =========================
# Shard data if needed
# =========================
start = int(int(len(processed_jsonl_data) / batch_size) / args.divider) * batch_size * (dist_number - 1)
end = int(int(len(processed_jsonl_data) / batch_size) / args.divider) * batch_size * dist_number
if dist_number == args.divider:
    end = len(processed_jsonl_data)

print("start, end:", start, end)

processed_jsonl_data = processed_jsonl_data[start:end]
image_paths = image_paths[start:end]

print(f"processed_jsonl_data after shard: {len(processed_jsonl_data)}")
print(f"image_paths after shard: {len(image_paths)}")


# =========================
# Create output dirs
# =========================
unshuffled_jsonl = args.unshuffled_jsonl
parent_dir = os.path.dirname(unshuffled_jsonl)
if parent_dir:
    os.makedirs(parent_dir, exist_ok=True)

feature_dir = Path(args.feature_dir)
feature_dir.mkdir(parents=True, exist_ok=True)

if args.debug_save_dir:
    os.makedirs(args.debug_save_dir, exist_ok=True)


# =========================
# Build model
# =========================
laq = LatentActionQuantization(
    dim=1024,
    quant_dim=32,
    codebook_size=args.codebook_size,
    image_size=256,
    patch_size=32,
    spatial_depth=args.layer,
    temporal_depth=args.layer,
    dim_head=64,
    heads=16,
    code_seq_len=args.code_seq_len,
).cuda()

laq.load(args.laq_checkpoint)
laq.eval()


# =========================
# Depth preprocessing
# =========================
resize_depth = T.Resize((256, 256), antialias=True)

def load_depth(path: str) -> torch.Tensor:
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if depth is None:
        raise RuntimeError(f"Cannot read depth image: {path}")

    if depth.ndim != 2:
        raise RuntimeError(f"Depth image not single-channel: {path}. Got shape {depth.shape}")

    depth = depth.astype(np.float32) / 65535.0
    depth = torch.from_numpy(depth).unsqueeze(0)  # [1, H, W]
    depth = resize_depth(depth)

    if repeat_depth_to_3ch:
        depth = depth.repeat(3, 1, 1)  # [3, H, W]

    return depth


# =========================
# Debug helpers
# =========================
def save_depth_preview(depth_tensor: torch.Tensor, save_prefix: str):
    depth_cpu = depth_tensor.detach().cpu()

    if depth_cpu.ndim != 3:
        raise ValueError(f"Expected [C,H,W], got {tuple(depth_cpu.shape)}")

    depth_vis = depth_cpu[0].float().clamp(0.0, 1.0).numpy()

    fixed_u8 = (depth_vis * 255.0).astype(np.uint8)
    cv2.imwrite(f"{save_prefix}_fixed.png", fixed_u8)

    dmin = float(depth_vis.min())
    dmax = float(depth_vis.max())

    if dmax > dmin:
        auto_vis = (depth_vis - dmin) / (dmax - dmin)
    else:
        auto_vis = np.zeros_like(depth_vis)

    auto_u8 = (auto_vis * 255.0).astype(np.uint8)
    cv2.imwrite(f"{save_prefix}_auto.png", auto_u8)

    torch.save(depth_cpu, f"{save_prefix}.pt")

    with open(f"{save_prefix}_stats.txt", "w") as f:
        f.write(f"shape={tuple(depth_cpu.shape)}\n")
        f.write(f"dtype={depth_cpu.dtype}\n")
        f.write(f"min={depth_cpu.min().item():.8f}\n")
        f.write(f"max={depth_cpu.max().item():.8f}\n")
        f.write(f"mean={depth_cpu.mean().item():.8f}\n")
        f.write(f"std={depth_cpu.std().item():.8f}\n")


# =========================
# Dataset
# =========================
class AsyncDepthDataset(Dataset):
    def __init__(self, file_paths, debug_save_dir="", debug_num_samples=5):
        self.file_paths = file_paths
        self.debug_save_dir = debug_save_dir
        self.debug_num_samples = debug_num_samples

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        cur_path, next_path = self.file_paths[index]

        depth1 = load_depth(cur_path)
        depth2 = load_depth(next_path)

        if self.debug_save_dir and index < self.debug_num_samples:
            print(f"[DEBUG] index={index}")
            print(f"  cur_path={cur_path}")
            print(f"  next_path={next_path}")
            print(
                f"  depth1.shape={tuple(depth1.shape)}, dtype={depth1.dtype}, "
                f"min={depth1.min().item():.6f}, max={depth1.max().item():.6f}, "
                f"mean={depth1.mean().item():.6f}, std={depth1.std().item():.6f}"
            )
            print(
                f"  depth2.shape={tuple(depth2.shape)}, dtype={depth2.dtype}, "
                f"min={depth2.min().item():.6f}, max={depth2.max().item():.6f}, "
                f"mean={depth2.mean().item():.6f}, std={depth2.std().item():.6f}"
            )

            save_depth_preview(depth1, os.path.join(self.debug_save_dir, f"{index:04d}_depth1"))
            save_depth_preview(depth2, os.path.join(self.debug_save_dir, f"{index:04d}_depth2"))

        clip = torch.stack([depth1, depth2], dim=1)  # [C, 2, H, W]
        return clip


# =========================
# Feature writer
# =========================
feature_buffer = {
    "id": [],
    "video_id": [],
    "image": [],
    "depth_pair": [],
    "z_depth_indices": [],
    "z_depth_feature": [],
    "z_depth_tokens": [],
}

feature_parts = []
feature_part_idx = 0
feature_total_samples = 0


def flush_feature_buffer(force=False):
    global feature_part_idx, feature_total_samples

    n = len(feature_buffer["id"])
    if n == 0:
        return

    if not force and n < args.feature_part_size:
        return

    out_path = feature_dir / f"{args.feature_prefix}_part{feature_part_idx:05d}.pt"

    z_depth_indices = torch.stack(feature_buffer["z_depth_indices"], dim=0).long()
    z_depth_feature = torch.stack(feature_buffer["z_depth_feature"], dim=0).float()

    pkg = {
        "id": list(feature_buffer["id"]),
        "video_id": list(feature_buffer["video_id"]),
        "image": list(feature_buffer["image"]),
        "depth_pair": list(feature_buffer["depth_pair"]),
        "z_depth_indices": z_depth_indices,
        "z_depth_feature": z_depth_feature,
    }

    part_info = {
        "part": feature_part_idx,
        "path": str(out_path),
        "num_samples": n,
        "z_depth_indices_shape": list(z_depth_indices.shape),
        "z_depth_feature_shape": list(z_depth_feature.shape),
    }

    if args.save_z_depth_tokens:
        if len(feature_buffer["z_depth_tokens"]) != n:
            raise RuntimeError(
                "save_z_depth_tokens=True but z_depth_tokens were not collected for every sample. "
                "Make sure laq.inference(..., return_features=True) returns z_depth_tokens."
            )
        z_depth_tokens = torch.stack(feature_buffer["z_depth_tokens"], dim=0).float()
        pkg["z_depth_tokens"] = z_depth_tokens
        part_info["z_depth_tokens_shape"] = list(z_depth_tokens.shape)

    torch.save(pkg, out_path)
    feature_parts.append(part_info)

    print(
        f"Saved feature part: {out_path} | samples={n} | "
        f"z_depth_feature_shape={list(z_depth_feature.shape)}"
    )

    feature_total_samples += n
    feature_part_idx += 1

    for k in feature_buffer:
        feature_buffer[k].clear()


# =========================
# Process data
# =========================
def process_data(processed_jsonl_data, laq, image_paths, batch_size, num_workers, debug_save_dir, debug_num_samples):
    cnt2 = 0
    dataset = AsyncDepthDataset(
        image_paths,
        debug_save_dir=debug_save_dir,
        debug_num_samples=debug_num_samples
    )

    dataloader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False
    )

    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = 2

    dataloader = DataLoader(**dataloader_kwargs)

    for img_batch in tqdm(dataloader):
        final_list = []

        with torch.no_grad():
            out = laq.inference(
                img_batch.cuda(non_blocking=True),
                return_features=True,
            )

        index_batch, feature_batch, token_batch = parse_laq_feature_output(out)

        index_batch = tensor_to_cpu(index_batch).long()
        feature_batch = tensor_to_cpu(feature_batch).float()
        token_batch = tensor_to_cpu(token_batch).float() if token_batch is not None else None

        print("Batch indices:", index_batch.shape)
        print("Batch z_depth_feature:", feature_batch.shape)
        if token_batch is not None:
            print("Batch z_depth_tokens:", token_batch.shape)

        if index_batch.ndim != 2:
            raise RuntimeError(f"Expected index_batch [B,L], got {tuple(index_batch.shape)}")

        if feature_batch.shape[0] != index_batch.shape[0]:
            raise RuntimeError(
                f"Feature batch size mismatch: indices={index_batch.shape[0]}, "
                f"features={feature_batch.shape[0]}"
            )

        if token_batch is not None and token_batch.shape[0] != index_batch.shape[0]:
            raise RuntimeError(
                f"Token batch size mismatch: indices={index_batch.shape[0]}, "
                f"tokens={token_batch.shape[0]}"
            )

        index = 0
        batch_start = batch_size * cnt2
        batch_end = min(batch_size * (cnt2 + 1), len(image_paths))

        for idx in range(batch_start, batch_end):
            src_elem = processed_jsonl_data[idx]
            video_id = extract_video_id(src_elem)
            sample_id = make_sample_id(video_id, image_paths[idx][0])

            elem_dict = {}
            elem_dict["id"] = sample_id
            elem_dict["video_id"] = video_id
            elem_dict["image"] = image_paths[idx][0]
            elem_dict["delta"] = [str(i) for i in index_batch[index].tolist()]
            elem_dict["instruction"] = src_elem.get("instruction", "")
            elem_dict["vision"] = src_elem.get("vision", [])
            elem_dict["fields"] = "[instruction],[vision],delta"

            final_list.append(elem_dict)

            feature_buffer["id"].append(sample_id)
            feature_buffer["video_id"].append(video_id)
            feature_buffer["image"].append(image_paths[idx][0])
            feature_buffer["depth_pair"].append(image_paths[idx])
            feature_buffer["z_depth_indices"].append(index_batch[index])
            feature_buffer["z_depth_feature"].append(feature_batch[index])

            if args.save_z_depth_tokens:
                if token_batch is None:
                    raise RuntimeError(
                        "--save_z_depth_tokens was set, but LAQ output does not contain z_depth_tokens"
                    )
                feature_buffer["z_depth_tokens"].append(token_batch[index])

            flush_feature_buffer(force=False)

            index += 1

        cnt2 += 1
        yield final_list


# =========================
# Write output JSONL and feature shards
# =========================
with open(unshuffled_jsonl, "w") as file:
    cnt = 0
    for entry in process_data(
        processed_jsonl_data=processed_jsonl_data,
        laq=laq,
        image_paths=image_paths,
        batch_size=batch_size,
        num_workers=args.num_workers,
        debug_save_dir=args.debug_save_dir,
        debug_num_samples=args.debug_num_samples,
    ):
        for elem in entry:
            file.write(json.dumps(elem, ensure_ascii=False) + "\n")
        cnt += 1

flush_feature_buffer(force=True)

manifest = {
    "prefix": args.feature_prefix,
    "total_samples": feature_total_samples,
    "num_parts": len(feature_parts),
    "feature_dir": str(feature_dir),
    "source_jsonl": args.input_file,
    "unshuffled_jsonl": unshuffled_jsonl,
    "parts": feature_parts,
}

manifest_path = feature_dir / f"{args.feature_prefix}_manifest.json"
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print(f"Done. Wrote output JSONL to: {unshuffled_jsonl}")
print(f"Done. Wrote feature manifest to: {manifest_path}")
if args.debug_save_dir:
    print(f"Debug previews saved to: {args.debug_save_dir}")
