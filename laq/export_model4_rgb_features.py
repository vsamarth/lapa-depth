import argparse
import json
import os
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm
from PIL import Image

from laq_model.latent_action_quantization_feature_export import LatentActionQuantization


# =========================
# Argument parsing
# =========================
parser = argparse.ArgumentParser(
    description="Model 1.4-rgb inference: save z_rgb_indices, z_rgb_feature, and optional z_rgb_tokens"
)

parser.add_argument("--input_file", type=str, required=True, help="Path to input JSONL file")
parser.add_argument("--dist_number", type=int, required=True, help="Shard index, starting from 1")
parser.add_argument("--codebook_size", type=int, required=True, help="Codebook size")
parser.add_argument("--laq_checkpoint", type=str, required=True, help="Path to Model 1.4-rgb LAQ checkpoint")
parser.add_argument("--divider", type=int, default=1, help="Number of shards")
parser.add_argument("--window_size", type=int, required=True, help="Temporal offset")
parser.add_argument("--code_seq_len", type=int, required=True, help="Code sequence length")
parser.add_argument("--layer", type=int, required=True, help="Spatial and temporal depth")
parser.add_argument("--unshuffled_jsonl", type=str, required=True, help="Output JSONL path")

# Feature output
parser.add_argument(
    "--feature_dir",
    type=str,
    required=True,
    help="Output directory for z_rgb .pt parts and manifest",
)

parser.add_argument(
    "--feature_prefix",
    type=str,
    default="z_rgb_model_1_4",
    help="Prefix for saved .pt feature parts and manifest",
)

parser.add_argument(
    "--feature_part_size",
    type=int,
    default=8192,
    help="Number of samples per saved feature .pt part",
)

parser.add_argument(
    "--save_z_rgb_tokens",
    action="store_true",
    help="Also save token-level z_rgb_tokens if returned by LAQ inference",
)

# Runtime
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")

# Debug
parser.add_argument(
    "--debug_save_dir",
    type=str,
    default="",
    help="Optional folder to save preprocessed RGB previews and tensors",
)

parser.add_argument(
    "--debug_num_samples",
    type=int,
    default=5,
    help="How many samples to save for debug",
)

args = parser.parse_args()


# =========================
# Print config
# =========================
print("========== Model 1.4-rgb z_rgb inference ==========")
print("input_file:", args.input_file)
print("dist_number:", args.dist_number)
print("divider:", args.divider)
print("laq_checkpoint:", args.laq_checkpoint)
print("codebook_size:", args.codebook_size)
print("window_size:", args.window_size)
print("code_seq_len:", args.code_seq_len)
print("layer:", args.layer)
print("unshuffled_jsonl:", args.unshuffled_jsonl)
print("feature_dir:", args.feature_dir)
print("feature_prefix:", args.feature_prefix)
print("feature_part_size:", args.feature_part_size)
print("save_z_rgb_tokens:", args.save_z_rgb_tokens)
print("batch_size:", args.batch_size)
print("num_workers:", args.num_workers)
print("debug_save_dir:", args.debug_save_dir)
print("debug_num_samples:", args.debug_num_samples)
print("===================================================")


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
    Expected output from LatentActionQuantization.inference(..., return_features=True):

        {
            "z_rgb_indices": Tensor [B, L],
            "z_rgb_feature": Tensor [B, D],
            "z_rgb_tokens": Tensor [B, L, D] optional
        }

    Also supports old/depth-style keys:
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
        index_keys = [
            "z_rgb_indices",
            "z_depth_indices",
            "indices",
            "codebook_ids",
            "codebook_indices",
            "ids",
            "delta",
        ]

        feature_keys = [
            "z_rgb_feature",
            "z_rgb_features",
            "z_depth_feature",
            "z_depth_features",
            "feature",
            "features",
            "z_features",
        ]

        token_keys = [
            "z_rgb_tokens",
            "z_depth_tokens",
            "tokens",
            "z_tokens",
        ]

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
                "Cannot parse LAQ output. Need indices and feature. "
                f"Available keys: {list(out.keys())}"
            )

        return indices, feature, tokens

    if isinstance(out, (tuple, list)):
        if len(out) < 2:
            raise RuntimeError(f"Expected tuple/list with at least 2 elements, got length {len(out)}")

        indices = out[0]
        feature = out[1]
        tokens = out[2] if len(out) >= 3 else None

        return indices, feature, tokens

    raise TypeError(f"Unsupported LAQ output type: {type(out)}")


# =========================
# Load input JSONL
# =========================
processed_jsonl_data = []

with open(args.input_file, "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()
        if not line:
            continue
        processed_jsonl_data.append(json.loads(line))

print(f"processed_jsonl_data: {len(processed_jsonl_data)}")


# =========================
# Build current RGB + future RGB pairs
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
            key=natural_key,
        )

        if len(frames) == 0:
            raise RuntimeError(f"No image files found in folder: {folder}")

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
dist_number = args.dist_number
batch_size = args.batch_size

if args.divider < 1:
    raise ValueError("--divider must be >= 1")

if dist_number < 1 or dist_number > args.divider:
    raise ValueError("--dist_number must be in [1, divider]")

num_full_batches = int(len(processed_jsonl_data) / batch_size)
batches_per_shard = int(num_full_batches / args.divider)

start = batches_per_shard * batch_size * (dist_number - 1)
end = batches_per_shard * batch_size * dist_number

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

print("Loaded checkpoint:", args.laq_checkpoint)


# =========================
# RGB preprocessing
# =========================
rgb_transform = T.Compose(
    [
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((256, 256)),
        T.ToTensor(),  # [3, H, W], float32, range [0, 1]
    ]
)


def load_rgb(path: str) -> torch.Tensor:
    try:
        img = Image.open(path)
    except Exception as e:
        raise RuntimeError(f"Cannot read RGB image: {path}") from e

    return rgb_transform(img)


# =========================
# Debug helpers
# =========================
def save_rgb_preview(rgb_tensor: torch.Tensor, save_prefix: str):
    """
    Save RGB preview and tensor.

    rgb_tensor: [3, H, W], range [0, 1]
    """
    rgb_cpu = rgb_tensor.detach().cpu()

    if rgb_cpu.ndim != 3 or rgb_cpu.shape[0] != 3:
        raise ValueError(f"Expected [3,H,W], got {tuple(rgb_cpu.shape)}")

    rgb_np = rgb_cpu.clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    rgb_u8 = (rgb_np * 255.0).astype("uint8")

    Image.fromarray(rgb_u8).save(f"{save_prefix}.png")
    torch.save(rgb_cpu, f"{save_prefix}.pt")

    with open(f"{save_prefix}_stats.txt", "w", encoding="utf-8") as f:
        f.write(f"shape={tuple(rgb_cpu.shape)}\n")
        f.write(f"dtype={rgb_cpu.dtype}\n")
        f.write(f"min={rgb_cpu.min().item():.8f}\n")
        f.write(f"max={rgb_cpu.max().item():.8f}\n")
        f.write(f"mean={rgb_cpu.mean().item():.8f}\n")
        f.write(f"std={rgb_cpu.std().item():.8f}\n")


# =========================
# Dataset
# =========================
class AsyncRGBDataset(Dataset):
    def __init__(self, file_paths, debug_save_dir="", debug_num_samples=5):
        self.file_paths = file_paths
        self.debug_save_dir = debug_save_dir
        self.debug_num_samples = debug_num_samples

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        cur_path, next_path = self.file_paths[index]

        rgb1 = load_rgb(cur_path)
        rgb2 = load_rgb(next_path)

        if self.debug_save_dir and index < self.debug_num_samples:
            print(f"[DEBUG] index={index}")
            print(f"  cur_path={cur_path}")
            print(f"  next_path={next_path}")
            print(
                f"  rgb1.shape={tuple(rgb1.shape)}, dtype={rgb1.dtype}, "
                f"min={rgb1.min().item():.6f}, max={rgb1.max().item():.6f}, "
                f"mean={rgb1.mean().item():.6f}, std={rgb1.std().item():.6f}"
            )
            print(
                f"  rgb2.shape={tuple(rgb2.shape)}, dtype={rgb2.dtype}, "
                f"min={rgb2.min().item():.6f}, max={rgb2.max().item():.6f}, "
                f"mean={rgb2.mean().item():.6f}, std={rgb2.std().item():.6f}"
            )

            save_rgb_preview(rgb1, os.path.join(self.debug_save_dir, f"{index:04d}_rgb1"))
            save_rgb_preview(rgb2, os.path.join(self.debug_save_dir, f"{index:04d}_rgb2"))

        # Per sample output shape: [3, 2, 256, 256]
        clip = torch.cat([rgb1.unsqueeze(1), rgb2.unsqueeze(1)], dim=1)

        return clip


# =========================
# Feature writer
# =========================
feature_buffer = {
    "id": [],
    "video_id": [],
    "image": [],
    "rgb_pair": [],
    "z_rgb_indices": [],
    "z_rgb_feature": [],
    "z_rgb_tokens": [],
}

feature_parts = []
feature_part_idx = 0
feature_total_samples = 0


def clear_feature_buffer():
    for k in feature_buffer:
        feature_buffer[k].clear()


def flush_feature_buffer(force=False):
    global feature_part_idx, feature_total_samples

    n = len(feature_buffer["id"])

    if n == 0:
        return

    if not force and n < args.feature_part_size:
        return

    out_path = feature_dir / f"{args.feature_prefix}_part{feature_part_idx:05d}.pt"

    z_rgb_indices = torch.stack(feature_buffer["z_rgb_indices"], dim=0).long()
    z_rgb_feature = torch.stack(feature_buffer["z_rgb_feature"], dim=0).float()

    pkg = {
        "id": list(feature_buffer["id"]),
        "video_id": list(feature_buffer["video_id"]),
        "image": list(feature_buffer["image"]),
        "rgb_pair": list(feature_buffer["rgb_pair"]),
        "z_rgb_indices": z_rgb_indices,
        "z_rgb_feature": z_rgb_feature,
        "z_rgb_features": z_rgb_feature,
    }

    part_info = {
        "part": feature_part_idx,
        "path": str(out_path),
        "num_samples": n,
        "z_rgb_indices_shape": list(z_rgb_indices.shape),
        "z_rgb_feature_shape": list(z_rgb_feature.shape),
    }

    if args.save_z_rgb_tokens:
        if len(feature_buffer["z_rgb_tokens"]) != n:
            raise RuntimeError(
                "save_z_rgb_tokens=True but z_rgb_tokens were not collected for every sample. "
                "Make sure laq.inference(..., return_features=True) returns tokens."
            )

        z_rgb_tokens = torch.stack(feature_buffer["z_rgb_tokens"], dim=0).float()
        pkg["z_rgb_tokens"] = z_rgb_tokens
        part_info["z_rgb_tokens_shape"] = list(z_rgb_tokens.shape)

    torch.save(pkg, out_path)
    feature_parts.append(part_info)

    print(
        f"Saved feature part: {out_path} | samples={n} | "
        f"z_rgb_indices_shape={list(z_rgb_indices.shape)} | "
        f"z_rgb_feature_shape={list(z_rgb_feature.shape)}"
    )

    if args.save_z_rgb_tokens:
        print(f"z_rgb_tokens_shape={part_info.get('z_rgb_tokens_shape')}")

    feature_total_samples += n
    feature_part_idx += 1
    clear_feature_buffer()


# =========================
# Process data
# =========================
def process_data(
    processed_jsonl_data,
    laq,
    image_paths,
    batch_size,
    num_workers,
    debug_save_dir,
    debug_num_samples,
):
    dataset = AsyncRGBDataset(
        image_paths,
        debug_save_dir=debug_save_dir,
        debug_num_samples=debug_num_samples,
    )

    dataloader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )

    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = 2

    dataloader = DataLoader(**dataloader_kwargs)

    global_sample_offset = 0

    for img_batch in tqdm(dataloader):
        with torch.no_grad():
            out = laq.inference(
                img_batch.cuda(non_blocking=True),
                return_features=True,
            )

        index_batch, feature_batch, token_batch = parse_laq_feature_output(out)

        index_batch = tensor_to_cpu(index_batch).long()
        feature_batch = tensor_to_cpu(feature_batch).float()
        token_batch = tensor_to_cpu(token_batch).float() if token_batch is not None else None

        print("Batch z_rgb_indices:", index_batch.shape)
        print("Batch z_rgb_feature:", feature_batch.shape)

        if token_batch is not None:
            print("Batch z_rgb_tokens:", token_batch.shape)

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

        current_batch_size = index_batch.shape[0]
        final_list = []

        for local_idx in range(current_batch_size):
            global_idx = global_sample_offset + local_idx

            src_elem = processed_jsonl_data[global_idx]
            video_id = extract_video_id(src_elem)
            sample_id = make_sample_id(video_id, image_paths[global_idx][0])

            elem_dict = {
                "id": sample_id,
                "video_id": video_id,
                "image": image_paths[global_idx][0],
                "delta": [str(i) for i in index_batch[local_idx].tolist()],
                "instruction": src_elem.get("instruction", ""),
                "vision": src_elem.get("vision", []),
                "fields": "[instruction],[vision],delta",
            }

            final_list.append(elem_dict)

            feature_buffer["id"].append(sample_id)
            feature_buffer["video_id"].append(video_id)
            feature_buffer["image"].append(image_paths[global_idx][0])
            feature_buffer["rgb_pair"].append(image_paths[global_idx])
            feature_buffer["z_rgb_indices"].append(index_batch[local_idx])
            feature_buffer["z_rgb_feature"].append(feature_batch[local_idx])

            if args.save_z_rgb_tokens:
                if token_batch is None:
                    raise RuntimeError(
                        "--save_z_rgb_tokens was set, but LAQ output does not contain tokens"
                    )
                feature_buffer["z_rgb_tokens"].append(token_batch[local_idx])

            flush_feature_buffer(force=False)

        global_sample_offset += current_batch_size

        yield final_list


# =========================
# Write output JSONL and feature shards
# =========================
with open(unshuffled_jsonl, "w", encoding="utf-8") as file:
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

flush_feature_buffer(force=True)

manifest = {
    "model": "model_1_4_rgb",
    "prefix": args.feature_prefix,
    "total_samples": feature_total_samples,
    "num_parts": len(feature_parts),
    "feature_dir": str(feature_dir),
    "source_jsonl": args.input_file,
    "unshuffled_jsonl": unshuffled_jsonl,
    "checkpoint": args.laq_checkpoint,
    "save_z_rgb_tokens": args.save_z_rgb_tokens,
    "parts": feature_parts,
}

manifest_path = feature_dir / f"{args.feature_prefix}_manifest.json"

with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print(f"Done. Wrote output JSONL to: {unshuffled_jsonl}")
print(f"Done. Wrote feature manifest to: {manifest_path}")

if args.debug_save_dir:
    print(f"Debug previews saved to: {args.debug_save_dir}")
