import os
import json
import argparse
import time
from typing import Optional, Dict, List

import numpy as np
from PIL import Image

from latent_pretraining.sampler_latent_pretrain import DeltaSampler
from latent_pretraining.delta_llama import VideoLLaMAConfig
from tux import JaxDistributedConfig, set_random_seed


class FLAGSClass:
    def __init__(self, flag_dict):
        for key, value in flag_dict.items():
            setattr(self, key, value)


class LAPAInference:
    def __init__(self, image_size: int = 256, **kwargs) -> None:
        flags = FLAGSClass(kwargs)
        self.model = DeltaSampler(FLAGS=flags)
        self.image_size = image_size
        self.tokens_per_delta = kwargs["tokens_per_delta"]

    def inference(self, image: np.ndarray, task_description: Optional[str] = None):
        assert image.dtype == np.uint8
        image_pil = Image.fromarray(image)
        prompts = [{"image": [image_pil], "question": task_description}]
        latent_output = self.model(prompts)
        latent_action = latent_output[0]
        return latent_action


def load_all_labels(label_root: str) -> Dict[str, Dict[str, str]]:
    """
    Return:
        {
            "1": {"instruction": "...", "split": "train"},
            "2": {"instruction": "...", "split": "validation"},
            ...
        }
    """
    id_to_meta = {}
    split_files = {
        "train": "train.json",
        # "validation": "validation.json",
    }

    for split_name, filename in split_files.items():
        path = os.path.join(label_root, filename)
        if not os.path.exists(path):
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            vid = str(item["id"])
            instruction = item["label"]
            id_to_meta[vid] = {
                "instruction": instruction,
                "split": split_name,
            }

    return id_to_meta


def sorted_frame_files(frame_dir: str) -> List[str]:
    files = [
        f for f in os.listdir(frame_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    files.sort()
    return files


def load_rgb_image(path: str, image_size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img = img.resize((image_size, image_size))
    return np.array(img, dtype=np.uint8)


def save_debug_json(
    save_path: str,
    video_id: str,
    instruction: str,
    split: str,
    frame_files: List[str],
    z_rgb_indices: List[List[int]],
):
    data = {
        "video_id": video_id,
        "instruction": instruction,
        "split": split,
        "data": [],
    }

    for f, z in zip(frame_files, z_rgb_indices):
        data["data"].append({
            "frame": f,
            "z_rgb": z,
        })

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="something-something-v2")
    parser.add_argument("--frames_dirname", type=str, default="frames_10")
    parser.add_argument("--labels_dirname", type=str, default="labels")
    parser.add_argument("--output_dirname", type=str, default="z_rgb_indices_stage2_val")
    parser.add_argument("--debug_dirname", type=str, default="z_rgb_indices_stage2_val_debug")
    parser.add_argument("--save_debug_json", action="store_true")
    parser.add_argument("--debug_max_videos", type=int, default=20)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--max_videos", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true")

    parser.add_argument("--tokens_per_delta", type=int, default=4)
    parser.add_argument("--vqgan_checkpoint", type=str, default="lapa_checkpoints/vqgan")
    parser.add_argument("--vocab_file", type=str, default="lapa_checkpoints/tokenizer.model")
    parser.add_argument("--multi_image", type=int, default=1)
    parser.add_argument("--jax_distributed", type=dict, default=JaxDistributedConfig.get_default_config())
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--mesh_dim", type=str, default="1,-1,1,1")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--load_llama_config", type=str, default="7b")
    parser.add_argument(
        "--update_llama_config",
        type=str,
        default="dict(delta_vocab_size=8,sample_mode='text',theta=50000000,max_sequence_length=32768,scan_attention=False,scan_query_chunk_size=128,scan_key_chunk_size=128,scan_mlp=False,scan_mlp_chunk_size=8192,scan_layers=True)"
    )
    parser.add_argument("--load_checkpoint", type=str, default="params::lapa_checkpoints/params")
    parser.add_argument("--codebook_size", type=int, default=8)

    args = parser.parse_args()

    args.tokenizer = VideoLLaMAConfig.get_tokenizer_config()
    args.llama = VideoLLaMAConfig.get_default_config()
    args.tokenizer.vocab_file = args.vocab_file

    JaxDistributedConfig.initialize(args.jax_distributed)
    set_random_seed(args.seed)

    lapa = LAPAInference(
        image_size=args.image_size,
        tokens_per_delta=args.tokens_per_delta,
        vqgan_checkpoint=args.vqgan_checkpoint,
        vocab_file=args.vocab_file,
        multi_image=args.multi_image,
        jax_distributed=args.jax_distributed,
        seed=args.seed,
        mesh_dim=args.mesh_dim,
        dtype=args.dtype,
        load_llama_config=args.load_llama_config,
        update_llama_config=args.update_llama_config,
        load_checkpoint=args.load_checkpoint,
        tokenizer=args.tokenizer,
        llama=args.llama,
    )

    dataset_root = args.dataset_root
    frames_root = os.path.join(dataset_root, args.frames_dirname)
    labels_root = os.path.join(dataset_root, args.labels_dirname)
    output_root = os.path.join(dataset_root, args.output_dirname)
    debug_root = os.path.join(dataset_root, args.debug_dirname)

    os.makedirs(output_root, exist_ok=True)
    if args.save_debug_json:
        os.makedirs(debug_root, exist_ok=True)

    id_to_meta = load_all_labels(labels_root)

    video_ids = [
        d for d in os.listdir(frames_root)
        if os.path.isdir(os.path.join(frames_root, d))
    ]
    video_ids.sort(key=lambda x: int(x) if x.isdigit() else x)

    if args.max_videos is not None:
        video_ids = video_ids[:args.max_videos]

    debug_count = 0

    for idx, video_id in enumerate(video_ids):
        if video_id not in id_to_meta:
            print(f"[Skip] video_id={video_id} not found in train/validation label files")
            continue

        npz_path = os.path.join(output_root, f"{video_id}.npz")
        if args.skip_existing and os.path.exists(npz_path):
            print(f"[Skip] {video_id}.npz already exists")
            continue

        frame_dir = os.path.join(frames_root, video_id)
        frame_files = sorted_frame_files(frame_dir)

        if len(frame_files) == 0:
            print(f"[Skip] video_id={video_id} has no frames")
            continue

        instruction = id_to_meta[video_id]["instruction"]
        split = id_to_meta[video_id]["split"]

        z_rgb_indices = []

        t_start = time.time()
        for frame_name in frame_files:
            frame_path = os.path.join(frame_dir, frame_name)
            image = load_rgb_image(frame_path, args.image_size)

            latent_action = lapa.inference(image, instruction)

            if isinstance(latent_action, np.ndarray):
                latent_ids = latent_action.astype(np.int16).tolist()
            else:
                latent_ids = np.array(latent_action, dtype=np.int16).tolist()

            if len(latent_ids) == 1 and isinstance(latent_ids[0], list):
                latent_ids = latent_ids[0]

            z_rgb_indices.append(latent_ids)

        np.savez_compressed(
            npz_path,
            video_id=video_id,
            instruction=instruction,
            split=split,
            frame_files=np.array(frame_files),
            z_rgb_indices=np.array(z_rgb_indices, dtype=np.int16),
        )

        if args.save_debug_json and debug_count < args.debug_max_videos:
            json_path = os.path.join(debug_root, f"{video_id}.json")
            save_debug_json(
                json_path,
                video_id=video_id,
                instruction=instruction,
                split=split,
                frame_files=frame_files,
                z_rgb_indices=z_rgb_indices,
            )
            debug_count += 1

        print(
            f"[{idx+1}/{len(video_ids)}] Saved {video_id}.npz | "
            f"split={split} | frames={len(frame_files)} | "
            f"time={time.time() - t_start:.2f}s"
        )


if __name__ == "__main__":
    main()