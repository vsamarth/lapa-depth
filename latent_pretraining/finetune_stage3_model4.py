#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import albumentations as A
import cv2
import flax
import jax
import jax.numpy as jnp
import msgpack
import numpy as np
import optax
import torch
from PIL import Image
from flax import linen as nn
from flax.serialization import from_bytes, from_state_dict, to_state_dict
from flax.training import checkpoints, train_state
from flax.traverse_util import empty_node, flatten_dict, unflatten_dict
from ml_collections import ConfigDict
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from tux import JaxDistributedConfig, open_file, set_random_seed

from laq.laq_model.latent_action_quantization_stage25_feature_model4 import (
    LatentActionQuantizationStage25Model4,
)
from latent_pretraining.data import get_preprocessor
from latent_pretraining.delta_llama_action import (
    FlaxDeltaActionLaMAForCausalLMModule,
    VideoLLaMAConfig,
)
from latent_pretraining.vqgan import VQGAN


class ActionMLP(nn.Module):
    hidden_dim: int = 1024
    out_dim: int = 7

    @nn.compact
    def __call__(self, x):
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.hidden_dim // 2)(x)
        x = nn.gelu(x)
        return nn.Dense(self.out_dim)(x)


@flax.struct.dataclass
class Batch:
    input_ids: np.ndarray
    vision_masks: np.ndarray
    delta_masks: np.ndarray
    action_masks: np.ndarray
    attention_mask: np.ndarray
    depth1: np.ndarray
    actions: np.ndarray


class Stage3RealDataset(Dataset):
    def __init__(self, dataset_path: str, data_root: str = ""):
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"dataset_path not found: {path}")

        self.data_root = Path(data_root).resolve() if data_root else None
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                self.items = [json.loads(line) for line in handle if line.strip()]
        else:
            with path.open("r", encoding="utf-8") as handle:
                self.items = json.load(handle)

        if not self.items:
            raise RuntimeError(f"Dataset is empty: {path}")

    def __len__(self) -> int:
        return len(self.items)

    def _resolve_path(self, value: str) -> str:
        if os.path.isabs(value):
            return value
        if self.data_root is None:
            return str(Path(value).resolve())
        normalized = value[5:] if value.startswith("data/") else value
        return str((self.data_root / normalized).resolve())

    def _extract_instruction(self, item: Dict[str, Any]) -> str:
        if "instruction" in item:
            return str(item["instruction"])

        conversations = item.get("conversations")
        if isinstance(conversations, list) and conversations:
            first = conversations[0]
            if isinstance(first, dict) and "value" in first:
                prompt = str(first["value"]).replace("<image>\n", "")
                return f"<s> You are a helpful assistant. USER: {prompt} ASSISTANT:"

        raise KeyError("Could not find instruction or conversations[0].value")

    def _extract_actions(self, item: Dict[str, Any]) -> np.ndarray:
        if "raw_actions" in item:
            raw = item["raw_actions"]
        else:
            conversations = item.get("conversations")
            if not isinstance(conversations, list) or len(conversations) < 2:
                raise KeyError("Could not find raw_actions")
            raw = conversations[1]["raw_actions"]
        arr = np.asarray(raw, dtype=np.float32)
        if arr.shape != (7,):
            raise RuntimeError(f"Expected raw_actions shape (7,), got {arr.shape}")
        return arr

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.items[index]
        image_key = "image"
        depth_key = "depth" if "depth" in item else "depth_path"
        if image_key not in item or depth_key not in item:
            raise KeyError(f"Sample missing image/depth keys: {item.keys()}")

        return {
            "id": str(item.get("id", index)),
            "image_path": self._resolve_path(str(item[image_key])),
            "depth_path": self._resolve_path(str(item[depth_key])),
            "instruction": self._extract_instruction(item),
            "raw_actions": self._extract_actions(item),
        }


def collate_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "id": [record["id"] for record in records],
        "image_path": [record["image_path"] for record in records],
        "depth_path": [record["depth_path"] for record in records],
        "instruction": [record["instruction"] for record in records],
        "raw_actions": np.stack([record["raw_actions"] for record in records], axis=0),
    }


def load_depth_batch(
    depth_paths: Sequence[str],
    *,
    depth_scale: float,
    repeat_depth_to_3ch: bool,
    image_size: int,
) -> np.ndarray:
    resize = A.Resize(image_size, image_size, interpolation=cv2.INTER_NEAREST)
    out: List[np.ndarray] = []
    for path in depth_paths:
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise RuntimeError(f"Cannot read depth image: {path}")
        if depth.ndim != 2:
            raise RuntimeError(f"Expected single-channel depth image, got shape {depth.shape} for {path}")
        depth = depth.astype(np.float32) / depth_scale
        depth = np.clip(depth, 0.0, 1.0)
        depth = resize(image=depth)["image"]
        depth = depth[None, ...]
        if repeat_depth_to_3ch:
            depth = np.repeat(depth, 3, axis=0)
        out.append(depth.astype(np.float32))
    return np.stack(out, axis=0)


def process_rgb_batch(image_paths: Sequence[str], preprocessor: A.Compose) -> np.ndarray:
    images = []
    for path in image_paths:
        image = np.array(Image.open(open_file(path, "rb")))
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)
        elif image.ndim == 3 and image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        images.append(image.astype(np.uint8))
    processed = np.array([preprocessor(image=img)["image"] for img in images])
    return (processed / 127.5 - 1.0).astype(np.float32)


def build_sequence_batch(
    records: Dict[str, Any],
    *,
    tokenizer,
    vqgan: VQGAN,
    preprocessor: A.Compose,
    n_tokens_per_frame: int,
) -> Batch:
    vision_start = tokenizer.encode("<vision>")
    vision_end = tokenizer.encode("</vision>")
    eof_token = 8192
    eov_token = 8193

    rgb_images = process_rgb_batch(records["image_path"], preprocessor)
    encoded_images = jax.device_get(vqgan.encode(rgb_images))[1].astype(np.int32)

    token_rows: List[List[int]] = []
    vision_rows: List[List[bool]] = []
    for instruction, vision_tokens in zip(records["instruction"], encoded_images):
        flat_vision = vision_tokens.reshape(-1).tolist()
        if len(flat_vision) != n_tokens_per_frame:
            raise RuntimeError(
                f"Expected {n_tokens_per_frame} vision tokens, got {len(flat_vision)}"
            )

        tokens = [tokenizer.bos_token_id]
        vision_mask = [False]

        prompt_tokens = tokenizer.encode(instruction)
        tokens.extend(prompt_tokens)
        vision_mask.extend([False] * len(prompt_tokens))

        tokens.extend(vision_start)
        vision_mask.extend([False] * len(vision_start))

        tokens.extend(flat_vision)
        vision_mask.extend([True] * len(flat_vision))

        tokens.append(eov_token)
        vision_mask.append(True)

        tokens.extend(vision_end)
        vision_mask.extend([False] * len(vision_end))

        tokens.append(tokenizer.eos_token_id)
        vision_mask.append(False)

        token_rows.append(tokens)
        vision_rows.append(vision_mask)

    max_len = max(len(tokens) for tokens in token_rows)
    batch_size = len(token_rows)

    input_ids = np.full((batch_size, max_len), tokenizer.pad_token_id, dtype=np.int32)
    vision_masks = np.zeros((batch_size, max_len), dtype=bool)
    attention_mask = np.zeros((batch_size, max_len), dtype=np.int32)

    for idx, (tokens, vision_mask) in enumerate(zip(token_rows, vision_rows)):
        input_ids[idx, : len(tokens)] = np.asarray(tokens, dtype=np.int32)
        vision_masks[idx, : len(vision_mask)] = np.asarray(vision_mask, dtype=bool)
        attention_mask[idx, : len(tokens)] = 1

    depth1 = load_depth_batch(
        records["depth_path"],
        depth_scale=65535.0,
        repeat_depth_to_3ch=True,
        image_size=256,
    )

    zeros_mask = np.zeros_like(vision_masks, dtype=bool)
    actions = np.asarray(records["raw_actions"], dtype=np.float32)

    return Batch(
        input_ids=input_ids,
        vision_masks=vision_masks,
        delta_masks=zeros_mask,
        action_masks=zeros_mask,
        attention_mask=attention_mask,
        depth1=depth1,
        actions=actions,
    )


def build_llama_config(args, tokenizer) -> VideoLLaMAConfig:
    if args.load_llama_config:
        llama_config = VideoLLaMAConfig.load_config(args.load_llama_config)
    else:
        llama_config = VideoLLaMAConfig(**VideoLLaMAConfig.get_default_config())

    if args.update_llama_config:
        llama_config.update(dict(eval(args.update_llama_config)))

    llama_config.update(
        dict(
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    )
    return llama_config


def load_streaming_tree(path, target):
    flattened_target = flatten_dict(to_state_dict(target), keep_empty_nodes=True)
    flattened_tree: Dict[tuple, Any] = {}

    with open_file(path) as handle:
        unpacker = msgpack.Unpacker(handle, read_size=83886080, max_buffer_size=32 * 2**30)
        for key, value in unpacker:
            flattened_tree[tuple(key)] = from_bytes(None, value)

    for key, value in flattened_target.items():
        if key not in flattened_tree and value == empty_node:
            flattened_tree[key] = value

    tree = unflatten_dict(flattened_tree)
    return from_state_dict(target, tree)


def load_vla_params(module, config, args):
    seq_len = 32
    init_params = module.init(
        jax.random.PRNGKey(args.seed),
        input_ids=jnp.zeros((1, seq_len), dtype=jnp.int32),
        vision_masks=jnp.zeros((1, seq_len), dtype=bool),
        delta_masks=jnp.zeros((1, seq_len), dtype=bool),
        action_masks=jnp.zeros((1, seq_len), dtype=bool),
        attention_mask=jnp.ones((1, seq_len), dtype=jnp.int32),
        deterministic=True,
        output_hidden_states=True,
    )
    if not args.load_checkpoint:
        return init_params["params"]

    load_type, load_path = args.load_checkpoint.split("::", 1)
    if load_type != "params":
        raise ValueError(
            f"Unsupported load_checkpoint type '{load_type}'. "
            "Use params::/path/to/checkpoint for this stage-3 trainer."
        )
    return load_streaming_tree(load_path, init_params["params"])


def create_model4(args) -> LatentActionQuantizationStage25Model4:
    model4 = LatentActionQuantizationStage25Model4(
        dim=1024,
        image_size=256,
        patch_size=32,
        spatial_depth=8,
        dim_head=64,
        heads=16,
        code_seq_len=4,
        z_rgb_feature_dim=4096,
        z_depth_feature_dim=1024,
        predict_token_features=False,
        feature_loss_weight=1.0,
        cosine_loss_weight=0.1,
    )
    model4.load(args.model4_checkpoint, strict=False)
    model4.eval()
    for param in model4.parameters():
        param.requires_grad_(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model4.to(device)


def compute_model4_depth_feature(model4, depth1: np.ndarray, z_rgb_feature: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        device = next(model4.parameters()).device
        depth_tensor = torch.from_numpy(depth1).to(device=device, dtype=torch.float32)
        rgb_tensor = torch.from_numpy(z_rgb_feature).to(device=device, dtype=torch.float32)
        z_depth = model4.extract_z_depth_feature(depth_tensor, rgb_tensor)
        return z_depth.detach().cpu().numpy().astype(np.float32)


def pool_rgb_feature(hidden_states: jnp.ndarray, vision_masks: jnp.ndarray) -> jnp.ndarray:
    weights = vision_masks.astype(hidden_states.dtype)[..., None]
    denom = jnp.clip(weights.sum(axis=1), a_min=1.0)
    return (hidden_states * weights).sum(axis=1) / denom


def create_train_state(vla_params, head_module, learning_rate):
    head_params = head_module.init(
        jax.random.PRNGKey(0),
        jnp.zeros((1, 4096 + 1024), dtype=jnp.float32),
    )["params"]
    params = flax.core.freeze({"vla": vla_params, "head": head_params})
    tx = optax.adamw(learning_rate=learning_rate, weight_decay=0.0)
    return train_state.TrainState.create(apply_fn=None, params=params, tx=tx)


def make_extract_fn(vla_module):
    @jax.jit
    def extract_rgb(params, batch: Batch) -> jnp.ndarray:
        outputs = vla_module.apply(
            {"params": params["vla"]},
            jnp.asarray(batch.input_ids),
            jnp.asarray(batch.vision_masks),
            jnp.asarray(batch.delta_masks),
            jnp.asarray(batch.action_masks),
            attention_mask=jnp.asarray(batch.attention_mask),
            deterministic=True,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]
        return pool_rgb_feature(hidden_states, jnp.asarray(batch.vision_masks))

    return extract_rgb


def make_train_step(vla_module, head_module):
    @jax.jit
    def train_step(state, batch: Batch, z_depth_feature: jnp.ndarray, rng):
        def loss_fn(params):
            outputs = vla_module.apply(
                {"params": params["vla"]},
                jnp.asarray(batch.input_ids),
                jnp.asarray(batch.vision_masks),
                jnp.asarray(batch.delta_masks),
                jnp.asarray(batch.action_masks),
                attention_mask=jnp.asarray(batch.attention_mask),
                deterministic=False,
                rngs={"dropout": rng},
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states[-1]
            z_rgb_feature = pool_rgb_feature(hidden_states, jnp.asarray(batch.vision_masks))
            fused = jnp.concatenate([z_rgb_feature, z_depth_feature], axis=-1)
            pred_actions = head_module.apply({"params": params["head"]}, fused)
            action_targets = jnp.asarray(batch.actions)
            mse_loss = jnp.mean((pred_actions - action_targets) ** 2)
            l1_loss = jnp.mean(jnp.abs(pred_actions - action_targets))
            loss = mse_loss + 0.1 * l1_loss
            metrics = {
                "loss": loss,
                "mse_loss": mse_loss,
                "l1_loss": l1_loss,
            }
            return loss, metrics

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, metrics), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        metrics = {
            **metrics,
            "grad_norm": optax.global_norm(grads),
            "param_norm": optax.global_norm(state.params),
        }
        return state, metrics

    return train_step


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage-3 finetuning with trainable VLA, frozen model4, and continuous real-action MLP.",
    )
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--data_root", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--load_checkpoint", required=True)
    parser.add_argument("--model4_checkpoint", required=True)
    parser.add_argument("--vqgan_checkpoint", required=True)
    parser.add_argument("--vocab_file", required=True)
    parser.add_argument("--load_llama_config", default="7b")
    parser.add_argument(
        "--update_llama_config",
        default="dict(action_vocab_size=256,delta_vocab_size=8,theta=50000000,max_sequence_length=2048,use_flash_attention=True,scan_attention=False,scan_query_chunk_size=128,scan_key_chunk_size=128,remat_attention='nothing_saveable',scan_mlp=False,scan_mlp_chunk_size=8192,remat_mlp='nothing_saveable',remat_block='nothing_saveable',scan_layers=True)",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--total_steps", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_aug", action="store_true")
    parser.add_argument("--resume", default="")
    parser.add_argument("--jax_distributed", type=dict, default=JaxDistributedConfig.get_default_config())
    return parser.parse_args()


def save_metadata(args, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {key: value for key, value in vars(args).items() if key != "jax_distributed"}
    with (output_dir / "stage3_config.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def infinite_loader(loader: Iterable[Dict[str, Any]]):
    while True:
        for batch in loader:
            yield batch


def main():
    args = parse_args()
    JaxDistributedConfig.initialize(args.jax_distributed)
    set_random_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    tokenizer_config = VideoLLaMAConfig.get_tokenizer_config()
    tokenizer_config.vocab_file = args.vocab_file
    tokenizer = VideoLLaMAConfig.get_tokenizer(tokenizer_config)

    llama_config = build_llama_config(args, tokenizer)
    dtype = jnp.bfloat16 if jax.default_backend() != "cpu" else jnp.float32
    vla_module = FlaxDeltaActionLaMAForCausalLMModule(llama_config, dtype=dtype)
    head_module = ActionMLP()

    vla_params = load_vla_params(vla_module, llama_config, args)
    state = create_train_state(vla_params, head_module, args.learning_rate)

    output_dir = Path(args.output_dir).resolve()
    save_metadata(args, output_dir)
    if args.resume:
        state = checkpoints.restore_checkpoint(args.resume, target=state)

    dataset = Stage3RealDataset(args.dataset_path, data_root=args.data_root)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_records,
        drop_last=False,
    )
    loader_iter = infinite_loader(loader)

    preprocessor = get_preprocessor(image_aug=args.image_aug)
    vqgan = VQGAN(args.vqgan_checkpoint, replicate=False)
    model4 = create_model4(args)

    extract_rgb = make_extract_fn(vla_module)
    train_step = make_train_step(vla_module, head_module)

    rng = jax.random.PRNGKey(args.seed)
    progress = tqdm(range(args.total_steps), ncols=0)
    for step in progress:
        raw_batch = next(loader_iter)
        batch = build_sequence_batch(
            raw_batch,
            tokenizer=tokenizer,
            vqgan=vqgan,
            preprocessor=preprocessor,
            n_tokens_per_frame=256,
        )

        z_rgb_feature = np.asarray(extract_rgb(state.params, batch), dtype=np.float32)
        z_depth_feature = compute_model4_depth_feature(model4, batch.depth1, z_rgb_feature)

        rng, step_rng = jax.random.split(rng)
        state, metrics = train_step(state, batch, jnp.asarray(z_depth_feature), step_rng)

        if step % args.log_every == 0:
            metrics_np = {k: float(v) for k, v in jax.device_get(metrics).items()}
            metrics_np["step"] = step
            progress.write(json.dumps(metrics_np))

        if args.save_every > 0 and (step + 1) % args.save_every == 0:
            checkpoints.save_checkpoint(
                ckpt_dir=str(output_dir),
                target=state,
                step=step + 1,
                overwrite=True,
                keep=3,
            )

    checkpoints.save_checkpoint(
        ckpt_dir=str(output_dir),
        target=state,
        step=args.total_steps,
        overwrite=True,
        keep=3,
    )


if __name__ == "__main__":
    main()
