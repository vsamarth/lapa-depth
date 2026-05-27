#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from laq_model.data_stage25_feature_model4 import Stage252DatasetModel4
from laq_model.laq_stage25_trainer_feature_model4 import LAQStage25TrainerModel4
from laq_model.latent_action_quantization_stage25_feature_model4 import (
    LatentActionQuantizationStage25Model4,
)


DEFAULT_Z_DEPTH_PATH = (
    "/datasets/ssv2_libero_90/stage2_z_rgb_ssv2_libero90/"
    "z_depth_train_mixed.jsonl"
)

DEFAULT_Z_RGB_FEATURE_MANIFEST = (
    "/datasets/ssv2_libero_90/stage2_z_rgb_ssv2_libero90/"
    "z_rgb_train_mixed_manifest.json"
)

DEFAULT_Z_DEPTH_FEATURE_MANIFEST = (
    "/datasets/ssv2_libero_90/stage2_z_rgb_ssv2_libero90/"
    "z_depth_train_mixed_manifest.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune stage-25 model4 from a pretrained checkpoint.",
    )

    parser.add_argument("--pretrained_checkpoint", required=True)
    parser.add_argument("--resume_checkpoint", default=None)
    parser.add_argument("--z_depth_path", default=DEFAULT_Z_DEPTH_PATH)
    parser.add_argument("--z_rgb_feature_manifest", default=DEFAULT_Z_RGB_FEATURE_MANIFEST)
    parser.add_argument("--z_depth_feature_manifest", default=DEFAULT_Z_DEPTH_FEATURE_MANIFEST)
    parser.add_argument("--results_folder", default="results_model4_finetune")
    parser.add_argument("--num_train_steps", type=int, default=65001)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_model_every", type=int, default=5000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--strict_model", action="store_true")
    parser.add_argument("--strict_resume", action="store_true")

    return parser


def infer_feature_dims(dataset):
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty; cannot infer feature dimensions.")

    first = dataset[0]
    last = dataset[-1]

    for name, sample in [("first", first), ("last", last)]:
        if "z_rgb_features" not in sample:
            raise KeyError(f"{name} sample missing z_rgb_features")
        if "z_depth_feature" not in sample:
            raise KeyError(f"{name} sample missing z_depth_feature")

    z_rgb_features = first["z_rgb_features"]
    z_depth_feature = first["z_depth_feature"]

    if not torch.is_tensor(z_rgb_features):
        raise TypeError(f"Expected tensor z_rgb_features, got {type(z_rgb_features)}")
    if not torch.is_tensor(z_depth_feature):
        raise TypeError(f"Expected tensor z_depth_feature, got {type(z_depth_feature)}")

    if z_rgb_features.ndim != 1:
        raise RuntimeError(f"Expected z_rgb_features shape [D], got {tuple(z_rgb_features.shape)}")

    if z_depth_feature.ndim == 1:
        z_depth_feature_dim = int(z_depth_feature.shape[0])
        predict_token_features = False
    elif z_depth_feature.ndim == 2:
        z_depth_feature_dim = int(z_depth_feature.shape[1])
        predict_token_features = True
    else:
        raise RuntimeError(
            "Unsupported z_depth_feature shape: "
            f"{tuple(z_depth_feature.shape)}. Expected [D] or [L, D]."
        )

    if last["z_rgb_features"].shape != z_rgb_features.shape:
        raise RuntimeError(
            "z_rgb_features shape mismatch between first and last sample: "
            f"first={tuple(z_rgb_features.shape)} last={tuple(last['z_rgb_features'].shape)}"
        )

    if last["z_depth_feature"].ndim != z_depth_feature.ndim:
        raise RuntimeError(
            "z_depth_feature rank mismatch between first and last sample: "
            f"first={z_depth_feature.ndim} last={last['z_depth_feature'].ndim}"
        )

    if z_depth_feature.ndim == 1 and last["z_depth_feature"].shape != z_depth_feature.shape:
        raise RuntimeError(
            "z_depth_feature shape mismatch between first and last sample: "
            f"first={tuple(z_depth_feature.shape)} last={tuple(last['z_depth_feature'].shape)}"
        )

    if z_depth_feature.ndim == 2 and last["z_depth_feature"].shape[1] != z_depth_feature.shape[1]:
        raise RuntimeError(
            "z_depth_feature token width mismatch between first and last sample: "
            f"first={tuple(z_depth_feature.shape)} last={tuple(last['z_depth_feature'].shape)}"
        )

    return {
        "z_rgb_feature_dim": int(z_rgb_features.shape[0]),
        "z_depth_feature_dim": z_depth_feature_dim,
        "predict_token_features": predict_token_features,
    }


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    dataset = Stage252DatasetModel4(
        z_depth_path=args.z_depth_path,
        z_rgb_feature_manifest=args.z_rgb_feature_manifest,
        z_depth_feature_manifest=args.z_depth_feature_manifest,
        image_size=256,
        repeat_depth_to_3ch=True,
        depth_scale=65535.0,
        check_length_alignment=True,
        keep_z_rgb_indices=False,
    )

    feature_dims = infer_feature_dims(dataset)

    model = LatentActionQuantizationStage25Model4(
        dim=1024,
        image_size=256,
        patch_size=32,
        spatial_depth=8,
        dim_head=64,
        heads=16,
        code_seq_len=4,
        z_rgb_feature_dim=feature_dims["z_rgb_feature_dim"],
        z_depth_feature_dim=feature_dims["z_depth_feature_dim"],
        predict_token_features=feature_dims["predict_token_features"],
        feature_loss_weight=1.0,
        cosine_loss_weight=0.1,
    ).cuda()

    model.load(args.pretrained_checkpoint, strict=args.strict_model)

    trainer = LAQStage25TrainerModel4(
        model,
        dataset=dataset,
        batch_size=args.batch_size,
        grad_accum_every=1,
        num_train_steps=args.num_train_steps,
        results_folder=args.results_folder,
        lr=args.lr,
        save_model_every=args.save_model_every,
        log_every=args.log_every,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        wandb_project="lapa_depth_model4",
        wandb_run_name=Path(args.results_folder).name,
        save_best=True,
        best_metric="loss",
    )

    if args.resume_checkpoint:
        trainer.load(args.resume_checkpoint, strict=args.strict_resume)

    trainer.train()


if __name__ == "__main__":
    main()
