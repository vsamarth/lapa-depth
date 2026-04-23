import math
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from einops import rearrange

from laq_model import LatentActionQuantization
from laq_model.data import ImageVideoDataset


def compute_batch_metrics(pred, target, eps=1e-8):
    """
    pred, target: [B, C, H, W], assumed in [0, 1]
    """
    pred = pred.float()
    target = target.float()

    l1 = F.l1_loss(pred, target, reduction="mean").item()
    mse = F.mse_loss(pred, target, reduction="mean").item()
    rmse = math.sqrt(max(mse, 0.0))
    psnr = 10.0 * math.log10(1.0 / max(mse, eps))

    return {
        "l1": l1,
        "mse": mse,
        "rmse": rmse,
        "psnr": psnr,
    }


def save_vis(input_first, gt_last, pred, save_path, nrow=3, max_items=8):
    """
    input_first, gt_last, pred: [B, C, H, W]
    Save a grid of [input_first, gt_last, pred]
    """
    b = min(input_first.shape[0], max_items)
    input_first = input_first[:b]
    gt_last = gt_last[:b]
    pred = pred[:b]

    imgs = torch.stack([input_first, gt_last, pred], dim=0)   # [3, B, C, H, W]
    imgs = rearrange(imgs, "r b c h w -> (b r) c h w")
    imgs = imgs.detach().cpu().clamp(0.0, 1.0)

    grid = make_grid(imgs, nrow=nrow, normalize=True, value_range=(0, 1))
    save_image(grid, save_path)


def load_model(checkpoint_path, token_embed_path, device):
    model = LatentActionQuantization(
        dim=1024,
        quant_dim=32,
        codebook_size=8,
        image_size=256,
        patch_size=32,
        spatial_depth=8,
        temporal_depth=8,
        dim_head=64,
        heads=16,
        code_seq_len=4,
    ).to(device)

    # 1) load full LAQ checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)

    # 2) override rgb_token_embed with the separately saved Case-3 token embedding
    token_ckpt = torch.load(token_embed_path, map_location=device)
    model.rgb_token_embed.load_state_dict(token_ckpt["rgb_token_embed"])

    print(f"Loaded token embed from: {token_embed_path}")
    if "meta" in token_ckpt:
        print("Token embed meta:", token_ckpt["meta"])

    model.eval()
    return model


def main():
    # ======== Config ========
    rgb_path = "/media/do/data1/philo/lapa/something-something-v2/ssv2-mini-2k-5/frames_val"
    depth_path = "/media/do/data1/philo/lapa/something-something-v2/ssv2-mini-2k-5/depth_val"
    z_rgb_path = "/media/do/data1/philo/lapa/something-something-v2/ssv2-mini-2k-5/z_rgb_indices_stage2_val"

    checkpoint_path = "/home/do/Workspace/philo/External/LAPA/laq/results_case3/vae.9500.pt"
    token_embed_path = "/home/do/Workspace/philo/External/LAPA/laq/results_case3/token_embed.9500.pt"

    offsets = 30
    batch_size = 1
    num_workers = 0
    modality = "both"   # Case 3: depth1 + z_rgb_latent -> predict depth2

    output_dir = Path("eval_case3_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    save_every = 20
    max_vis_batches = 600

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ======== Dataset / Loader ========
    ds = ImageVideoDataset(
        rgb_path,
        depth_path,
        z_rgb_path,
        image_size=(256, 256),
        offset=offsets
    )

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    # ======== Model ========
    model = load_model(checkpoint_path, token_embed_path, device)
    print(f"Loaded checkpoint from: {checkpoint_path}")

    ckpt_name = Path(checkpoint_path).stem
    token_name = Path(token_embed_path).stem

    # ======== Eval Loop ========
    total_l1 = 0.0
    total_mse = 0.0
    total_rmse = 0.0
    total_psnr = 0.0
    total_batches = 0

    vis_saved = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dl):
            rgb, depth, z_rgb = batch

            rgb = rgb.to(device)
            depth = depth.to(device)
            z_rgb = z_rgb.to(device).long()

            if modality == "rgb":
                video = rgb
                recons = model(video=video, return_recons_only=True)
                gt_last = rgb[:, :, -1]
                input_first = rgb[:, :, 0]

            elif modality == "depth":
                video = depth
                recons = model(video=video, return_recons_only=True)
                gt_last = depth[:, :, -1]
                input_first = depth[:, :, 0]

            elif modality == "both":
                # Case 3:
                # current depth + z_rgb_index embedding -> predict future depth
                video = depth
                decoder_video = depth

                recons = model(
                    video=video,
                    decoder_video=decoder_video,
                    depth=depth,
                    z_rgb=z_rgb,
                    return_recons_only=True
                )

                gt_last = depth[:, :, -1]
                input_first = depth[:, :, 0]

            else:
                raise ValueError(f"Unsupported modality: {modality}")

            metrics = compute_batch_metrics(recons, gt_last)
            total_l1 += metrics["l1"]
            total_mse += metrics["mse"]
            total_rmse += metrics["rmse"]
            total_psnr += metrics["psnr"]
            total_batches += 1

            if batch_idx % save_every == 0 and vis_saved < max_vis_batches:
                save_path = output_dir / f"{ckpt_name}_{token_name}_batch_{batch_idx:05d}.png"
                save_vis(
                    input_first=input_first,
                    gt_last=gt_last,
                    pred=recons,
                    save_path=str(save_path),
                    nrow=3,
                    max_items=1,
                )
                vis_saved += 1

            if batch_idx % 20 == 0:
                print(
                    f"[{batch_idx}/{len(dl)}] "
                    f"L1={metrics['l1']:.6f}, "
                    f"MSE={metrics['mse']:.6f}, "
                    f"RMSE={metrics['rmse']:.6f}, "
                    f"PSNR={metrics['psnr']:.4f}"
                )

    # ======== Final Results ========
    final_metrics = {
        "checkpoint": checkpoint_path,
        "token_embed_checkpoint": token_embed_path,
        "num_batches": total_batches,
        "avg_l1": total_l1 / max(total_batches, 1),
        "avg_mse": total_mse / max(total_batches, 1),
        "avg_rmse": total_rmse / max(total_batches, 1),
        "avg_psnr": total_psnr / max(total_batches, 1),
    }

    print("\n===== Final Eval Results =====")
    for k, v in final_metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

    metrics_path = output_dir / f"{ckpt_name}_{token_name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)

    print(f"\nSaved metrics to: {metrics_path}")
    print(f"Saved visualizations to: {output_dir}")


if __name__ == "__main__":
    main()