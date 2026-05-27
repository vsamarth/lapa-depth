"""
LAQ Stage 2.5 Model 4 Trainer

Purpose:
    Train a geometry-aware latent feature predictor:

        depth1 + z_rgb_features -> pred_z_depth_feature

    with ground truth:

        z_depth_feature

This version does NOT use z_depth_indices.

Expected model API:
    model(
        depth1=depth1,
        z_rgb_features=z_rgb_features,
        z_depth_feature=z_depth_feature,
    )

Expected model output:
    loss, logs, pred_z_depth_feature

Expected batch format:
    {
        "depth1": torch.FloatTensor [B, C, H, W],
        "z_rgb_features": torch.FloatTensor [B, feature_dim], usually [B, 4096],
        "z_depth_feature": torch.FloatTensor [B, D] or [B, L, D],
    }
"""

from pathlib import Path
from typing import Callable, Optional, Dict, Any

import json
import time

import wandb
import torch
from torch import nn
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedDataParallelKwargs

from laq_model.optimizer import get_optimizer
from torchvision.utils import save_image, make_grid


def noop(*args, **kwargs):
    pass


def cycle(dl):
    while True:
        for data in dl:
            yield data


def detach_item(value):
    if torch.is_tensor(value):
        return value.detach().float().mean().item()
    if isinstance(value, (int, float)):
        return float(value)
    return value


class LAQStage25TrainerModel4(nn.Module):
    """
    Trainer for Model 4:

        depth1 + z_rgb_features -> z_depth_feature

    Main loss is computed inside model.forward():

        MSE(pred_z_depth_feature, gt_z_depth_feature)
        + cosine loss

    No z_depth_indices, no token accuracy, no CE loss.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        dataset,
        num_train_steps: int,
        batch_size: int,
        lr: float = 1e-4,
        wd: float = 0.0,
        grad_accum_every: int = 1,
        max_grad_norm: Optional[float] = 0.5,
        save_model_every: int = 1000,
        results_folder: str = "./model4_stage25_feature_distill_results",
        use_wandb: bool = True,
        wandb_project: str = "lapa_depth_model4",
        wandb_run_name: Optional[str] = None,
        accelerate_kwargs: Optional[Dict[str, Any]] = None,
        num_workers: int = 4,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        log_every: int = 50,
        debug_save_input: bool = True,
        debug_num_samples: int = 8,
        save_final: bool = True,
        save_best: bool = True,
        best_metric: str = "loss",
    ):
        super().__init__()

        if accelerate_kwargs is None:
            accelerate_kwargs = {}

        if best_metric not in {"loss", "mse", "cosine"}:
            raise ValueError(f"best_metric must be one of 'loss', 'mse', 'cosine', got {best_metric}")

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            **accelerate_kwargs,
            kwargs_handlers=[ddp_kwargs],
        )

        self.model = model
        self.dataset = dataset

        self.num_train_steps = int(num_train_steps)
        self.batch_size = int(batch_size)
        self.grad_accum_every = int(grad_accum_every)
        self.max_grad_norm = max_grad_norm
        self.save_model_every = int(save_model_every)
        self.log_every = int(log_every)
        self.save_final = bool(save_final)
        self.save_best = bool(save_best)
        self.best_metric = best_metric

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)
        self.results_folder_str = str(self.results_folder)

        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name or self.results_folder.name

        self.register_buffer("steps", torch.tensor([0], dtype=torch.long))

        self.debug_save_input = debug_save_input
        self.debug_num_samples = debug_num_samples
        self.debug_saved_input = False

        self.best_loss = float("inf")
        self.best_mse = float("inf")
        self.best_cosine = float("inf")
        self.best_step = -1

        self.optim = get_optimizer(
            self.model.parameters(),
            lr=lr,
            wd=wd,
        )

        loader_kwargs = dict(
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = prefetch_factor

        self.dl = DataLoader(dataset, **loader_kwargs)

        self.model, self.optim, self.dl = self.accelerator.prepare(
            self.model,
            self.optim,
            self.dl,
        )

        self.dl_iter = cycle(self.dl)
        self.lr = lr
        self.wd = wd
        self.start_time = time.time()

    def save_debug_input_batch(self, batch, depth1, z_rgb_features, z_depth_feature):
        if not self.is_main:
            return
        if self.debug_saved_input:
            return
        if not self.debug_save_input:
            return

        debug_dir = self.results_folder / "debug_inputs"
        debug_dir.mkdir(parents=True, exist_ok=True)

        n = min(self.debug_num_samples, depth1.shape[0])

        debug_data = {
            "depth1": depth1[:n].detach().cpu(),
            "z_rgb_features": z_rgb_features[:n].detach().cpu(),
            "z_depth_feature": z_depth_feature[:n].detach().cpu(),
        }

        if "id" in batch:
            debug_data["id"] = list(batch["id"][:n])

        if "depth1_path" in batch:
            debug_data["depth1_path"] = list(batch["depth1_path"][:n])

        torch.save(debug_data, debug_dir / "debug_batch.pt")

        depth_vis = depth1[:n].detach().cpu().float()
        if depth_vis.shape[1] == 3:
            depth_vis = depth_vis[:, :1]

        grid = make_grid(depth_vis, nrow=min(n, 4), normalize=True, value_range=(0, 1))
        save_image(grid, debug_dir / "depth1_grid.png")

        meta = []
        for i in range(n):
            z_rgb_feat = z_rgb_features[i].detach().cpu().float()
            z_depth_feat = z_depth_feature[i].detach().cpu().float()

            item = {
                "z_rgb_features_shape": list(z_rgb_feat.shape),
                "z_rgb_features_mean": float(z_rgb_feat.mean()),
                "z_rgb_features_std": float(z_rgb_feat.std()),
                "z_depth_feature_shape": list(z_depth_feat.shape),
                "z_depth_feature_mean": float(z_depth_feat.mean()),
                "z_depth_feature_std": float(z_depth_feat.std()),
            }

            if "id" in batch:
                item["id"] = str(batch["id"][i])

            if "depth1_path" in batch:
                item["depth1_path"] = str(batch["depth1_path"][i])

            meta.append(item)

        with open(debug_dir / "debug_batch.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        self.print(f"Saved debug input batch to {debug_dir}")
        self.debug_saved_input = True

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def print(self, msg):
        self.accelerator.print(msg)

    def save(self, path, extra: Optional[Dict[str, Any]] = None):
        if not self.is_local_main:
            return

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        pkg = {
            "model": unwrapped_model.state_dict(),
            "optim": self.optim.state_dict(),
            "steps": int(self.steps.item()),
            "best_loss": float(self.best_loss),
            "best_mse": float(self.best_mse),
            "best_cosine": float(self.best_cosine),
            "best_step": int(self.best_step),
            "best_metric": self.best_metric,
        }

        if extra is not None:
            pkg.update(extra)

        torch.save(pkg, path)

    def load(self, path, strict: bool = False):
        path = Path(path)
        assert path.exists(), f"Checkpoint not found: {path}"

        pkg = torch.load(path, map_location="cpu")
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.load_state_dict(pkg["model"], strict=strict)

        if "optim" in pkg:
            self.optim.load_state_dict(pkg["optim"])

        if "steps" in pkg:
            self.steps[...] = int(pkg["steps"])

        if "best_loss" in pkg:
            self.best_loss = float(pkg["best_loss"])
        if "best_mse" in pkg:
            self.best_mse = float(pkg["best_mse"])
        if "best_cosine" in pkg:
            self.best_cosine = float(pkg["best_cosine"])
        if "best_step" in pkg:
            self.best_step = int(pkg["best_step"])

    def _move_batch_to_device(self, batch):
        required_keys = ["depth1", "z_rgb_features", "z_depth_feature"]
        for key in required_keys:
            if key not in batch:
                raise KeyError(
                    f"Missing key '{key}' in batch. "
                    f"Expected keys: {required_keys}. Got: {list(batch.keys())}"
                )

        depth1 = batch["depth1"].to(self.device, non_blocking=True).float()
        z_rgb_features = batch["z_rgb_features"].to(self.device, non_blocking=True).float()
        z_depth_feature = batch["z_depth_feature"].to(self.device, non_blocking=True).float()

        return depth1, z_rgb_features, z_depth_feature

    def _maybe_save_best(self, logs: Dict[str, Any]):
        if not self.save_best or not self.is_main:
            return

        step = int(logs["step"])
        loss = float(logs["model4/loss"])
        mse = float(logs["model4/feature_mse_loss"])
        cosine = float(logs["model4/feature_cosine_loss"])

        metric_value = {
            "loss": loss,
            "mse": mse,
            "cosine": cosine,
        }[self.best_metric]

        best_value = {
            "loss": self.best_loss,
            "mse": self.best_mse,
            "cosine": self.best_cosine,
        }[self.best_metric]

        if metric_value >= best_value:
            return

        self.best_loss = min(self.best_loss, loss)
        self.best_mse = min(self.best_mse, mse)
        self.best_cosine = min(self.best_cosine, cosine)
        self.best_step = step

        best_path = self.results_folder / "model4.best.pt"
        self.save(
            best_path,
            extra={
                "checkpoint_type": "best",
                "current_step": step,
                "current_loss": loss,
                "current_mse": mse,
                "current_cosine": cosine,
                "best_metric_value": metric_value,
            },
        )

        meta = {
            "best_step": self.best_step,
            "best_metric": self.best_metric,
            "best_loss": self.best_loss,
            "best_mse": self.best_mse,
            "best_cosine": self.best_cosine,
            "current_loss": loss,
            "current_mse": mse,
            "current_cosine": cosine,
            "path": str(best_path),
        }

        with open(self.results_folder / "model4.best.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        self.print(
            f"{step}: saved best model to {best_path} | "
            f"loss={loss:.6f} | mse={mse:.6f} | cosine={cosine:.6f}"
        )

    def train_step(self):
        self.model.train()

        steps = int(self.steps.item())
        total_loss = 0.0
        total_mse = 0.0
        total_cosine = 0.0

        for _ in range(self.grad_accum_every):
            batch = next(self.dl_iter)
            depth1, z_rgb_features, z_depth_feature = self._move_batch_to_device(batch)

            self.save_debug_input_batch(batch, depth1, z_rgb_features, z_depth_feature)

            loss, model_logs, pred_z_depth_feature = self.model(
                depth1=depth1,
                z_rgb_features=z_rgb_features,
                z_depth_feature=z_depth_feature,
            )

            loss_for_backward = loss / self.grad_accum_every
            self.accelerator.backward(loss_for_backward)

            total_loss += detach_item(loss) / self.grad_accum_every
            total_mse += detach_item(model_logs["feature_mse_loss"]) / self.grad_accum_every
            total_cosine += detach_item(model_logs["feature_cosine_loss"]) / self.grad_accum_every

        if self.max_grad_norm is not None:
            self.accelerator.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm,
            )

        self.optim.step()
        self.optim.zero_grad()

        logs = {
            "model4/loss": total_loss,
            "model4/feature_mse_loss": total_mse,
            "model4/feature_cosine_loss": total_cosine,
            "model4/lr": self.lr,
            "model4/best_loss": self.best_loss,
            "model4/best_mse": self.best_mse,
            "model4/best_cosine": self.best_cosine,
            "model4/best_step": self.best_step,
            "step": steps,
        }

        # self._maybe_save_best(logs)

        if self.is_main and self.use_wandb and steps % self.log_every == 0:
            wandb.log(logs)

        if self.is_main and self.save_model_every > 0 and steps % self.save_model_every == 0:
            ckpt_path = self.results_folder / f"model4.{steps}.pt"
            self.save(ckpt_path)
            self.print(f"{steps}: saved model to {ckpt_path}")

        self.steps += 1
        return logs

    def train(self, log_fn: Callable = noop):
        if self.is_main and self.use_wandb:
            wandb.init(
                project=self.wandb_project,
                name=self.wandb_run_name,
                config={
                    "learning_rate": self.lr,
                    "weight_decay": self.wd,
                    "batch_size": self.batch_size,
                    "num_train_steps": self.num_train_steps,
                    "grad_accum_every": self.grad_accum_every,
                    "max_grad_norm": self.max_grad_norm,
                    "input_type": "depth1+z_rgb_features",
                    "target_type": "z_depth_feature",
                    "uses_z_depth_indices": False,
                    "model": "model4",
                    "best_metric": self.best_metric,
                },
            )

        while int(self.steps.item()) < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

            if self.is_main and int(self.steps.item()) % self.log_every == 0:
                current_step = int(self.steps.item())
                elapsed = time.time() - self.start_time

                sec_per_step = elapsed / max(current_step, 1)
                remaining_steps = self.num_train_steps - current_step
                eta_sec = remaining_steps * sec_per_step

                elapsed_hours = elapsed / 3600
                eta_hours = eta_sec / 3600

                self.print(
                    f"step {current_step}/{self.num_train_steps} | "
                    f"loss {logs['model4/loss']:.6f} | "
                    f"mse {logs['model4/feature_mse_loss']:.6f} | "
                    f"cosine {logs['model4/feature_cosine_loss']:.6f} | "
                    f"best_step {self.best_step} | "
                    f"{sec_per_step:.3f}s/step | "
                    f"elapsed {elapsed_hours:.2f}h | "
                    f"ETA {eta_hours:.2f}h"
                )

        if self.save_final and self.is_main:
            final_step = int(self.steps.item())
            final_ckpt_path = self.results_folder / f"model4.{final_step}.pt"
            self.save(
                final_ckpt_path,
                extra={
                    "checkpoint_type": "final",
                    "current_step": final_step,
                },
            )
            self.print(f"{final_step}: saved final model to {final_ckpt_path}")

        self.print("Model 4 training complete")

        if self.is_main and self.use_wandb:
            wandb.finish()
