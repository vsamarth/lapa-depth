# Model 4 Stage-25 Fine-Tune Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local fine-tune launcher for the stage-25 `model4` feature predictor so it can start from a pretrained checkpoint and continue training on the aligned depth/RGB feature dataset.

**Architecture:** Use the upstream `lapa-depth25` model4 implementation as the source of truth: a depth-frame encoder plus RGB-feature projection predicts `z_depth_feature`, and the trainer optimizes MSE plus cosine loss. The launcher will infer feature dimensions from one dataset sample, load pretrained weights explicitly, and hand off to the existing Accelerate-based trainer without changing the model architecture or dataset schema.

**Tech Stack:** Python, PyTorch, Accelerate, wandb, torchvision, cv2, einops, JSONL manifests, Bash

---

### Task 1: Restore the stage-25 model4 support modules

**Files:**
- Create: `laq/laq_model/laq_stage25_trainer_feature_model4.py`
- Create: `laq/laq_model/data_stage25_feature_model4.py`
- Modify: `laq/laq_model/latent_action_quantization_stage25_feature_model4.py` if any import or API mismatch shows up during smoke checks

- [ ] **Step 1: Write the module files from the upstream reference**

Copy the upstream `lapa-depth25` trainer and dataset modules into the local repo, preserving the public classes and behavior:

```python
from laq_model.laq_stage25_trainer_feature_model4 import LAQStage25TrainerModel4
from laq_model.data_stage25_feature_model4 import Stage252DatasetModel4
```

Keep the existing contracts intact:

```python
loss, logs, pred_z_depth_feature = model(
    depth1=depth1,
    z_rgb_features=z_rgb_features,
    z_depth_feature=z_depth_feature,
)
```

The dataset must return:

```python
{
    "depth1": FloatTensor,          # [C, H, W]
    "z_rgb_features": FloatTensor,  # [feature_dim]
    "z_depth_feature": FloatTensor, # [D] or [L, D]
    "id": str,
    "depth1_path": str,
}
```

- [ ] **Step 2: Run a pure syntax check on the new support files**

Run:

```bash
python3 -m py_compile \
  laq/laq_model/latent_action_quantization_stage25_feature_model4.py \
  laq/laq_model/laq_stage25_trainer_feature_model4.py \
  laq/laq_model/data_stage25_feature_model4.py
```

Expected: no output, exit code `0`.

- [ ] **Step 3: Commit the restored support modules**

```bash
git add laq/laq_model/latent_action_quantization_stage25_feature_model4.py \
        laq/laq_model/laq_stage25_trainer_feature_model4.py \
        laq/laq_model/data_stage25_feature_model4.py
git commit -m "feat: restore stage25 model4 support modules"
```

### Task 2: Add a pretrained finetune entrypoint

**Files:**
- Create: `laq/train_stage25_sthv2_feature_model4_corl_da.py`

- [ ] **Step 1: Write the failing launcher shape first**

The entrypoint should parse these CLI arguments and fail early if they are missing or invalid:

```python
parser.add_argument("--pretrained_checkpoint", required=True)
parser.add_argument("--resume_checkpoint", default=None)
parser.add_argument("--z_depth_path", default="/datasets/ssv2_libero_90/stage2_z_rgb_ssv2_libero90/z_depth_train_mixed.jsonl")
parser.add_argument("--z_rgb_feature_manifest", default="/datasets/ssv2_libero_90/stage2_z_rgb_ssv2_libero90/z_rgb_train_mixed_manifest.json")
parser.add_argument("--z_depth_feature_manifest", default="/datasets/ssv2_libero_90/stage2_z_rgb_ssv2_libero90/z_depth_train_mixed_manifest.json")
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
```

The launcher should:

1. build `Stage252DatasetModel4`
2. read `dataset[0]` and `dataset[-1]` to infer `z_rgb_feature_dim`, `z_depth_feature_dim`, and whether token-level prediction is needed
3. construct `LatentActionQuantizationStage25Model4(...)`
4. load `--pretrained_checkpoint`
5. optionally resume from `--resume_checkpoint`
6. construct `LAQStage25TrainerModel4(...)`
7. call `trainer.train()`

The launcher must use the local `laq_model` package import style:

```python
from laq_model.data_stage25_feature_model4 import Stage252DatasetModel4
from laq_model.laq_stage25_trainer_feature_model4 import LAQStage25TrainerModel4
from laq_model.latent_action_quantization_stage25_feature_model4 import LatentActionQuantizationStage25Model4
```

- [ ] **Step 2: Run the launcher syntax check**

Run:

```bash
python3 -m py_compile laq/train_stage25_sthv2_feature_model4_corl_da.py
python3 laq/train_stage25_sthv2_feature_model4_corl_da.py --help
```

Expected:

- `py_compile` succeeds
- `--help` shows the pretrained and resume checkpoint flags

- [ ] **Step 3: Commit the launcher**

```bash
git add laq/train_stage25_sthv2_feature_model4_corl_da.py
git commit -m "feat: add model4 finetune launcher"
```

### Task 3: Add a shell wrapper for the pretrained run

**Files:**
- Create: `laq/train_stage25_sthv2_feature_model4_corl_da.sh`

- [ ] **Step 1: Write the wrapper so it runs from the `laq/` directory**

The wrapper should be a small `bash` launcher that forwards the pretrained checkpoint and training paths to the Python entrypoint:

```bash
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 -u train_stage25_sthv2_feature_model4_corl_da.py \
  --pretrained_checkpoint "${PRETRAINED_CHECKPOINT:?set PRETRAINED_CHECKPOINT}" \
  --z_depth_path "${Z_DEPTH_PATH:-/datasets/ssv2_libero_90/stage2_z_rgb_ssv2_libero90/z_depth_train_mixed.jsonl}" \
  --z_rgb_feature_manifest "${Z_RGB_FEATURE_MANIFEST:-/datasets/ssv2_libero_90/stage2_z_rgb_ssv2_libero90/z_rgb_train_mixed_manifest.json}" \
  --z_depth_feature_manifest "${Z_DEPTH_FEATURE_MANIFEST:-/datasets/ssv2_libero_90/stage2_z_rgb_ssv2_libero90/z_depth_train_mixed_manifest.json}" \
  --results_folder "${RESULTS_FOLDER:-results_model4_finetune}"
```

Keep the wrapper pretrained-first: it should require `PRETRAINED_CHECKPOINT` and treat resume as optional only if the Python entrypoint exposes it.

- [ ] **Step 2: Lint the shell wrapper**

Run:

```bash
bash -n laq/train_stage25_sthv2_feature_model4_corl_da.sh
```

Expected: no output, exit code `0`.

- [ ] **Step 3: Commit the wrapper**

```bash
git add laq/train_stage25_sthv2_feature_model4_corl_da.sh
git commit -m "feat: add model4 finetune wrapper"
```

### Task 4: Smoke-test the local model and wiring

**Files:**
- Modify: only if the smoke test reveals a mismatch in `laq/laq_model/latent_action_quantization_stage25_feature_model4.py`, `laq/laq_model/laq_stage25_trainer_feature_model4.py`, or `laq/train_stage25_sthv2_feature_model4_corl_da.py`

- [ ] **Step 1: Run a dummy forward pass against the local model**

Use a tiny synthetic batch to validate the shape logic without needing real data:

```python
import torch
from laq_model.latent_action_quantization_stage25_feature_model4 import LatentActionQuantizationStage25Model4

model = LatentActionQuantizationStage25Model4(
    dim=64,
    image_size=32,
    patch_size=16,
    spatial_depth=1,
    heads=4,
    dim_head=16,
    z_rgb_feature_dim=8,
    z_depth_feature_dim=4,
)

depth1 = torch.randn(2, 3, 32, 32)
z_rgb_features = torch.randn(2, 8)
z_depth_feature = torch.randn(2, 4)

loss, logs, pred = model(
    depth1=depth1,
    z_rgb_features=z_rgb_features,
    z_depth_feature=z_depth_feature,
)

assert pred.shape == z_depth_feature.shape
assert torch.isfinite(loss)
assert "feature_mse_loss" in logs
assert "feature_cosine_loss" in logs
print("model4 smoke test passed")
```

- [ ] **Step 2: Run the smoke test and the syntax checks together**

Run:

```bash
python3 - <<'PY'
import torch
from laq_model.latent_action_quantization_stage25_feature_model4 import LatentActionQuantizationStage25Model4

model = LatentActionQuantizationStage25Model4(
    dim=64,
    image_size=32,
    patch_size=16,
    spatial_depth=1,
    heads=4,
    dim_head=16,
    z_rgb_feature_dim=8,
    z_depth_feature_dim=4,
)

depth1 = torch.randn(2, 3, 32, 32)
z_rgb_features = torch.randn(2, 8)
z_depth_feature = torch.randn(2, 4)

loss, logs, pred = model(
    depth1=depth1,
    z_rgb_features=z_rgb_features,
    z_depth_feature=z_depth_feature,
)

assert pred.shape == z_depth_feature.shape
assert torch.isfinite(loss)
assert "feature_mse_loss" in logs
assert "feature_cosine_loss" in logs
print("model4 smoke test passed")
PY

python3 -m py_compile \
  laq/laq_model/latent_action_quantization_stage25_feature_model4.py \
  laq/laq_model/laq_stage25_trainer_feature_model4.py \
  laq/laq_model/data_stage25_feature_model4.py \
  laq/train_stage25_sthv2_feature_model4_corl_da.py

bash -n laq/train_stage25_sthv2_feature_model4_corl_da.sh
```

Expected:

- the dummy forward pass prints `model4 smoke test passed`
- all syntax checks succeed

- [ ] **Step 3: Commit the verified finetune path**

```bash
git add laq/laq_model/latent_action_quantization_stage25_feature_model4.py \
        laq/laq_model/laq_stage25_trainer_feature_model4.py \
        laq/laq_model/data_stage25_feature_model4.py \
        laq/train_stage25_sthv2_feature_model4_corl_da.py \
        laq/train_stage25_sthv2_feature_model4_corl_da.sh
git commit -m "feat: add model4 finetune path"
```

