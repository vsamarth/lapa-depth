# LAQ Case4 Finetune Wrapper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a clean, configurable LAQ case4 finetuning wrapper that can launch the existing `laq/train_sthv2_case4.py` pipeline with custom data paths, output paths, and optional resume checkpoint support.

**Architecture:** Keep the core LAQ/case4 model and trainer unchanged, but make the training entrypoint configurable through explicit CLI arguments and a thin shell wrapper. The Python entrypoint should import the case4 modules directly instead of relying on commented `__init__.py` exports, then assemble the dataset, model, trainer, and optional resume checkpoint from parsed arguments. The shell wrapper should provide sane defaults and mirror the style of the repo’s existing finetune launch scripts.

**Tech Stack:** Python 3.10+, PyTorch, `accelerate`, `laq_model` case4 modules, Bash.

---

### Task 1: Make the case4 training entrypoint configurable and import-safe

**Files:**
- Modify: `laq/train_sthv2_case4.py`

- [ ] **Step 1: Update the imports and add argument parsing**

Replace the current top-level imports with direct case4 imports and add a small `argparse` config block so the script can be driven from a wrapper:

```python
import argparse

from laq_model.latent_action_quantization_case4 import LatentActionQuantization
from laq_model.laq_trainer_case4 import LAQTrainer
from laq_model.data_org import ImageVideoDatasetDepth
```

Add CLI options with defaults matching the current hardcoded values:

```python
parser = argparse.ArgumentParser()
parser.add_argument("--rgb_path", default="/media/do/data1/philo/lapa/something-something-v2/ssv2-mini-2k-5/frames_train")
parser.add_argument("--depth_path", default="/media/do/data1/philo/lapa/something-something-v2/ssv2-mini-2k-5/depth_train")
parser.add_argument("--z_rgb_path", default="/media/do/data1/philo/lapa/something-something-v2/ssv2-mini-2k-5/z_rgb_indices_stage2_train")
parser.add_argument("--offsets", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--num_train_steps", type=int, default=10000)
parser.add_argument("--results_folder", default="results_case4")
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--save_model_every", type=int, default=500)
parser.add_argument("--save_results_every", type=int, default=200)
parser.add_argument("--modality", choices=("rgb", "depth", "both"), default="both")
parser.add_argument("--resume_checkpoint", default="")
args = parser.parse_args()
```

- [ ] **Step 2: Wire the parsed values into the dataset, model, trainer, and optional resume path**

Instantiate the existing case4 pipeline from the parsed arguments:

```python
laq = LatentActionQuantization(
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
).cuda()

trainer = LAQTrainer(
    laq,
    folder=args.rgb_path,
    depth_folder=args.depth_path,
    z_rgb_folder=args.z_rgb_path,
    offsets=args.offsets,
    batch_size=args.batch_size,
    grad_accum_every=1,
    train_on_images=False,
    use_ema=False,
    num_train_steps=args.num_train_steps,
    results_folder=args.results_folder,
    lr=args.lr,
    save_model_every=args.save_model_every,
    save_results_every=args.save_results_every,
    modality=args.modality,
)

if args.resume_checkpoint:
    trainer.load(args.resume_checkpoint)

trainer.train()
```

This preserves the existing model shape while making the script usable as a reusable launch target instead of a one-off hardcoded demo.

---

### Task 2: Add a shell wrapper that mirrors the repo’s finetune launch style

**Files:**
- Create: `laq/train_case4_finetune.sh`

- [ ] **Step 1: Create a thin launcher with sensible defaults**

Model it after the existing `laq/train_ssv2_case4.sh`, but make it configurable through environment variables and safer for repeated runs:

```bash
#!/usr/bin/env bash
set -euo pipefail

export SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"
cd "$PROJECT_DIR"
export PYTHONPATH="$PYTHONPATH:$PROJECT_DIR"

RGB_PATH="${RGB_PATH:-/media/do/data1/philo/lapa/something-something-v2/ssv2-mini-2k-5/frames_train}"
DEPTH_PATH="${DEPTH_PATH:-/media/do/data1/philo/lapa/something-something-v2/ssv2-mini-2k-5/depth_train}"
Z_RGB_PATH="${Z_RGB_PATH:-/media/do/data1/philo/lapa/something-something-v2/ssv2-mini-2k-5/z_rgb_indices_stage2_train}"
RESULTS_FOLDER="${RESULTS_FOLDER:-results_case4}"
MODALITY="${MODALITY:-both}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

accelerate launch train_sthv2_case4.py \
  --rgb_path "$RGB_PATH" \
  --depth_path "$DEPTH_PATH" \
  --z_rgb_path "$Z_RGB_PATH" \
  --results_folder "$RESULTS_FOLDER" \
  --modality "$MODALITY" \
  ${RESUME_CHECKPOINT:+--resume_checkpoint "$RESUME_CHECKPOINT"}
```

- [ ] **Step 2: Make the wrapper executable and keep it consistent with existing scripts**

Ensure the file uses the same style as the current LAQ scripts, including `accelerate launch` and no hidden assumptions about the dataset root. The wrapper should not hardcode a single machine path; it should only provide defaults and let environment variables override them.

---

### Task 3: Verify the wrapper is syntactically correct and launchable

**Files:**
- Modify: `laq/train_sthv2_case4.py`
- Create/Modify: `laq/train_case4_finetune.sh`

- [ ] **Step 1: Check Bash syntax**

Run:

```bash
bash -n laq/train_case4_finetune.sh
```

Expected: no output, exit code `0`.

- [ ] **Step 2: Check Python syntax**

Run:

```bash
python -m py_compile laq/train_sthv2_case4.py laq/laq_model/latent_action_quantization_case4.py laq/laq_model/laq_trainer_case4.py laq/laq_model/data_org.py
```

Expected: no output, exit code `0`.

- [ ] **Step 3: Smoke the wrapper command shape**

Run:

```bash
RGB_PATH=/tmp/rgb DEPTH_PATH=/tmp/depth Z_RGB_PATH=/tmp/zrgb RESULTS_FOLDER=/tmp/results MODALITY=both bash -x laq/train_case4_finetune.sh
```

Expected: the script expands the `accelerate launch train_sthv2_case4.py ...` command with the overridden paths. If the data paths are empty, the process should fail because of missing files, not because of argument parsing or shell syntax.

---

### Coverage Check

- The plan covers the LAQ/case4 training path by fixing the import wiring, parameterizing the entrypoint, and adding a shell launcher.
- The plan does not change the core LAQ model architecture or trainer logic.
- The plan includes a resume checkpoint path so the wrapper can be used for actual finetuning, not only scratch training.

