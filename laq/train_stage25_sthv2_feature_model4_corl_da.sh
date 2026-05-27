#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

args=(
  --pretrained_checkpoint "${PRETRAINED_CHECKPOINT:?set PRETRAINED_CHECKPOINT}"
  --z_depth_path "${Z_DEPTH_PATH:-/datasets/ssv2_libero_90/stage2_z_rgb_ssv2_libero90/z_depth_train_mixed.jsonl}"
  --z_rgb_feature_manifest "${Z_RGB_FEATURE_MANIFEST:-/datasets/ssv2_libero_90/stage2_z_rgb_ssv2_libero90/z_rgb_train_mixed_manifest.json}"
  --z_depth_feature_manifest "${Z_DEPTH_FEATURE_MANIFEST:-/datasets/ssv2_libero_90/stage2_z_rgb_ssv2_libero90/z_depth_train_mixed_manifest.json}"
  --results_folder "${RESULTS_FOLDER:-results_model4_finetune}"
  --num_train_steps "${NUM_TRAIN_STEPS:-65001}"
  --batch_size "${BATCH_SIZE:-128}"
  --lr "${LR:-1e-4}"
  --save_model_every "${SAVE_MODEL_EVERY:-5000}"
  --log_every "${LOG_EVERY:-100}"
  --num_workers "${NUM_WORKERS:-12}"
  --pin_memory
)

if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  args+=(--resume_checkpoint "${RESUME_CHECKPOINT}")
fi

python3 -u train_stage25_sthv2_feature_model4_corl_da.py "${args[@]}"
