#!/usr/bin/env bash
set -euo pipefail

export SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"
cd "$PROJECT_DIR"
export PYTHONPATH="$PYTHONPATH:$PROJECT_DIR"

export absolute_path="${absolute_path:-$PROJECT_DIR}"
export dataset_path="${dataset_path:-$absolute_path/data/real_finetune.jsonl}"
export data_root="${data_root:-$absolute_path/data}"
export output_dir="${output_dir:-$absolute_path/outputs/finetune_real_depth}"
export load_checkpoint="${load_checkpoint:-params::$absolute_path/lapa_checkpoints/params}"
export model4_checkpoint="${model4_checkpoint:-}"
export vqgan_checkpoint="${vqgan_checkpoint:-$absolute_path/lapa_checkpoints/vqgan}"
export vocab_file="${vocab_file:-$absolute_path/lapa_checkpoints/tokenizer.model}"

export batch_size="${batch_size:-8}"
export total_steps="${total_steps:-2000}"
export learning_rate="${learning_rate:-2e-5}"
export save_every="${save_every:-500}"
export log_every="${log_every:-10}"
export num_workers="${num_workers:-4}"

if [[ -z "$model4_checkpoint" ]]; then
  echo "ERROR: set model4_checkpoint to the pretrained/frozen stage-2.5 model4 checkpoint."
  exit 1
fi

python3 -u -m latent_pretraining.finetune_stage3_model4 \
  --dataset_path "$dataset_path" \
  --data_root "$data_root" \
  --output_dir "$output_dir" \
  --load_checkpoint "$load_checkpoint" \
  --model4_checkpoint "$model4_checkpoint" \
  --vqgan_checkpoint "$vqgan_checkpoint" \
  --vocab_file "$vocab_file" \
  --batch_size "$batch_size" \
  --total_steps "$total_steps" \
  --learning_rate "$learning_rate" \
  --save_every "$save_every" \
  --log_every "$log_every" \
  --num_workers "$num_workers"
