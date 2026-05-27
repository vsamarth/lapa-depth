#!/usr/bin/env bash
# ============================================================
# LAPA Depth: Training Script
# Run setup.sh first to prepare data and checkpoints
# ============================================================
set -eu
cd "$(dirname "$0")"

source .venv/bin/activate

if [ -f .lapa_data_root ]; then
    # shellcheck disable=SC1091
    source .lapa_data_root
fi

STEPS="${STEPS:-20000}"
MODALITY="${MODALITY:-vision,depth,action}"
MODEL_SIZE="${MODEL_SIZE:-7b}"
MESH="${MESH:-1,-1,1,1}"

if [ -n "${DATA_ROOT:-}" ]; then
    mkdir -p "$DATA_ROOT"
    DATA_ROOT="$(python3 -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$DATA_ROOT")"
else
    for candidate in "/data" "$HOME/lapa-data" "$PWD/.lapa-data"; do
        if [ -d "$candidate" ]; then
            DATA_ROOT="$(python3 -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$candidate")"
            break
        fi
    done
    if [ -z "${DATA_ROOT:-}" ]; then
        DATA_ROOT="$(python3 -c 'import os; print(os.path.abspath(".lapa_data"))')"
        mkdir -p "$DATA_ROOT"
    fi
fi

OUT_DIR="${OUT_DIR:-$DATA_ROOT/libero_finetune}"
PROCESSED_JSONL="${PROCESSED_JSONL:-$OUT_DIR/processed.jsonl}"
ACTION_SCALE_CSV="${ACTION_SCALE_CSV:-$OUT_DIR/action_scale.csv}"
IMAGE_ABSOLUTE_PATH="${IMAGE_ABSOLUTE_PATH:-$OUT_DIR/}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PWD/outputs}"
LOGGER_OUTPUT_DIR="${LOGGER_OUTPUT_DIR:-$OUTPUT_ROOT/lapa_finetune/experiment}"

mkdir -p "$OUTPUT_ROOT"
mkdir -p "$LOGGER_OUTPUT_DIR"

if [ ! -f "$PROCESSED_JSONL" ]; then
    echo "ERROR: Data not found. Run 'bash setup.sh' first."
    echo "Expected processed data at: $PROCESSED_JSONL"
    exit 1
fi

# Get action vocab size
ACTION_VOCAB_SIZE=$(python3 -c "
import csv
with open('$ACTION_SCALE_CSV') as f:
    rows = list(csv.reader(f))
    print(max(len([v for v in r if v.strip()]) for r in rows[1:]) - 1)")

echo "=== LAPA Fine-tuning ==="
echo "Modality: $MODALITY | Steps: $STEPS | Mesh: $MESH | Action vocab: $ACTION_VOCAB_SIZE"
echo "Data root: $DATA_ROOT | Output dir: $LOGGER_OUTPUT_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true
echo ""

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export NCCL_P2P_DISABLE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION="0.95"

if [ "$MODALITY" = "vision,depth,action" ]; then
python3 -u -m latent_pretraining.train \
    --modality='vision,depth,action' \
    --mesh_dim="$MESH" \
    --dtype='bf16' \
    --total_steps="$STEPS" \
    --log_freq=10 --eval_steps=0 --save_model_freq=0 --eval_log_freq=50 \
    --save_milestone_freq="$STEPS" \
    --load_llama_config="$MODEL_SIZE" \
    --load_checkpoint="params::$(pwd)/checkpoints/params" \
    --update_llama_config="dict(action_vocab_size=${ACTION_VOCAB_SIZE},depth_vocab_size=8448,theta=50000000,max_sequence_length=2048,use_flash_attention=True,scan_attention=True,scan_query_chunk_size=512,scan_key_chunk_size=1024,remat_attention='nothing_saveable',scan_mlp=True,scan_mlp_chunk_size=8192,remat_mlp='nothing_saveable',remat_block='nothing_saveable',scan_layers=True)" \
    --tokenizer.vocab_file="$(pwd)/checkpoints/tokenizer.model" \
    --optimizer.type='adamw' --llama.action_vocab_size="$ACTION_VOCAB_SIZE" \
    --optimizer.accumulate_gradient_steps=1 --optimizer.adamw_optimizer.weight_decay=0 \
    --optimizer.adamw_optimizer.lr=2e-5 --optimizer.adamw_optimizer.end_lr=2e-5 \
    --optimizer.adamw_optimizer.lr_warmup_steps=0 --optimizer.adamw_optimizer.lr_decay_steps=100 \
    --use_data_sharded_loader=True \
    --train_dataset.type='json_vision_depth_action' \
    --train_dataset.vision_depth_action_processor.fields_from_example='fields' \
    --train_dataset.vision_depth_action_processor.n_tokens_per_action=7 \
    --train_dataset.vision_depth_action_processor.img_aug=True \
    --train_dataset.vision_depth_action_processor.vqgan_checkpoint_path="$(pwd)/checkpoints/vqgan" \
    --train_dataset.vision_depth_action_processor.image_absolute_path="$IMAGE_ABSOLUTE_PATH" \
    --train_dataset.vision_depth_action_processor.depth_absolute_path="$IMAGE_ABSOLUTE_PATH" \
    --train_dataset.vision_depth_action_processor.max_n_frames=1 \
    --train_dataset.vision_depth_action_processor.max_vq_tokens=64 \
    --train_dataset.json_vision_depth_action_dataset.mode='pad' \
    --train_dataset.json_vision_depth_action_dataset.path="$PROCESSED_JSONL" \
    --train_dataset.json_vision_depth_action_dataset.seq_length=192 \
    --train_dataset.json_vision_depth_action_dataset.batch_size=4 \
    --train_dataset.json_vision_depth_action_dataset.tokenizer_processes=1 \
    --train_dataset.json_vision_depth_action_dataset.tokenizer_parallel_chunk_size=128 \
    --train_dataset.json_vision_depth_action_dataset.tokenizer_parallel_batch_size=128 \
    --train_dataset.json_vision_depth_action_dataset.use_data_sharded_loader=True \
    --checkpointer.save_optimizer_state=False --autoresume=False \
    --logger.append_uuid=False --logger.online=False \
    --logger.output_dir="$LOGGER_OUTPUT_DIR"
else
python3 -u -m latent_pretraining.train \
    --modality='vision,action,delta' \
    --mesh_dim="$MESH" \
    --dtype='bf16' \
    --total_steps="$STEPS" \
    --log_freq=10 --eval_steps=0 --save_model_freq=0 --eval_log_freq=50 \
    --save_milestone_freq="$STEPS" \
    --load_llama_config="$MODEL_SIZE" \
    --load_checkpoint="params::$(pwd)/checkpoints/params" \
    --update_llama_config="dict(action_vocab_size=${ACTION_VOCAB_SIZE},delta_vocab_size=8,theta=50000000,max_sequence_length=2048,use_flash_attention=True,scan_attention=True,scan_query_chunk_size=512,scan_key_chunk_size=1024,remat_attention='nothing_saveable',scan_mlp=True,scan_mlp_chunk_size=8192,remat_mlp='nothing_saveable',remat_block='nothing_saveable',scan_layers=True)" \
    --tokenizer.vocab_file="$(pwd)/checkpoints/tokenizer.model" \
    --optimizer.type='adamw' --llama.action_vocab_size="$ACTION_VOCAB_SIZE" --llama.delta_vocab_size=8 \
    --optimizer.accumulate_gradient_steps=1 --optimizer.adamw_optimizer.weight_decay=0 \
    --optimizer.adamw_optimizer.lr=2e-5 --optimizer.adamw_optimizer.end_lr=2e-5 \
    --optimizer.adamw_optimizer.lr_warmup_steps=0 --optimizer.adamw_optimizer.lr_decay_steps=100 \
    --use_data_sharded_loader=True \
    --train_dataset.type='json_vision_delta_action' \
    --train_dataset.delta_vision_action_processor.fields_from_example='fields' \
    --train_dataset.delta_vision_action_processor.n_tokens_per_action=7 \
    --train_dataset.delta_vision_action_processor.n_tokens_per_delta=4 \
    --train_dataset.delta_vision_action_processor.img_aug=True \
    --train_dataset.delta_vision_action_processor.vqgan_checkpoint_path="$(pwd)/checkpoints/vqgan" \
    --train_dataset.delta_vision_action_processor.image_absolute_path="$IMAGE_ABSOLUTE_PATH" \
    --train_dataset.delta_vision_action_processor.max_n_frames=1 \
    --train_dataset.json_delta_action_dataset.mode='pad' \
    --train_dataset.json_delta_action_dataset.path="$PROCESSED_JSONL" \
    --train_dataset.json_delta_action_dataset.seq_length=320 \
    --train_dataset.json_delta_action_dataset.batch_size=4 \
    --train_dataset.json_delta_action_dataset.tokenizer_processes=1 \
    --train_dataset.json_delta_action_dataset.tokenizer_parallel_chunk_size=128 \
    --train_dataset.json_delta_action_dataset.tokenizer_parallel_batch_size=128 \
    --train_dataset.json_delta_action_dataset.use_data_sharded_loader=True \
    --checkpointer.save_optimizer_state=False --autoresume=False \
    --logger.append_uuid=False --logger.online=False \
    --logger.output_dir="$LOGGER_OUTPUT_DIR"
fi

echo ""
echo "=== TRAINING COMPLETE ==="
echo "Checkpoints: $LOGGER_OUTPUT_DIR/streaming_checkpoints/"
