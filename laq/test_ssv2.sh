
  python3 inference_sthv2.py \
    --input_file /datasets/something-something-v2/nips/depth_train.jsonl \
    --dist_number 1 \
    --codebook_size 8 \
    --laq_checkpoint /workspace/lapa/laq/results/laq/results/vae.10500.pt \
    --divider 1 \
    --window_size 30 \
    --code_seq_len 4 \
    --layer 8 \
    --repeat_depth_to_3ch 1 \
    --debug_save_dir outputs/debug_depth \
    --debug_num_samples 10 \
    --unshuffled_jsonl /datasets/something-something-v2/nips/z_depth_train.jsonl


