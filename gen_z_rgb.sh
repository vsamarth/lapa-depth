env -u LD_LIBRARY_PATH python -m latent_pretraining.inference \
  --dataset_root /media/do/data1/philo/lapa/something-something-v2/ssv2-mini-2k-5 \
  --frames_dirname frames_train \
  --labels_dirname labels \
  --output_dirname z_rgb_indices_stage2_train \
  --save_debug_json


