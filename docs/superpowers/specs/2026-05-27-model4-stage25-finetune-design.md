# Model 4 Stage-25 Fine-Tune Design

## Goal

Add a local fine-tuning path for the stage-25 `model4` feature predictor in `laq/`, using the implementation that the upstream author pushed to the `lapa-depth25` branch.

The target model is the continuous feature predictor:

- input: `depth1 + z_rgb_features`
- target: `z_depth_feature`

This is the stage-25 `model4` path, not the older LAQ `case4` reconstruction model and not the LAPA JAX finetune stack.

## Scope

In scope:

- add the missing stage-25 `model4` module files under `laq/laq_model/`
- add a trainable entrypoint for the model4 feature predictor
- add a thin shell wrapper for the entrypoint
- support loading pretrained weights by default
- preserve the upstream training contract and checkpoint format

Out of scope:

- changing the model architecture
- changing the dataset schema
- adding a new LAPA JAX finetune path
- adding training-time support for `z_depth_indices`

## Reference Behavior

The upstream branch defines these core pieces:

- `laq/laq_model/latent_action_quantization_stage25_feature_model4.py`
- `laq/laq_model/laq_stage25_trainer_feature_model4.py`
- `laq/laq_model/data_stage25_feature_model4.py`
- `laq/train_stage25_sthv2_feature_model4_corl_da.py`

The model contract is:

- `depth1: [B, C, H, W]`
- `z_rgb_features: [B, feature_dim]`
- `z_depth_feature: [B, D]` or `[B, L, D]`
- output loss = `feature_mse_loss + cosine_loss`

The trainer contract is:

- checkpoint save format uses `model4.<step>.pt`
- best checkpoint uses `model4.best.pt`
- optional resume loads the stored `model` state dict

## Proposed Implementation

### 1. Add the stage-25 model4 module files

Add the missing local LAQ modules:

- `laq/laq_model/latent_action_quantization_stage25_feature_model4.py`
- `laq/laq_model/laq_stage25_trainer_feature_model4.py`
- `laq/laq_model/data_stage25_feature_model4.py`

These will mirror the upstream pushed implementation and expose the same class names:

- `LatentActionQuantizationStage25Model4`
- `LAQStage25TrainerModel4`
- `Stage252DatasetModel4`

### 2. Add a finetune entrypoint

Add a new executable Python launcher, modeled after the upstream script, with CLI args for:

- pretrained checkpoint path
- optional resume checkpoint path
- dataset manifest paths
- training hyperparameters
- output folder
- logging knobs

The launcher will:

- build `Stage252DatasetModel4`
- inspect a sample to infer `z_rgb_feature_dim` and `z_depth_feature_dim`
- create `LatentActionQuantizationStage25Model4`
- load the pretrained checkpoint
- create `LAQStage25TrainerModel4`
- start training

### 3. Add a shell wrapper

Add a small shell script that sets sane defaults and invokes the Python entrypoint.

The wrapper should make pretrained fine-tuning the default path, not scratch training.

## Error Handling

The new launcher should fail fast when:

- the pretrained checkpoint is missing
- any manifest path is missing
- the dataset sample cannot be parsed
- the feature dimensions do not match the model configuration

The dataset loader should preserve the current alignment checks and raise clear errors on sample count mismatch or malformed manifests.

## Validation

Validation will be lightweight and local:

- `bash -n` on the shell wrapper
- `python3 -m py_compile` on the new Python files
- a minimal import/shape smoke check against one dataset sample

If the script is wired correctly, the code should reach the trainer construction path without requiring changes to the model architecture.

## Success Criteria

This is done when:

- the stage-25 model4 files exist locally
- the fine-tune entrypoint can load a pretrained checkpoint
- the wrapper launches the trainer with the expected dataset contract
- the implementation is still compatible with the upstream checkpoint format and save layout

