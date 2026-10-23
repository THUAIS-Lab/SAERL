# Teach the Way It Thinks: Guiding LLM Post-training Data Engineering with Sparse Autoencoders

**SAERL** uses sparse autoencoder representations for data selection and curriculum construction in LLM reinforcement learning.

## Installation

Use Python 3.12 on Linux with NVIDIA CUDA 12.8.

```bash
conda create -n saerl python=3.12 pip -y
conda activate saerl
bash scripts/install_train.sh
```

Prepare the model, SAE checkpoints, and training/validation data separately. Dependencies are listed in [requirements.txt](requirements.txt).

## Usage

Configure data and model paths in [scripts/common.sh](scripts/common.sh), or set the corresponding `SAERL_*` environment variables.

| Step | Entry point |
| --- | --- |
| Extract SAE-projected gradients | `bash scripts/extract.sh` |
| Select training samples | `bash scripts/select.sh` |
| Extract prompt/solution activations | `python scripts/extract_activations.py` |
| Prepare difficulty features | `python scripts/prepare_difficulty_features.py` |
| Fit the difficulty predictor | `python scripts/fit_difficulty.py` |
| Build the curriculum | `python scripts/build_curriculum.py` |
| Export training row indices | `python scripts/export_training_order.py` |
| Train with the curriculum | `bash scripts/sae.sh` |

Python scripts accept `--help` for required arguments. For curriculum training, set `SAERL_PROCESSED_FILE` to the exported indices and `SAERL_TRAIN_DATA` to the same Parquet used to export them. Use the same batch size for curriculum construction and training.

For standard GRPO training, run `bash scripts/grpo.sh`.

## Tests

```bash
python -m unittest discover -s tests
```

## License

[Apache-2.0](LICENSE). Third-party copyright notices are retained in the relevant source files.
