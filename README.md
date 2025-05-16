# PPGR Transformer v1

PPGR Transformer is an implementation of a transformer based forecasting model for predicting postprandial glucose response from meal logs and user metadata. The project bundles data preprocessing utilities, a flexible PyTorch Lightning model, training scripts, and tools for visualizing forecasts.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Repository Layout](#repository-layout)
- [Cluster Usage](#cluster-usage)
- [Next Steps](#next-steps)

## Features
- **Data Pipeline** – `dataset.py` prepares time‑series slices from raw CSV files, scales real values, encodes categoricals, and caches processed datasets for reuse.
- **Forecasting Model** – `models/forecast_model.py` implements a meal‑aware transformer decoder with rotary positional embeddings and custom loss functions.
- **Configurable Training** – `main.py` exposes all options from `ExperimentConfig` via CLI flags and integrates custom callbacks such as a OneCycle learning rate scheduler.
- **Visualization Tools** – `plot_helpers.py` generates forecast plots, attention maps, and incremental AUC correlations to aid model analysis.
- **Hyperparameter Sweeps** – `sweeps/hyperparams-v0.yaml` provides a W&B Bayesian sweep template for exploring model settings.

## Installation

Install the required packages with pip (Python 3.8+ recommended):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install lightning torchmetrics
pip install pytorch-forecasting  # only required for NaNLabelEncoder
pip install scikit-learn loguru wandb p_tqdm matplotlib faker rich
```

## Quick Start
1. **Cache the dataset** using `create_cached_dataset` in `dataset.py`.
2. **Train the model**:

```bash
python main.py --dataset-version v0.5 --max-epochs 50 --learning-rate 1e-3
```

Use `--help` to view all available configuration flags. After training, evaluation results and example forecasts are saved automatically.

## Repository Layout
- `config.py` – Experiment configuration dataclass and helper for informative experiment names.
- `dataset.py` – Data loading, scaling/encoding, dataset creation, and caching.
- `models/` – Encoders, transformer blocks, and `MealGlucoseForecastModel`.
- `callbacks.py` – Custom Lightning callbacks (e.g., OneCycleLR scheduler).
- `main.py` – Command‑line entry point for training and evaluation.
- `plot_helpers.py` – Visualization utilities.
- `scripts/` – Example shell scripts, including an interactive GPU session helper.
- `sweeps/` – Sample W&B hyperparameter sweep configuration.

## Cluster Usage
The `run_slurm.sh` script demonstrates how to schedule a job on a SLURM cluster. For an interactive GPU session, use `scripts/run_interactive_gpu.sh`.

## Next Steps
- Dive into `dataset.py` to see how meal‑anchored time series are assembled.
- Inspect the encoders and transformer architecture under `models/`.
- Utilize the plotting helpers to visualize attention weights and forecast quality.


# Author
Sharada Mohanty <sharada.mohanty@epfl.ch>