#!/bin/bash
#SBATCH --job-name=run_ppgr_transformer_v1
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=32G
#SBATCH --cpus-per-gpu=8
#SBATCH --partition=h100
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%j.out
#SBATCH --error=logs/slurm/%j.err

# Configuration variables

# Change to the directory where this script is located
cd /home/mohanty/food/ppgr-transformer-v1

# Activate environment (uncomment if needed)
source ~/.bashrc
micromamba activate ppgr

# Check that a script is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <script.py> [args...]"
    exit 1
fi

time "$@"

