#!/bin/bash

# --mem=64G \  if you want to request specific memory for the whole run instead of per gpu

srun --gpus=1 \
     --mem-per-gpu=64G \
     --cpus-per-gpu=8 \
     --partition=h100 \
     --time=12:00:00 \
     --pty /bin/bash

# micromamba activate ppgr
# pip install jupyter 
# jupyter notebook --port 18888 --no-browser --ip=0.0.0.0
