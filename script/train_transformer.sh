#!/bin/bash
#SBATCH --account=jonmay_231
#SBATCH --partition=gpu
#SBATCH --cpus-per-gpu=8
#SBATCH --gpus=v100:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=MultimodalTransformer
#SBATCH --output=slurm/%j.out
#SBATCH --error=slurm/%j.err

python -m experiment.train_transformer
