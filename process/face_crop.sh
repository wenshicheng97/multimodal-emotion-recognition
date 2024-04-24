#!/bin/bash
#SBATCH --account=msoleyma_1026
#SBATCH --partition=gpu
#SBATCH --cpus-per-gpu=10
#SBATCH --gpus=a100:1
#SBATCH --mem=20G
#SBATCH --time=8:00:00
#SBATCH --job-name=MOSEI_CROP
#SBATCH --output=slurm/marlin/MOSEI_%j.out

python -m face_crop