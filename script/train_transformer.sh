#!/bin/bash
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a6000:6
#SBATCH --mem=128G
#SBATCH --time=48:00:00

python -m module.tf_module
