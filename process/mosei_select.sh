#!/bin/bash
#SBATCH --account=msoleyma_1026
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=8:00:00
#SBATCH --job-name=MOSEI
#SBATCH --output=slurm/marlin/MOSEI_%j.out

python mosei_select.py