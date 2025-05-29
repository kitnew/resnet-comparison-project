#!/bin/bash
#SBATCH --job-name=resnet101
#SBATCH --output=results/resnet101/slurm_%j.out
#SBATCH --error=results/resnet101/slurm_%j.err
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00

source .venv/bin/activate
python train.py --model resnet101
