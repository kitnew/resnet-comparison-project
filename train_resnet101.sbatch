#!/bin/bash
#SBATCH --job-name=resnet101
#SBATCH --output=results/resnet101/slurm_%j.out
#SBATCH --error=results/resnet101/slurm_%j.err
#SBATCH --open-mode=append
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00

module load cuda/12.3
module load python/3.12

source .venv/bin/activate
python src/train.py --model resnet101 --mode train --visualize --save-model
