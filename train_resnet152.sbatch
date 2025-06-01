#!/bin/bash
#SBATCH --job-name=resnet152
#SBATCH --output=results/resnet152/slurm_%j.out
#SBATCH --error=results/resnet152/slurm_%j.err
#SBATCH --open-mode=append
#SBATCH --partition=dgx
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00

module load cuda/12.3
module load python/3.12

source .venv/bin/activate
python src/train.py --model resnet152 --mode train --visualize --save-model
