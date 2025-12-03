#!/bin/bash
#SBATCH --partition=a100_long
#SBATCH --nodes=1                     # Request 1 node
#SBATCH --ntasks-per-node=4           # Four tasks per node
#SBATCH --cpus-per-task=4             # CPUs per task
#SBATCH --mem=160G                    # Memory per node
#SBATCH --gres=gpu:a100:4             # Request 4 A100 GPUs per node
#SBATCH --time=14-00:00:00            # 14 day training
#SBATCH --job-name=train
#SBATCH --output=logs/tgvn_%j.out
#SBATCH --error=logs/tgvn_%j.err

cd /gpfs/home/atalia01/BrainLongitudinal/tgvn
export PYTHONPATH=/gpfs/home/atalia01/BrainLongitudinal/tgvn:$PYTHONPATH
srun python /gpfs/home/atalia01/BrainLongitudinal/tgvn/scripts/main.py \
--config /gpfs/home/atalia01/BrainLongitudinal/tgvn/configs/tgvn.yaml