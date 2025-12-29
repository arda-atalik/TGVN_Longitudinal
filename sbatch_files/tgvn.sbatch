#!/bin/bash
#SBATCH --partition=a100_long,radiology
#SBATCH --nodes=4                     # Request 4 nodes
##SBATCH --exclude=a100-4038,a100-4039
#SBATCH --ntasks-per-node=2           # Two tasks per node
#SBATCH --cpus-per-task=4             # CPUs per task
#SBATCH --mem=160G                    # Memory per node
#SBATCH --gres=gpu:a100:2             # Request 2 A100 GPUs per node
#SBATCH --time=05-00:00:00            # 5 day training
#SBATCH --job-name=train
#SBATCH --output=logs/tgvn_%j.out
#SBATCH --error=logs/tgvn_%j.err

cd /gpfs/home/atalia01/BrainLongitudinal/tgvn
export PYTHONPATH=/gpfs/home/atalia01/BrainLongitudinal/tgvn:$PYTHONPATH
srun python /gpfs/home/atalia01/BrainLongitudinal/tgvn/scripts/main.py \
--config /gpfs/home/atalia01/BrainLongitudinal/tgvn/configs/tgvn.yaml