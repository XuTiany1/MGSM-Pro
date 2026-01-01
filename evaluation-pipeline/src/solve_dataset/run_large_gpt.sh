#!/usr/bin/env bash
#SBATCH --partition=short-unkillable
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=900GB
#SBATCH --time=3:00:00
#SBATCH --cpus-per-gpu=16
#SBATCH --output=sbatch_out/symbolic.%A.out
#SBATCH --job-name=symbolic-large

module load anaconda/3
conda activate oss
module load cuda/11.8

mkdir -p sbatch_out

cd /home/mila/x/xut/github/evaluation-pipeline/src

CONFIG_PATH=$1

python -m solve_dataset.solve_data_4_gpu --config $CONFIG_PATH




