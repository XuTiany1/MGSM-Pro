#!/usr/bin/env bash
#SBATCH --partition=long
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=240GB
#SBATCH --time=1:00:00
#SBATCH --cpus-per-gpu=16
#SBATCH --output=sbatch_out/symbolic.%A.out
#SBATCH --job-name=symbolic

module load anaconda/3
conda activate reasoners
module load cuda/11.8

mkdir -p sbatch_out

cd /home/mila/x/xut/github/evaluation-pipeline/src

CONFIG_PATH=$1

python -m solve_dataset.solve_data --config $CONFIG_PATH




