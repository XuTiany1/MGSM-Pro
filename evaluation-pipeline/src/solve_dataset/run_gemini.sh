#!/usr/bin/env bash
#SBATCH --partition=unkillable
#SBATCH --mem=16GB
#SBATCH --time 11:00:00
#SBATCH --cpus-per-task=2
#SBATCH --output=sbatch_out/symbolic.%A.out
#SBATCH --job-name=symbolic_cpu

module load anaconda/3
conda activate reasoners
module load cuda/11.8

mkdir -p sbatch_out

cd /home/mila/x/xut/github/evaluation-pipeline/src

CONFIG_PATH=$1

python -m solve_dataset.solve_data_2_gpu --config $CONFIG_PATH

