#!/usr/bin/env bash
#SBATCH --partition=long
#SBATCH --mem=16GB
#SBATCH --time=9:00:00
#SBATCH --cpus-per-task=2
#SBATCH --output=sbatch_out/symbolic.%A.out
#SBATCH --job-name=symbolic_cpu

module load anaconda/3
conda activate reasoners
module load cuda/11.8

mkdir -p sbatch_out

cd /home/mila/x/xut/github/evaluation-pipeline/src

CONFIG_PATH=$1

python -m solve_dataset.claude_manager_submit --config $CONFIG_PATH




