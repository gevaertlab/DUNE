#!/usr/bin/bash
#SBATCH --job-name=UKB1
#SBATCH --output=logs/dump.out
#SBATCH --error=logs/tqdmUKB1.out
#SBATCH --time=48:00:00
#SBATCH --partition=normal
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tbarba@stanford.edu

# INIT
source ~/.bashrc
cd ~/projects/MultiModalBrainSurvival


# SCRIPT
python src/preprocessing/preprocess_dataset.py