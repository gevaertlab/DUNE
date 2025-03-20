#!/usr/bin/bash
#SBATCH --job-name=vestseg
#SBATCH --output=logs/vestseg.out
#SBATCH --error=logs/dump.out
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