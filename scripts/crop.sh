#!/usr/bin/bash
#SBATCH --job-name=crop
#SBATCH --output=logs/dump.out
#SBATCH --error=logs/crop.out
#SBATCH --time=48:00:00
#SBATCH --partition=normal
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tbarba@stanford.edu

# INIT
source ~/.bashrc
cd ~/projects/MultiModalBrainSurvival
conda activate multimodal

# SCRIPT
python src/preprocessing/crop_nifti.py