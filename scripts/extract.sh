#!/usr/bin/bash
#SBATCH --job-name=extr
#SBATCH --output=logs/extr%a.out
#SBATCH --error=logs/extr%a.out
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tbarba@stanford.edu
#SBATCH --array=0-1

# INIT
source ~/.bashrc
conda activate multimodal
nvidia-smi --list-gpus
cd ~/projects/MultiModalBrainSurvival

# VARIABLES
export mod=( "uVAE/uVAE_UKB_b1e-4_att" "AE/AE_UKB_segm_attn" )


# SCRIPT
CUDA_VISIBLE_DEVICES=0,1 \
    python src/autoencoder/feature_extraction.py  -c ${mod[SLURM_ARRAY_TASK_ID]}



