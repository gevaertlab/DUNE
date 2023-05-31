#!/usr/bin/bash
#SBATCH --job-name=1mod
#SBATCH --output=logs/1mod%a.out
#SBATCH --error=logs/1mod%a.out
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=3
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tbarba@stanford.edu
#SBATCH --array=0-0


# INIT
source ~/.bashrc
conda activate multimodal
nvidia-smi --list-gpus
cd ~/projects/MultiModalBrainSurvival


# VARIABLES
export mod=( \
# "norm/1mod" \
"norm/1mod_uVAE_UKB" \
# "norm/2mod_uVAE_UKB" \
# "AE/AE_UKB" \
)   


# SCRIPT
echo -e "Processing model = ${mod[SLURM_ARRAY_TASK_ID]}"
CUDA_VISIBLE_DEVICES=0,1 \
    python src/autoencoder/train_ae.py -c ${mod[SLURM_ARRAY_TASK_ID]}
