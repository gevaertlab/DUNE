#!/usr/bin/bash
#SBATCH --job-name=UVAE
#SBATCH --output=logs/UVAE%a.out
#SBATCH --error=logs/UVAE%a.out
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=256G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tbarba@stanford.edu
#SBATCH --array=0-0
##SBATCH -C GPU_MEM:32GB

# INIT
source ~/.bashrc
conda activate multimodal

# VARIABLES
COL='\033[1;35m'
NC='\033[0m' # No Color

export mod=(
    # "norm/AE_selection_attn" \
    # "norm/VAE3D_selection" \
    "norm/UVAE_sel2"
    # "test/SCHIZO_rec" \
)

# SCRIPT
echo -e "\n${COL}\nProcessing model = ${mod[SLURM_ARRAY_TASK_ID]}${NC}"
nvidia-smi --list-gpus

CUDA_VISIBLE_DEVICES=0,1 \
    python src/autoencoder/train_ae.py -c ${mod[SLURM_ARRAY_TASK_ID]}
