#!/usr/bin/bash
#SBATCH --job-name=UVAEsel3
#SBATCH --output=logs/UVAEsel3%a.out
#SBATCH --error=logs/UVAEsel3%a.out
#SBATCH --time=48:00:00
#SBATCH --partition=ogevaert-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH -C GPU_MEM:32GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tbarba@stanford.edu
#SBATCH --array=0-0



# INIT
source ~/.bashrc
conda activate multimodal

# VARIABLES
COL='\033[1;35m'
NC='\033[0m' # No Color

export mod=( \
# "norm/AE_selection_attn" \
# "norm/VAE3D_selection" \
# "norm/AE_selection" \
"norm/UVAE_selection3" \
# "norm/VAE3D_selection2" \
# "norm/AE_selection2" \
# "test/all_cohorts2" \
# "norm/GAE_selection2" \
# "norm/VAE3D_selection3" \
# "norm/VAE3D_selection" \
)   


# SCRIPT
echo -e "\n${COL}\nProcessing model = ${mod[SLURM_ARRAY_TASK_ID]}${NC}"
nvidia-smi --list-gpus

CUDA_VISIBLE_DEVICES=0,1 \
    python src/autoencoder/train_ae.py -c ${mod[SLURM_ARRAY_TASK_ID]}
