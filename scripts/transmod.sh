#!/usr/bin/bash
#SBATCH --job-name=transmod
#SBATCH --output=logs/transmod%a.out
#SBATCH --error=logs/transmod%a.out
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --mem=64G
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
    "transmod/transmUNET" \
)   


# SCRIPT
echo -e "\n${COL}\nProcessing model = ${mod[SLURM_ARRAY_TASK_ID]}${NC}"
nvidia-smi --list-gpus

CUDA_VISIBLE_DEVICES=0,1 \
    python src/autoencoder/train_ae2.py -c ${mod[SLURM_ARRAY_TASK_ID]}

