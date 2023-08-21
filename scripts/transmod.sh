#!/usr/bin/bash
#SBATCH --job-name=transmod
#SBATCH --output=logs/transmod%a.out
#SBATCH --error=logs/transmod%a.out
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --mem=128G
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
    "transmod/transmUNET_rest" \
)   


# SCRIPT
echo -e "\n${COL}\nProcessing model = ${mod[SLURM_ARRAY_TASK_ID]}${NC}"
nvidia-smi --list-gpus

CUDA_VISIBLE_DEVICES=0,1 \
    python src/transmodality/train_ae.py -c ${mod[SLURM_ARRAY_TASK_ID]}

