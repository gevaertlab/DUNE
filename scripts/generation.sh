#!/usr/bin/bash
#SBATCH --job-name=generation
#SBATCH --output=logs/generation%a.out
#SBATCH --error=logs/generation%a.out
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tbarba@stanford.edu
#SBATCH --array=0-0
#SBATCH --mem-per-cpu=8G



# INIT
source ~/.bashrc
conda activate multimodal
nvidia-smi --list-gpus

COL='\033[1;34m'
NC='\033[0m' # No Color


# VARIABLES
export mod=( \

    "transmod/SCHIZO" \

)

model=transmUNET
architecture=unet
keep_single=False
output=wb3_$model

# SCRIPT
echo -e "\n${COL}Generating modality = ${mod[SLURM_ARRAY_TASK_ID]}${NC}"
python src/transmodality/modality_generation.py \
    -c ${mod[SLURM_ARRAY_TASK_ID]} \
    --other_model outputs/transmod/$model $architecture