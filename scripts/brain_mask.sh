#!/usr/bin/bash
#SBATCH --job-name=rad
#SBATCH --output=logs/rad%a.out
#SBATCH --error=logs/rad%a.out
#SBATCH --time=48:00:00
#SBATCH --partition=normal
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tbarba@stanford.edu
#SBATCH --array=0-0

# VARIABLES
COL='\033[1;34m'
NC='\033[0m' # No Color


export dataset=( "UKBIOBANK" )

#INIT
source ~/.bashrc
conda activate analysis


echo echo -e "\n${COL}Processing = ${dataset[SLURM_ARRAY_TASK_ID]}${NC}"
# python src/tools/format_datasets/compute_brain_mask.py ${dataset[SLURM_ARRAY_TASK_ID]}
python src/tools/format_datasets/metadata/extract_pyradiomics.py ${dataset[SLURM_ARRAY_TASK_ID]}