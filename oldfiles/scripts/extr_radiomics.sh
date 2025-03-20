#!/usr/bin/bash
#SBATCH --job-name=radiomics
#SBATCH --output=logs/radiomics%a.out
#SBATCH --error=logs/radiomics%a.out
#SBATCH --time=48:00:00
#SBATCH --partition=normal
#SBATCH --cpus-per-task=10
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tbarba@stanford.edu
#SBATCH --array=0-0
#SBATCH --mem-per-cpu=8G
##SBATCH --gres=gpu:1



# INIT
source ~/.bashrc
conda activate brainextract
nvidia-smi --list-gpus

COL='\033[1;34m'
NC='\033[0m' # No Color


# VARIABLES
export dataset=( \
    # UKB \
    # UPENN \
    # ADNI \
    SCHIZO \
    # UPENN \
    # UPENN \
    # "test/ADNI_rec" \
    # "test/SCHIZO_rec" \
    # "radiomics/WB_RAD_ADNI" \
    # "radiomics/WB_RAD_SCHIZO" \
    # "radiomics/WB_RAD_TCGA" \
    # "radiomics/WB_RAD_UCSF" \
    # "radiomics/WB_RAD_UPENN" \
    # "radiomics/WB_RAD_UKB" \
)

model=radiomics

# SCRIPT
echo -e "\n${COL}Extracting features = $repo/${dataset[SLURM_ARRAY_TASK_ID]}${NC}"
python src/tools/format_datasets/metadata/extract_pyradiomics.py ${dataset[SLURM_ARRAY_TASK_ID]}




