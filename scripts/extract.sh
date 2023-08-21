#!/usr/bin/bash
#SBATCH --job-name=AE_sel2
#SBATCH --output=logs/AE_sel2%a.out
#SBATCH --error=logs/AE_sel2%a.out
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=10
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tbarba@stanford.edu
#SBATCH --array=0-0
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1



# INIT
source ~/.bashrc
conda activate multimodal
nvidia-smi --list-gpus

COL='\033[1;34m'
NC='\033[0m' # No Color


# VARIABLES
export dataset=( \
    # UKB \
    # UPENN \
    # ADNI_rec \
    # SCHIZO \
    # SCHIZO_rec \
    # TCGA \
    UCSF \
    # "test/ADNI_rec" \
    # "test/SCHIZO_rec" \
    # "radiomics/WB_RAD_ADNI" \
    # "radiomics/WB_RAD_SCHIZO" \
    # "radiomics/WB_RAD_TCGA" \
    # "radiomics/WB_RAD_UCSF" \
    # "radiomics/WB_RAD_UPENN" \
    # "radiomics/WB_RAD_UKB" \
)

model=AE_sel2
architecture=ae
repo=test
keep_single=False
output=Y-$model

# SCRIPT
echo -e "\n${COL}Extracting features = $repo/${dataset[SLURM_ARRAY_TASK_ID]}${NC}"
python src/autoencoder/feature_extraction.py \
    -c $repo/${dataset[SLURM_ARRAY_TASK_ID]} \
    --other_model outputs/norm/$model $architecture \
    --output_name $output \
    -k $keep_single


# #Multi
echo -e "\n${COL}Multivariate analysis for model = $repo/${dataset[SLURM_ARRAY_TASK_ID]}${NC}"
python src/autoencoder/multivariate.py \
    -c $repo/${dataset[SLURM_ARRAY_TASK_ID]} \
    -f $output \
    -o $output

