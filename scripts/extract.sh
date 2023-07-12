#!/usr/bin/bash
#SBATCH --job-name=extrAE
#SBATCH --output=logs/extAE%a.out
#SBATCH --error=logs/extAE%a.out
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tbarba@stanford.edu
#SBATCH --array=0-2
#SBATCH --mem-per-cpu=8G



# INIT
source ~/.bashrc
conda activate multimodal
nvidia-smi --list-gpus

COL='\033[1;34m'
NC='\033[0m' # No Color


# VARIABLES
export mod=( \
    "test/TCGA" \
    "test/ADNI" \
    "test/SCHIZO" \
    # "test/ADNI_rec" \
    # "test/SCHIZO_rec" \
    # "test/UPENN" \
    # "test/UCSF" \
    # "test/UKB" \
    # "radiomics/WB_RAD_ADNI" \
    # "radiomics/WB_RAD_SCHIZO" \
    # "radiomics/WB_RAD_TCGA" \
    # "radiomics/WB_RAD_UCSF" \
    # "radiomics/WB_RAD_UPENN" \
    # "radiomics/WB_RAD_UKB" \
)

model=AE_sel2
architecture=ae
keep_single=False
output=Z-$model

# SCRIPT
echo -e "\n${COL}Extracting features = ${mod[SLURM_ARRAY_TASK_ID]}${NC}"
python src/autoencoder/feature_extraction.py \
    -c ${mod[SLURM_ARRAY_TASK_ID]} \
    --other_model outputs/norm/$model $architecture \
    --output_name $output \
    -k $keep_single


# #Multi
echo -e "\n${COL}Multivariate analysis for model = ${mod[SLURM_ARRAY_TASK_ID]}${NC}"
python src/autoencoder/multivariate.py \
    -c ${mod[SLURM_ARRAY_TASK_ID]} \
    -f $output \
    -o $output

# python src/autoencoder/multivariate.py \
#     -c ${mod[SLURM_ARRAY_TASK_ID]} \
#     -f wb_radiomics \
#     -o wb_radiomics
