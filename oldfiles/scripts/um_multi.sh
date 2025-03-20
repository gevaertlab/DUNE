#!/usr/bin/bash
#SBATCH --job-name=umaps
#SBATCH --output=logs/umaps%a.out
#SBATCH --error=logs/umaps%a.out
#SBATCH --time=48:00:00
#SBATCH --partition=normal
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tbarba@stanford.edu
#SBATCH --array=0-4


#INIT
source ~/.bashrc
COL='\033[1;34m'
NC='\033[0m' # No Color



# VARBIABLES
export mod=( \
    # "norm/test/UKB" \
    "norm/test/ADNI" \
    "norm/test/UCSF" \
    "norm/test/TCGA" \
    "norm/test/UPENN" \
    "norm/test/SCHIZO" \
)


# SCRIPT
# echo -e "\n${COL}UMAP for model = ${mod[SLURM_ARRAY_TASK_ID]}${NC}"
# conda activate analysis
# python src/autoencoder/compute_umaps.py -c ${mod[SLURM_ARRAY_TASK_ID]} 

echo -e "\n${COL}Multivariate analysis for model = ${mod[SLURM_ARRAY_TASK_ID]}${NC}"
conda activate multimodal
python src/autoencoder/multivariate.py -c ${mod[SLURM_ARRAY_TASK_ID]} -f whole_brain -o UVAE_single
