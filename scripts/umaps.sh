#!/usr/bin/bash
#SBATCH --job-name=umap
#SBATCH --output=logs/umap%a.out
#SBATCH --error=logs/umap%a.out
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tbarba@stanford.edu
#SBATCH --array=0-0


# VARIABLES
COL='\033[1;35m'
NC='\033[0m' # No Color


#INIT
source ~/.bashrc
conda activate analysis
cd ~/projects/MultiModalBrainSurvival

# VARBIABLES
export mod=( \
    # "test/TCGA" \
    # "test/ADNI" \
    # "test/UPENN" \
    # "test/UCSF" \
    # "test/SCHIZO" \
    "test/all_cohorts2" \
    # "test/UKB" \
)

keep_single=True


# CUDA_VISIBLE_DEVICES=0,1 \
#     python src/autoencoder/train_ae.py -c ${mod[SLURM_ARRAY_TASK_ID]}


# SCRIPT
echo -e "\n${COL}UMAP for model = ${mod[SLURM_ARRAY_TASK_ID]}${NC}"
python src/autoencoder/compute_umaps.py -c ${mod[SLURM_ARRAY_TASK_ID]} -k $keep_single
