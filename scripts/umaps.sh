#!/usr/bin/bash
#SBATCH --job-name=umap
#SBATCH --output=logs/umap%a.out
#SBATCH --error=logs/umap%a.out
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tbarba@stanford.edu
#SBATCH --array=0-4

# INIT
source ~/.bashrc
conda activate analysis

COL='\033[1;34m'
NC='\033[0m' # No Color

# VARIABLES
repo=test
keep_single=False
dataset=UKB

export model=(
    VAE3D_sel2
    AE_sel2
    UNet_sel2
    UVAE_sel2
    radiomics
)

output=Z-${model[SLURM_ARRAY_TASK_ID]}

# CUDA_VISIBLE_DEVICES=0,1 \
#     python src/autoencoder/train_ae.py -c ${mod[SLURM_ARRAY_TASK_ID]}

# SCRIPT
echo -e "\n${COL}UMAP for model = ${model[SLURM_ARRAY_TASK_ID]}${NC}"
python src/autoencoder/compute_umaps.py \
    -c $repo/$dataset \
    -k $keep_single \
    -f $output
