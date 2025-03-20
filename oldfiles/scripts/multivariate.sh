#!/usr/bin/bash
#SBATCH --job-name=multi
#SBATCH --output=logs/multi%a.out
#SBATCH --error=logs/multi%a.out
#SBATCH --time=48:00:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=5
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tbarba@stanford.edu
#SBATCH --array=0-0
#SBATCH --mem-per-cpu=8G



# INIT
source ~/.bashrc
conda activate multimodal


COL='\033[1;34m'
NC='\033[0m' # No Color


# VARIABLES
export mod=( \
    # "test/ADNI" \
    "test/TCGA" \
    # "test/SCHIZO" \
    # "test/UPENN" \
    # "test/UCSF" \
    # "test/UKB" \

)

model=AE_sel2
architecture=ae
keep_single=False

# SCRIPT

python src/autoencoder/feature_extraction.py  \
    -c ${mod[SLURM_ARRAY_TASK_ID]} \
    --other_model outputs/norm/$model $architecture \
    --output_name wb_$model \
    -k $keep_single


# #Multi
echo -e "\n${COL}Multivariate analysis for model = ${mod[SLURM_ARRAY_TASK_ID]}${NC}"
python src/autoencoder/multivariate.py \
    -c ${mod[SLURM_ARRAY_TASK_ID]} \
    -f wb_$model \
    -o wb_$model

    # -f wb_$model \