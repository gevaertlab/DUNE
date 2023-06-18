#!/usr/bin/bash
#SBATCH --job-name=extr
#SBATCH --output=logs/extr%a.out
#SBATCH --error=logs/extr%a.out
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tbarba@stanford.edu
#SBATCH --array=0-0

# INIT
source ~/.bashrc
conda activate multimodal
nvidia-smi --list-gpus

# VARIABLES
export mod=( \
    # "test/ADNI" \
    # "test/TCGA" \
    # "test/SCHIZO" \
    # "test/UPENN" \
    # "test/UCSF" \
    "test/all_cohorts2" \
    # "test/UKB" \

)

model=AE_selection2
architecture=ae
keep_single=True

# SCRIPT
# echo -e "\n${COL}\Extracting features = ${mod[SLURM_ARRAY_TASK_ID]}${NC}"


# CUDA_VISIBLE_DEVICES=0,1 \
#     python src/autoencoder/train_ae.py -c ${mod[SLURM_ARRAY_TASK_ID]}


python src/autoencoder/feature_extraction.py  -c ${mod[SLURM_ARRAY_TASK_ID]} \
    --other_model outputs/norm/$model $architecture -k $keep_single


# python src/tools/format_datasets/metadata/combine_radiomics.py \
# 	--model_path outputs/${mod[SLURM_ARRAY_TASK_ID]}


# #Multi
# echo -e "\n${COL}Multivariate analysis for model = ${mod[SLURM_ARRAY_TASK_ID]}${NC}"
# python src/autoencoder/multivariate.py -c ${mod[SLURM_ARRAY_TASK_ID]} -f whole_brain \
#     -o $model
