#!/usr/bin/bash
#SBATCH --job-name=extrV
#SBATCH --output=logs/extrV%a.out
#SBATCH --error=logs/extr%Va.out
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tbarba@stanford.edu
#SBATCH --array=0-0
#SBATCH --mem-per-cpu=8G

### # SBATCH --gres=gpu:1


# INIT
source ~/.bashrc
conda activate multimodal
nvidia-smi --list-gpus

# VARIABLES
export mod=( \
    # "test/ADNI" \
    "test/UKB" \
    # "test/TCGA_T1" \
    # "test/SCHIZO" \
    # "test/UPENN" \
    # "test/UCSF" \
    # "test/all_cohorts2" \
    # "test/UKB" \

)

model=VAE3D_sel2
architecture=vae3d
keep_single=False

# SCRIPT
# echo -e "\n${COL}\Extracting features = ${mod[SLURM_ARRAY_TASK_ID]}${NC}"


# CUDA_VISIBLE_DEVICES=0,1 \
#     python src/autoencoder/train_ae.py -c ${mod[SLURM_ARRAY_TASK_ID]}

CUDA_VISIBLE_DEVICES="" \
    python src/autoencoder/feature_extraction.py  -c ${mod[SLURM_ARRAY_TASK_ID]} \
    --other_model outputs/norm/$model $architecture -k $keep_single


# python src/tools/format_datasets/metadata/combine_radiomics.py \
# 	--model_path outputs/${mod[SLURM_ARRAY_TASK_ID]}


# #Multi
# echo -e "\n${COL}Multivariate analysis for model = ${mod[SLURM_ARRAY_TASK_ID]}${NC}"
# python src/autoencoder/multivariate.py -c ${mod[SLURM_ARRAY_TASK_ID]} -f whole_brain \
#     -o $model
