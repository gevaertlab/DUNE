#!/usr/bin/bash
#SBATCH --job-name=umAEatt
#SBATCH --output=logs/umAEatt%a.out
#SBATCH --error=logs/umAEatt%a.out
#SBATCH --time=3:00:00
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tbarba@stanford.edu
#SBATCH --array=0-0:1


#INIT
source ~/.bashrc
conda activate analysis
nvidia-smi --list-gpus
cd ~/projects/MultiModalBrainSurvival

# VARBIABLES
export mod=( \
"uVAE/uVAE_UKB_b1e-4" \
# "AE/AE_UKB_segm_attn" \
# "radiomics/RAD_UKB" \
)

# SCRIPT
echo -e "Processing model = ${mod[SLURM_ARRAY_TASK_ID]}"
python src/autoencoder/compute_umaps.py -c ${mod[SLURM_ARRAY_TASK_ID]}
