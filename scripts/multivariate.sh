#!/usr/bin/bash
#SBATCH --job-name=att_multi
#SBATCH --output=logs/att_multi%a.out
#SBATCH --error=logs/att_multi%a.out
#SBATCH --time=48:00:00
#SBATCH --partition=normal
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tbarba@stanford.edu
#SBATCH --array=0-1:1


# INIT
source ~/.bashrc
conda activate multimodal
cd ~/projects/MultiModalBrainSurvival


# VARIABLES
export mod=( \
	     "uVAE/uVAE_UKB_b1e-4_att" \
		 "AE/AE_UKB_segm_attn" \
		#  "UNet/other/UNet_UKB_5b8f" \
		#  "UNet/other/UNet_UKB_6b4f" \
		#  "UNet/other/UNet_UKB_6b8f" \
		#  "UNet/UNet_UPENN" \
		#  "UNet/UNet_UKB_segm" \
		#  "UNet/UNet_UCSF" \
)

# SCRIPT
echo -e "Processing model = ${mod[SLURM_ARRAY_TASK_ID]}"
python src/autoencoder/multivariate.py -c ${mod[SLURM_ARRAY_TASK_ID]} -f whole_brain -o WB_oversampling
python src/autoencoder/compute_umaps.py -c ${mod[SLURM_ARRAY_TASK_ID]}

