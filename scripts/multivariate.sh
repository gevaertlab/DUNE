#!/usr/bin/bash
#SBATCH --job-name=multi_UNet
#SBATCH --output=logs/multi_UNet%a.out
#SBATCH --error=logs/multi_UNet%a.out
#SBATCH --time=48:00:00
#SBATCH --partition=normal
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tbarba@stanford.edu
#SBATCH --array=0-2:1

source ~/.bashrc
conda activate multimodal


export mod=( \
	    #  "UNet/other/UNet_UKB_5b4f" \
		#  "UNet/other/UNet_UKB_5b8f" \
		#  "UNet/other/UNet_UKB_6b4f" \
		#  "UNet/other/UNet_UKB_6b8f" \
		 "UNet/UNet_UPENN_segm" \
		 "UNet/UNet_UKB_segm" \
		 "UNet/UNet_UCSF_segm" \
)

export features=( "whole_brain" )
cd /home/tbarba/projects/MultiModalBrainSurvival

echo -e "Processing model = ${mod[SLURM_ARRAY_TASK_ID]}"
python src/autoencoder/multivariate.py -c ${mod[SLURM_ARRAY_TASK_ID]} -f ${features} -o WB_oversampling

