#!/usr/bin/bash
#SBATCH --job-name=ext_UNet_UCSF_segm
#SBATCH --output=logs/ext_UNet_UCSF_segm.out
#SBATCH --error=logs/ext_UNet_UCSF_segm.out
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tbarba@stanford.edu


source ~/.bashrc
conda activate multimodal
nvidia-smi --list-gpus


export mod=( "UNet/UNet_UCSF_segm" )

cd /home/tbarba/projects/MultiModalBrainSurvival
CUDA_VISIBLE_DEVICES=0,1 \
    python src/autoencoder/feature_extraction.py  -c ${mod}



