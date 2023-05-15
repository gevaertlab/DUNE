#!/usr/bin/bash
#SBATCH --job-name=uUKB
#SBATCH --output=logs/uVAE_UKB.out
#SBATCH --error=logs/uVAE_UKB.out
#SBATCH --time=48:00:00
#SBATCH --partition=ogevaert-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=SEND,FAIL
#SBATCH --mail-user=tbarba@stanford.edu
#SBATCH --array=0-1:1



source ~/.bashrc
conda activate multimodal
nvidia-smi --list-gpus


export mod=( "uVAE/uVAE_UKB" )

cd /home/tbarba/projects/MultiModalBrainSurvival
CUDA_VISIBLE_DEVICES=0,1 \
    python src/autoencoder/train_ae.py \
        -c configs/${mod[${SLURM_ARRAY_TASK_ID}]}
