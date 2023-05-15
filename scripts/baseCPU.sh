#!/usr/bin/bash
#SBATCH --job-name=pyrad
#SBATCH --output=pyrad.out
#SBATCH --error=pyrad.out
#SBATCH --time=24:00:00
#SBATCH --partition=ogevaert-a100
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=SEND,FAIL
#SBATCH --mail-user=tbarba@stanford.edu


source ~/.bashrc
conda activate multimodal
nvidia-smi --list-gpus

export mod=( "AE/AE_UKB_segm" )

cd /home/tbarba/projects/MultiModalBrainSurvival
python src/autoencoder/multivariate.py -c ${mod} -f radiomics -o pyradiomics
