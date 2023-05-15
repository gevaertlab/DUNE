#!/usr/bin/bash
#SBATCH --job-name=AE_UKB
#SBATCH --output=logs/AE_UKB%a.out
#SBATCH --error=logs/AE_UKB%a.out
#SBATCH --time=48:00:00
#SBATCH --partition=ogevaert-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tbarba@stanford.edu
#SBATCH --array=0-0


source ~/.bashrc
conda activate multimodal
nvidia-smi --list-gpus
df -h /dev/shm


export mod=( \
"AE/AE_UKB" \
# "uVAE/uVAE_UKB_b1e-4" \
# "uVAE/uVAE_UKB_b1e-5" \
)
echo "Model = $mod"
CUDA_VISIBLE_DEVICES=0,1 \
    python src/autoencoder/train_ae.py -c ${mod[SLURM_ARRAY_TASK_ID]}
