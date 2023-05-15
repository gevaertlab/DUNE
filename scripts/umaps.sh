#!/usr/bin/bash
#SBATCH --job-name=umap_UNet_UCSF_segm
#SBATCH --output=logs/umap_UNet_UCSF_segm%a.out
#SBATCH --error=logs/umap_UNet_UCSF_segm%a.out
#SBATCH --time=1:00:00
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tbarba@stanford.edu
#SBATCH --array=0-0:1


source ~/.bashrc
conda activate analysis
nvidia-smi --list-gpus

# export mod=( \
# "UNet/other/UNet_UKB_5b4f" \
# "UNet/other/UNet_UKB_5b8f" \
# "UNet/other/UNet_UKB_6b4f" \
# "UNet/other/UNet_UKB_6b8f" \
# )
export mod=( \
"UNet/UNet_UCSF_segm" \
)

cd /home/tbarba/projects/MultiModalBrainSurvival
python src/autoencoder/compute_umaps.py -c ${mod[SLURM_ARRAY_TASK_ID]}


conda activate multimodal
python src/autoencoder/multivariate.py -c ${mod} -f whole_brain -o WB_oversampling
