#!/usr/bin/bash
#SBATCH --job-name=dicom
#SBATCH --output=logs/dicom%a.out
#SBATCH --error=logs/dicom%a.out
#SBATCH --time=48:00:00
#SBATCH --partition=normal
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tbarba@stanford.edu
#SBATCH --array=0-0


#INIT
source ~/.bashrc
COL='\033[1;34m'
NC='\033[0m' # No Color



echo -e "\n${COL}Importing DICOM${NC}"
conda activate brainextract
python src/preprocessing/import_dicom.py