#!/bin/bash
#
#SBATCH --job-name=testingGECO
#
#SBATCH --time=180:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G

python3 brainAE.py
