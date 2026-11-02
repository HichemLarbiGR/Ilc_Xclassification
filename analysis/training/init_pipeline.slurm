#!/bin/bash
#SBATCH --job-name=ilc_xclass
#SBATCH --output=output/results/logging_ilc_model_uni2_%j.out
#SBATCH --error=output/results/logging_ilc_model_uni2_%j.err
#SBATCH --partition=llmq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=5-0:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Exécuter
python3 pipeline_training_ilc.py