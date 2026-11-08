#!/bin/bash
#SBATCH --job-name=ilc_xclass
#SBATCH --output=output/results/logging_ilc_model_%j.out
#SBATCH --error=output/results/logging_ilc_model_%j.err
#SBATCH --partition=llmq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=5-0:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

cd /home/h_larbi/ilc_Xclass_20261026/analysis/training || exit 1

export PYTHONPATH=/home/h_larbi/ilc_Xclass_20261026/analysis/training:/home/h_larbi/ilc_Xclass_20261026:$PYTHONPATH

PY=/home/h_larbi/ilc_Xclass_20261026/env_ils_xclass/bin/python
$PY --version
$PY pipeline_training_ilc.py