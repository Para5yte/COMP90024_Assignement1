#!/bin/bash

# 8 tasks, 8 tasks per node = 8 cores
#SBATCH --job-name=1node8cores
#SBATCH --partition=snowy
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --time=48:00:00

# load required modules
module load foss/2020b
module load gcc/8.3.0
module load python/3.7.4
module load numpy/1.18.0-python-3.7.4
pip install --user ujson

# launch python code
srun -n 8 python3 main.py "bigTwitter.json" "1node8cores_result.txt"