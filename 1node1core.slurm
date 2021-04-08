#!/bin/bash

# 1 node, 1 task per node = 1 core
#SBATCH --job-name=1node1core
#SBATCH --partition=snowy
#SBATCH --ntasks=1
#SBATCH --time=48:00:00

#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR

# load required modules
module load foss/2020b
module load gcc/8.3.0
module load python/3.7.4
module load numpy/1.18.0-python-3.7.4
pip install --user ujson

# launch python code
srun -n 1 python3 main.py "bigTwitter.json" "1node1core_result.txt"