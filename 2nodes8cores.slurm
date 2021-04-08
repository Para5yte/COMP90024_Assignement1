#!/bin/bash

# 8 tasks, 4 tasks per node = 8 cores
#SBATCH --job-name=2nodes8cores
#SBATCH --partition=snowy
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
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
srun -n 8 python3 main.py "bigTwitter.json" "2nodes8cores_result.txt"