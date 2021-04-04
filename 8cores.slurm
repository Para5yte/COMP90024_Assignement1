#!/bin/bash

# 8 tasks, 8 tasks per node = 8 cores
#SBATCH --partition=snowy
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8

# load required modules
module load foss/2019b
module load python/3.7.4
module load numpy/1.18.0-python-3.7.4
pip install --user shapely

# launch python code
srun -n 8 python3 main2.py "bigTwitter.json"