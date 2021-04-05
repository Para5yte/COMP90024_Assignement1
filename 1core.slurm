#!/bin/bash

# 1 node, 1 task per node = 1 core
#SBATCH --job-name=1node1core
#SBATCH --partition=snowy
#SBATCH --ntasks=1
#SBATCH --time=48:00:00

# load required modules
module load foss/2019b
module load python/3.7.4
module load numpy/1.18.0-python-3.7.4
pip install --user shapely

# launch python code
srun -n 1 python3 main2.py "smallTwitter.json"