#!/bin/bash

# 1 node, 1 task per node = 1 core
#SBATCH --job-name=1node1core
#SBATCH --partition=snowy
#SBATCH --ntasks=1
#SBATCH --time=48:00:00

# load required modules
module load foss/2020b
module load gcc/8.3.0
module load python/3.7.4
module load numpy/1.18.0-python-3.7.4

# create symbolic link to current directory if they're not already here
ln -s /data/projects/COMP90024/bigTwitter.json
ln -s /data/projects/COMP90024/AFINN.txt
ln -s /data/projects/COMP90024/melbGrid.json

# launch python code
srun -n 1 python3 main.py "bigTwitter.json" "1node1core_result.txt"
