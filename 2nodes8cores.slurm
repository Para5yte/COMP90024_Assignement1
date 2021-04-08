#!/bin/bash

# 8 tasks, 4 tasks per node = 8 cores
#SBATCH --job-name=2nodes8cores
#SBATCH --partition=snowy
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --time=48:00:00

# load required modules
module load foss/2020b
module load gcc/8.3.0
module load python/3.7.4
module load numpy/1.18.0-python-3.7.4

ln -s /data/projects/COMP90024/smallTwitter.json
ln -s /data/projects/COMP90024/AFINN.txt
ln -s /data/projects/COMP90024/melbGrid.json

# launch python code
srun -n 8 python3 main.py "smallTwitter.json" "2nodes8cores_result.txt"
