#!/bin/bash

#SBATCH --partition=snowy
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4

module load foss/2019b
module load python/3.7.4
module load numpy/1.18.0-python-3.7.4
pip install --user shapely

srun -n 8 python3 main.py