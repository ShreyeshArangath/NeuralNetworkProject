#!/bin/sh
#SBATCH -J efficientnetb7
#SBATCH  -o %x.%j.out
#SBATCH  -e %x.%j.err
#SBATCH  -p quanah
#SBATCH -N 1
#SBATCH --ntasks-per-node 30

. $HOME/conda/etc/profile.d/conda.sh
conda activate

python efficientnetb7.py