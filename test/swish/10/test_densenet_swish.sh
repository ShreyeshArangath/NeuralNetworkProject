#!/bin/sh
#SBATCH -J test_densenet_swish
#SBATCH  -o %x.%j.out
#SBATCH  -e %x.%j.err
#SBATCH  -p quanah
#SBATCH -N 1
#SBATCH --ntasks-per-node 20

. $HOME/conda/etc/profile.d/conda.sh
conda activate

python test_densenet_swish.py