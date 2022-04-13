#!/bin/sh
#SBATCH -J test_eb7_swish
#SBATCH  -o %x.%j.out
#SBATCH  -e %x.%j.err
#SBATCH  -p quanah
#SBATCH -N 1
#SBATCH --ntasks-per-node 30

. $HOME/conda/etc/profile.d/conda.sh
conda activate

python test_eb7_swish.py