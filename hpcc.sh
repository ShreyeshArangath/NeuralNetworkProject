wd
#$ -S /bin/bash
#$ -N MLTESTS
#$ -o $JOB_NAME.o$JOB_ID
#$ -e $JOB_NAME.e$JOB_ID
#$ -q omni
#$ -M shreyesh.arangath@ttu.edu
#$ -m beas
#$ -pe sm 36
#$ -l h_vmem=5.3G
#$ -l h_rt=48:00:00
#$ -P quanah
. $HOME/conda/etc/profile.d/conda.sh
conda activate

python3 main.py
