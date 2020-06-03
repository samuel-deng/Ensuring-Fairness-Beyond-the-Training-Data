#!/bin/sh
#
#SBATCH --account=free # The account name for the job.
#SBATCH --job-name=Fairness_Checking_T25 # The job name.
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sd3013@columbia.edu
#SBATCH --exclusive
#SBATCH -N 1 # The number of cpu cores to use.
#SBATCH --time=6:00:00 # The time the job will take to run.

### B = 0.275 ###

python main.py --num_cores 8 --no_output y --T_inner 1000 --T 1 --eta 0.15 --eta_inner 0.5 --gamma_2 0.05 --constraint dp --B 0.275 --dataset compas --onlyh0 y

python main.py --num_cores 8 --no_output y --T_inner 1000 --T 1 --eta 0.15 --eta_inner 0.55 --gamma_2 0.05 --constraint dp --B 0.275 --dataset compas --onlyh0 y

python main.py --num_cores 8 --no_output y --T_inner 1000 --T 1 --eta 0.15 --eta_inner 0.6 --gamma_2 0.05 --constraint dp --B 0.275 --dataset compas --onlyh0 y

python main.py --num_cores 8 --no_output y --T_inner 1000 --T 1 --eta 0.15 --eta_inner 0.65 --gamma_2 0.05 --constraint dp --B 0.275 --dataset compas --onlyh0 y