#!/bin/sh
#
#SBATCH --account=free # The account name for the job.
#SBATCH --job-name=Fairness_Checking_T50 # The job name.
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sd3013@columbia.edu
#SBATCH --exclusive
#SBATCH -c 8 # The number of cpu cores to use.
#SBATCH --time=6:00:00 # The time the job will take to run.

module load anaconda/3-2019.03
source activate /rigel/home/sd3013/.conda/envs/fairness_checking

### Split 4 ###

python main.py --num_cores 8 --T_inner 500 --T 50 --eta 0.15 --eta_inner 0.55 --gamma_2 0.05 --constraint dp --B 0.15 --dataset adult --name final_dp_adult --split 4

python main.py --num_cores 8 --T_inner 500 --T 50 --eta 0.15 --eta_inner 0.5 --gamma_2 0.05 --constraint dp --B 0.165 --dataset lawschool --name final_dp_lawschool --split 4

python main.py --num_cores 8 --T_inner 500 --T 50 --eta 0.15 --eta_inner 0.55 --gamma_2 0.05 --constraint dp --B 0.15 --dataset communities --name final_dp_communities --split 4

### Split 5 ###

python main.py --num_cores 8 --T_inner 500 --T 50 --eta 0.15 --eta_inner 0.55 --gamma_2 0.05 --constraint dp --B 0.15 --dataset adult --name final_dp_adult --split 5

python main.py --num_cores 8 --T_inner 500 --T 50 --eta 0.15 --eta_inner 0.5 --gamma_2 0.05 --constraint dp --B 0.165 --dataset lawschool --name final_dp_lawschool --split 5

python main.py --num_cores 8 --T_inner 500 --T 50 --eta 0.15 --eta_inner 0.55 --gamma_2 0.05 --constraint dp --B 0.15 --dataset communities --name final_dp_communities --split 5