#!/bin/sh
#
#SBATCH --account=free # The account name for the job.
#SBATCH --job-name=Fairness_Checking_T50 # The job name.
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sd3013@columbia.edu
#SBATCH --exclusive
#SBATCH -N 1
#SBATCH --time=6:00:00 # The time the job will take to run.

module load anaconda/3-2019.03
source activate /rigel/home/sd3013/.conda/envs/fairness_checking

### Split 1 ###

python main.py --num_cores 16 --T_inner 1000 --T 5 --eta 0.15 --eta_inner 0.45 --gamma_2 0.05 --constraint eo --B 0.25 --dataset adult --name final_eo_adult

python main.py --num_cores 16 --T_inner 1000 --T 5 --eta 0.15 --eta_inner 0.45 --gamma_2 0.05 --constraint eo --B 0.25 --dataset lawschool --name final_eo_lawschool

python main.py --num_cores 16 --T_inner 1000 --T 5 --eta 0.15 --eta_inner 0.45 --gamma_2 0.05 --constraint eo --B 0.25 --dataset communities --name final_eo_communities

### Split 2 ###

python main.py --num_cores 16 --T_inner 1000 --T 5 --eta 0.15 --eta_inner 0.45 --gamma_2 0.05 --constraint eo --B 0.25 --dataset adult --name final_eo_adult --split 2

python main.py --num_cores 16 --T_inner 1000 --T 5 --eta 0.15 --eta_inner 0.45 --gamma_2 0.05 --constraint eo --B 0.25 --dataset lawschool --name final_eo_lawschool --split 2

python main.py --num_cores 16 --T_inner 1000 --T 5 --eta 0.15 --eta_inner 0.45 --gamma_2 0.05 --constraint eo --B 0.25 --dataset communities --name final_eo_communities --split 2

### Split 3 ###

python main.py --num_cores 16 --T_inner 1000 --T 5 --eta 0.15 --eta_inner 0.45 --gamma_2 0.05 --constraint eo --B 0.25 --dataset adult --name final_eo_adult --split 3

python main.py --num_cores 16 --T_inner 1000 --T 5 --eta 0.15 --eta_inner 0.45 --gamma_2 0.05 --constraint eo --B 0.25 --dataset lawschool --name final_eo_lawschool --split 3

python main.py --num_cores 16 --T_inner 1000 --T 5 --eta 0.15 --eta_inner 0.45 --gamma_2 0.05 --constraint eo --B 0.25 --dataset communities --name final_eo_communities --split 3

### Split 4 ###

python main.py --num_cores 16 --T_inner 1000 --T 5 --eta 0.15 --eta_inner 0.45 --gamma_2 0.05 --constraint eo --B 0.25 --dataset adult --name final_eo_adult --split 4

python main.py --num_cores 16 --T_inner 1000 --T 5 --eta 0.15 --eta_inner 0.45 --gamma_2 0.05 --constraint eo --B 0.25 --dataset lawschool --name final_eo_lawschool --split 4

python main.py --num_cores 16 --T_inner 1000 --T 5 --eta 0.15 --eta_inner 0.45 --gamma_2 0.05 --constraint eo --B 0.25 --dataset communities --name final_eo_communities --split 4

### Split 5 ###

python main.py --num_cores 16 --T_inner 1000 --T 5 --eta 0.15 --eta_inner 0.45 --gamma_2 0.05 --constraint eo --B 0.25 --dataset adult --name final_eo_adult --split 5

python main.py --num_cores 16 --T_inner 1000 --T 5 --eta 0.15 --eta_inner 0.45 --gamma_2 0.05 --constraint eo --B 0.25 --dataset lawschool --name final_eo_lawschool --split 5

python main.py --num_cores 16 --T_inner 1000 --T 5 --eta 0.15 --eta_inner 0.45 --gamma_2 0.05 --constraint eo --B 0.25 --dataset communities --name final_eo_communities --split 5