#!/bin/sh
#
#SBATCH --account=free # The account name for the job.
#SBATCH --job-name=Fairness_Checking # The job name.
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sd3013@columbia.edu
#SBATCH -c 8 # The number of cpu cores to use.
#SBATCH --time=4:00:00 # The time the job will take to run.

python main.py --T_inner 1000 --T 1 --B 0.2 --eta_inner 0.5 --constraint dp --dataset communities --gp_wt_bd 0.049845 --onlyh0 y --no_output y

python main.py --T_inner 1000 --T 1 --B 0.2 --eta_inner 0.55 --constraint dp --dataset communities --gp_wt_bd 0.049845 --onlyh0 y --no_output y

python main.py --T_inner 1000 --T 1 --B 0.2 --eta_inner 0.6 --constraint dp --dataset communities --gp_wt_bd 0.049845 --onlyh0 y --no_output y

python main.py --T_inner 1000 --T 1 --B 0.2 --eta_inner 0.65 --constraint dp --dataset communities --gp_wt_bd 0.049845 --onlyh0 y --no_output y

python main.py --T_inner 1000 --T 1 --B 0.225 --eta_inner 0.5 --constraint dp --dataset communities --gp_wt_bd 0.049845 --onlyh0 y --no_output y

python main.py --T_inner 1000 --T 1 --B 0.225 --eta_inner 0.55 --constraint dp --dataset communities --gp_wt_bd 0.049845 --onlyh0 y --no_output y

python main.py --T_inner 1000 --T 1 --B 0.225 --eta_inner 0.6 --constraint dp --dataset communities --gp_wt_bd 0.049845 --onlyh0 y --no_output y

python main.py --T_inner 1000 --T 1 --B 0.225 --eta_inner 0.65 --constraint dp --dataset communities --gp_wt_bd 0.049845 --onlyh0 y --no_output y

python main.py --T_inner 1000 --T 1 --B 0.25 --eta_inner 0.5 --constraint dp --dataset communities --gp_wt_bd 0.049845 --onlyh0 y --no_output y

python main.py --T_inner 1000 --T 1 --B 0.25 --eta_inner 0.55 --constraint dp --dataset communities --gp_wt_bd 0.049845 --onlyh0 y --no_output y

python main.py --T_inner 1000 --T 1 --B 0.25 --eta_inner 0.6 --constraint dp --dataset communities --gp_wt_bd 0.049845 --onlyh0 y --no_output y

python main.py --T_inner 1000 --T 1 --B 0.25 --eta_inner 0.65 --constraint dp --dataset communities --gp_wt_bd 0.049845 --onlyh0 y --no_output y

python main.py --T_inner 1000 --T 1 --B 0.275 --eta_inner 0.5 --constraint dp --dataset communities --gp_wt_bd 0.049845 --onlyh0 y --no_output y

python main.py --T_inner 1000 --T 1 --B 0.275 --eta_inner 0.55 --constraint dp --dataset communities --gp_wt_bd 0.049845 --onlyh0 y --no_output y

python main.py --T_inner 1000 --T 1 --B 0.275 --eta_inner 0.6 --constraint dp --dataset communities --gp_wt_bd 0.049845 --onlyh0 y --no_output y

python main.py --T_inner 1000 --T 1 --B 0.275 --eta_inner 0.65 --constraint dp --dataset communities --gp_wt_bd 0.049845 --onlyh0 y --no_output y