#!/bin/sh
#
#
#SBATCH --account=free # The account name for the job.
#SBATCH --job-name=Fairness_Checking # The job name.
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sd3013@columbia.edu
#SBATCH --exclusive
#SBATCH -N 1 # The number of cpu cores to use.
#SBATCH --time=4:00:00 # The time the job will take to run.

module load anaconda/3-2019.03
source activate /rigel/home/sd3013/.conda/envs/fairness_checking

#Command to execute Python program
python main.py --solver ECOS --num_cores 14 --T_inner 500 --T 1 --eta_inner 0.2 --gamma_2 0.05 --constraint dp --B 0.105 --no_output y --dataset communities --gp_wt_bd 0.0498

python main.py --solver ECOS --num_cores 14 --T_inner 500 --T 1 --eta_inner 0.25 --gamma_2 0.05 --constraint dp --B 0.105 --no_output y --dataset communities --gp_wt_bd 0.0498

python main.py --solver ECOS --num_cores 14 --T_inner 500 --T 1 --eta_inner 0.3 --gamma_2 0.05 --constraint dp --B 0.105 --no_output y --dataset communities --gp_wt_bd 0.0498

python main.py --solver ECOS --num_cores 14 --T_inner 500 --T 1 --eta_inner 0.35 --gamma_2 0.05 --constraint dp --B 0.105 --no_output y --dataset communities --gp_wt_bd 0.0498

python main.py --solver ECOS --num_cores 14 --T_inner 500 --T 1 --eta_inner 0.4 --gamma_2 0.05 --constraint dp --B 0.105 --no_output y --dataset communities --gp_wt_bd 0.0498

python main.py --solver ECOS --num_cores 14 --T_inner 500 --T 1 --eta_inner 0.45 --gamma_2 0.05 --constraint dp --B 0.105 --no_output y --dataset communities --gp_wt_bd 0.0498

python main.py --solver ECOS --num_cores 14 --T_inner 500 --T 1 --eta_inner 0.5 --gamma_2 0.05 --constraint dp --B 0.105 --no_output y --dataset communities --gp_wt_bd 0.0498

python main.py --solver ECOS --num_cores 14 --T_inner 500 --T 1 --eta_inner 0.55 --gamma_2 0.05 --constraint dp --B 0.105 --no_output y --dataset communities --gp_wt_bd 0.0498

python main.py --solver ECOS --num_cores 14 --T_inner 500 --T 1 --eta_inner 0.6 --gamma_2 0.05 --constraint dp --B 0.105 --no_output y --dataset communities --gp_wt_bd 0.0498

python main.py --solver ECOS --num_cores 14 --T_inner 500 --T 1 --eta_inner 0.65 --gamma_2 0.05 --constraint dp --B 0.105 --no_output y --dataset communities --gp_wt_bd 0.0498

python main.py --solver ECOS --num_cores 14 --T_inner 500 --T 1 --eta_inner 0.7 --gamma_2 0.05 --constraint dp --B 0.105 --no_output y --dataset communities --gp_wt_bd 0.0498

python main.py --solver ECOS --num_cores 14 --T_inner 500 --T 1 --eta_inner 0.75 --gamma_2 0.05 --constraint dp --B 0.105 --no_output y --dataset communities --gp_wt_bd 0.0498

python main.py --solver ECOS --num_cores 14 --T_inner 500 --T 1 --eta_inner 0.8 --gamma_2 0.05 --constraint dp --B 0.105 --no_output y --dataset communities --gp_wt_bd 0.0498

#End of script
