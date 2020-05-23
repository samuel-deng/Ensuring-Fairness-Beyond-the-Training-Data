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
python main.py --solver ECOS --num_cores 14 --T_inner 400 --T 1 --eta_inner 0.5 --gamma_2 0.05 --constraint eo --B 0.05 --no_output y --dataset compas

python main.py --solver ECOS --num_cores 14 --T_inner 400 --T 1 --eta_inner 0.5 --gamma_2 0.05 --constraint eo --B 0.1 --no_output y --dataset compas

python main.py --solver ECOS --num_cores 14 --T_inner 400 --T 1 --eta_inner 0.5 --gamma_2 0.05 --constraint eo --B 0.15 --no_output y --dataset compas

python main.py --solver ECOS --num_cores 14 --T_inner 400 --T 1 --eta_inner 0.5 --gamma_2 0.05 --constraint eo --B 0.2 --no_output y --dataset compas

python main.py --solver ECOS --num_cores 14 --T_inner 400 --T 1 --eta_inner 0.5 --gamma_2 0.05 --constraint eo --B 0.25 --no_output y --dataset compas

python main.py --solver ECOS --num_cores 14 --T_inner 400 --T 1 --eta_inner 0.5 --gamma_2 0.05 --constraint eo --B 0.35 --no_output y --dataset compas

python main.py --solver ECOS --num_cores 14 --T_inner 400 --T 1 --eta_inner 0.5 --gamma_2 0.05 --constraint eo --B 0.5 --no_output y --dataset compas

python main.py --solver ECOS --num_cores 14 --T_inner 400 --T 1 --eta_inner 0.5 --gamma_2 0.05 --constraint eo --B 0.75 --no_output y --dataset compas

python main.py --solver ECOS --num_cores 14 --T_inner 400 --T 1 --eta_inner 0.5 --gamma_2 0.05 --constraint eo --B 1 --no_output y --dataset compas

#End of script
