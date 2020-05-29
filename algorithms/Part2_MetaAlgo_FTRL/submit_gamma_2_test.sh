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
python main.py --solver ECOS --num_cores 14 --no_output y --name gamma_0.001 --gamma_2 0.001

python main.py --solver ECOS --num_cores 14 --no_output y --name gamma_0.01 --gamma_2 0.01

python main.py --solver ECOS --num_cores 14 --no_output y --name gamma_0.02 --gamma_2 0.02

python main.py --solver ECOS --num_cores 14 --no_output y --name gamma_0.05 --gamma_2 0.05

python main.py --solver ECOS --num_cores 14 --no_output y --name gamma_0.1 --gamma_2 0.1

python main.py --solver ECOS --num_cores 14 --no_output y --name gamma_0.2 --gamma_2 0.2

python main.py --solver ECOS --num_cores 14 --no_output y --name gamma_0.3 --gamma_2 0.3

#End of script
