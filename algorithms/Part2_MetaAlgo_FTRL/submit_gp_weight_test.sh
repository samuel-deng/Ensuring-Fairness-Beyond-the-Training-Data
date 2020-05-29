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
python main.py --solver ECOS --num_cores 14 --name gp_0.05 --gp_wt_bd 0.05

python main.py --solver ECOS --num_cores 14 --name gp_0.1 --gp_wt_bd 0.1

python main.py --solver ECOS --num_cores 14 --name gp_0.15 --gp_wt_bd 0.15

python main.py --solver ECOS --num_cores 14 --name gp_0.2 --gp_wt_bd 0.2

python main.py --solver ECOS --num_cores 14 --name gp_0.25 --gp_wt_bd 0.25

python main.py --solver ECOS --num_cores 14 --name gp_0.3 --gp_wt_bd 0.3

python main.py --solver ECOS --num_cores 14 --name gp_0.35 --gp_wt_bd 0.35

python main.py --solver ECOS --num_cores 14 --name gp_0.4 --gp_wt_bd 0.4

python main.py --solver ECOS --num_cores 14 --name gp_0.45 --gp_wt_bd 0.45

python main.py --solver ECOS --num_cores 14 --name gp_0.5 --gp_wt_bd 0.5

python main.py --solver ECOS --num_cores 14 --name gp_0.55 --gp_wt_bd 0.55

python main.py --solver ECOS --num_cores 14 --name gp_0.6 --gp_wt_bd 0.6

python main.py --solver ECOS --num_cores 14 --name gp_0.65 --gp_wt_bd 0.65

#End of script
