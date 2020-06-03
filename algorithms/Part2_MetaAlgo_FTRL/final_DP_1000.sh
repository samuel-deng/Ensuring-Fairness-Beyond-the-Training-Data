#!/bin/sh

### Split 1 ###

python main.py --num_cores 8 --T_inner 1000 --T 1--eta 0.15 --eta_inner 0.55 --gamma_2 0.05 --constraint dp --B 0.275 --dataset communities --name final_dp_communities --no_output y

### Split 1 ###

python main.py --num_cores 8 --T_inner 1000 --T 1 --eta 0.15 --eta_inner 0.65 --gamma_2 0.05 --constraint dp --B 0.25 --dataset adult --name final_dp_adult

python main.py --num_cores 8 --T_inner 1000 --T 1 --eta 0.15 --eta_inner 0.6 --gamma_2 0.05 --constraint dp --B 0.21 --dataset lawschool --name final_dp_lawschool

python main.py --num_cores 8 --T_inner 1000 --T 1 --eta 0.15 --eta_inner 0.55 --gamma_2 0.05 --constraint dp --B 0.275 --dataset communities --name final_dp_communities

### Split 1 ###

python main.py --num_cores 8 --T_inner 1000 --T 1 --eta 0.15 --eta_inner 0.65 --gamma_2 0.05 --constraint dp --B 0.25 --dataset adult --name final_dp_adult

python main.py --num_cores 8 --T_inner 1000 --T 1 --eta 0.15 --eta_inner 0.6 --gamma_2 0.05 --constraint dp --B 0.21 --dataset lawschool --name final_dp_lawschool

python main.py --num_cores 8 --T_inner 1000 --T 1 --eta 0.15 --eta_inner 0.55 --gamma_2 0.05 --constraint dp --B 0.275 --dataset communities --name final_dp_communities

### Split 2 ###

python main.py --num_cores 8 --T_inner 1000 --T 1 --eta 0.15 --eta_inner 0.65 --gamma_2 0.05 --constraint dp --B 0.25 --dataset adult --name final_dp_adult

python main.py --num_cores 8 --T_inner 1000 --T 1 --eta 0.15 --eta_inner 0.6 --gamma_2 0.05 --constraint dp --B 0.21 --dataset lawschool --name final_dp_lawschool

python main.py --num_cores 8 --T_inner 1000 --T 1 --eta 0.15 --eta_inner 0.55 --gamma_2 0.05 --constraint dp --B 0.275 --dataset communities --name final_dp_communities