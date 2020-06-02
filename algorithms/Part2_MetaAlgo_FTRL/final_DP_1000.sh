#!/bin/sh

### Split 1 ###

python main.py --num_cores 8 --T_inner 1000 --T 10 --eta 0.15 --eta_inner 0.55 --gamma_2 0.05 --constraint dp --B 0.275 --dataset communities --name final_dp_communities --split 5

### Split 1 ###

python main.py --num_cores 8 --T_inner 1000 --T 10 --eta 0.15 --eta_inner 0.65 --gamma_2 0.05 --constraint dp --B 0.25 --dataset adult --name final_dp_adult --split 4

python main.py --num_cores 8 --T_inner 1000 --T 10 --eta 0.15 --eta_inner 0.6 --gamma_2 0.05 --constraint dp --B 0.21 --dataset lawschool --name final_dp_lawschool --split 4

python main.py --num_cores 8 --T_inner 1000 --T 10 --eta 0.15 --eta_inner 0.55 --gamma_2 0.05 --constraint dp --B 0.275 --dataset communities --name final_dp_communities --split 4

### Split 1 ###

python main.py --num_cores 8 --T_inner 1000 --T 10 --eta 0.15 --eta_inner 0.65 --gamma_2 0.05 --constraint dp --B 0.25 --dataset adult --name final_dp_adult --split 3

python main.py --num_cores 8 --T_inner 1000 --T 10 --eta 0.15 --eta_inner 0.6 --gamma_2 0.05 --constraint dp --B 0.21 --dataset lawschool --name final_dp_lawschool --split 3

python main.py --num_cores 8 --T_inner 1000 --T 10 --eta 0.15 --eta_inner 0.55 --gamma_2 0.05 --constraint dp --B 0.275 --dataset communities --name final_dp_communities --split 3

### Split 2 ###

python main.py --num_cores 8 --T_inner 1000 --T 10 --eta 0.15 --eta_inner 0.65 --gamma_2 0.05 --constraint dp --B 0.25 --dataset adult --name final_dp_adult --split 2

python main.py --num_cores 8 --T_inner 1000 --T 10 --eta 0.15 --eta_inner 0.6 --gamma_2 0.05 --constraint dp --B 0.21 --dataset lawschool --name final_dp_lawschool --split 2

python main.py --num_cores 8 --T_inner 1000 --T 10 --eta 0.15 --eta_inner 0.55 --gamma_2 0.05 --constraint dp --B 0.275 --dataset communities --name final_dp_communities --split 2