#!/bin/sh

### Split 1 ###

python main.py --num_cores 8 --T_inner 1000 --T 5 --eta 0.15 --eta_inner 0.65 --gamma_2 0.05 --constraint dp --onlyh0 y --B 0.25 --dataset adult --name final_dp_adult

python main.py --num_cores 8 --T_inner 1000 --T 5 --eta 0.15 --eta_inner 0.6 --gamma_2 0.05 --constraint dp --onlyh0 y --B 0.21 --dataset lawschool --name final_dp_lawschool

python main.py --num_cores 8 --T_inner 1000 --T 5 --eta 0.15 --eta_inner 0.55 --gamma_2 0.05 --constraint dp --onlyh0 y --B 0.275 --dataset communities --name final_dp_communities

python main.py --num_cores 8 --T_inner 1000 --T 5 --eta 0.15 --eta_inner 0.6 --gamma_2 0.05 --constraint dp --onlyh0 y --B 0.05 --dataset compas --name final_dp_compas

### Split 2 ###

python main.py --num_cores 8 --T_inner 1000 --T 5 --eta 0.15 --eta_inner 0.65 --gamma_2 0.05 --constraint dp --onlyh0 y --B 0.25 --dataset adult --name final_dp_adult --split 2

python main.py --num_cores 8 --T_inner 1000 --T 5 --eta 0.15 --eta_inner 0.6 --gamma_2 0.05 --constraint dp --onlyh0 y --B 0.21 --dataset lawschool --name final_dp_lawschool --split 2

python main.py --num_cores 8 --T_inner 1000 --T 5 --eta 0.15 --eta_inner 0.55 --gamma_2 0.05 --constraint dp --onlyh0 y --B 0.275 --dataset communities --name final_dp_communities --split 2

python main.py --num_cores 8 --T_inner 1000 --T 5 --eta 0.15 --eta_inner 0.6 --gamma_2 0.05 --constraint dp --onlyh0 y --B 0.05 --dataset compas --name final_dp_compas --split 2


### Split 3 ###

python main.py --num_cores 8 --T_inner 1000 --T 5 --eta 0.15 --eta_inner 0.65 --gamma_2 0.05 --constraint dp --onlyh0 y --B 0.25 --dataset adult --name final_dp_adult --split 3

python main.py --num_cores 8 --T_inner 1000 --T 5 --eta 0.15 --eta_inner 0.6 --gamma_2 0.05 --constraint dp --onlyh0 y --B 0.21 --dataset lawschool --name final_dp_lawschool --split 3

python main.py --num_cores 8 --T_inner 1000 --T 5 --eta 0.15 --eta_inner 0.55 --gamma_2 0.05 --constraint dp --onlyh0 y --B 0.275 --dataset communities --name final_dp_communities --split 3

python main.py --num_cores 8 --T_inner 1000 --T 5 --eta 0.15 --eta_inner 0.6 --gamma_2 0.05 --constraint dp --onlyh0 y --B 0.05 --dataset compas --name final_dp_compas --split 3

### Split 4 ###

python main.py --num_cores 8 --T_inner 1000 --T 5 --eta 0.15 --eta_inner 0.65 --gamma_2 0.05 --constraint dp --onlyh0 y --B 0.25 --dataset adult --name final_dp_adult --split 4

python main.py --num_cores 8 --T_inner 1000 --T 5 --eta 0.15 --eta_inner 0.6 --gamma_2 0.05 --constraint dp --onlyh0 y --B 0.21 --dataset lawschool --name final_dp_lawschool --split 4

python main.py --num_cores 8 --T_inner 1000 --T 5 --eta 0.15 --eta_inner 0.55 --gamma_2 0.05 --constraint dp --onlyh0 y --B 0.275 --dataset communities --name final_dp_communities --split 4

python main.py --num_cores 8 --T_inner 1000 --T 5 --eta 0.15 --eta_inner 0.6 --gamma_2 0.05 --constraint dp --onlyh0 y --B 0.05 --dataset compas --name final_dp_compas --split 4

### Split 5 ###

python main.py --num_cores 8 --T_inner 1000 --T 5 --eta 0.15 --eta_inner 0.65 --gamma_2 0.05 --constraint dp --onlyh0 y --B 0.25 --dataset adult --name final_dp_adult --split 5

python main.py --num_cores 8 --T_inner 1000 --T 5 --eta 0.15 --eta_inner 0.6 --gamma_2 0.05 --constraint dp --onlyh0 y --B 0.21 --dataset lawschool --name final_dp_lawschool --split 5

python main.py --num_cores 8 --T_inner 1000 --T 5 --eta 0.15 --eta_inner 0.55 --gamma_2 0.05 --constraint dp --onlyh0 y --B 0.275 --dataset communities --name final_dp_communities --split 5

python main.py --num_cores 8 --T_inner 1000 --T 5 --eta 0.15 --eta_inner 0.6 --gamma_2 0.05 --constraint dp --onlyh0 y --B 0.05 --dataset compas --name final_dp_compas --split 5
