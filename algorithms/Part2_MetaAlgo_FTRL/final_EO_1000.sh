#!/bin/sh
#

### Split 5 ###

python main.py --num_cores 8 --T_inner 500 --T 10 --eta 0.15 --eta_inner 0.5 --gamma_2 0.05 --constraint eo --B 0.11 --dataset adult --name final_eo_adult --split 5

python main.py --num_cores 8 --T_inner 500 --T 10 --eta 0.15 --eta_inner 0.5 --gamma_2 0.05 --constraint eo --B 0.11 --dataset lawschool --name final_eo_lawschool --split 5

python main.py --num_cores 8 --T_inner 500 --T 10 --eta 0.15 --eta_inner 0.5 --gamma_2 0.05 --constraint eo --B 0.11 --dataset communities --name final_eo_communities --split 5

### Split 4 ###

python main.py --num_cores 8 --T_inner 500 --T 10 --eta 0.15 --eta_inner 0.5 --gamma_2 0.05 --constraint eo --B 0.11 --dataset adult --name final_eo_adult --split 4

python main.py --num_cores 8 --T_inner 500 --T 10 --eta 0.15 --eta_inner 0.5 --gamma_2 0.05 --constraint eo --B 0.11 --dataset lawschool --name final_eo_lawschool --split 4

python main.py --num_cores 8 --T_inner 500 --T 10 --eta 0.15 --eta_inner 0.5 --gamma_2 0.05 --constraint eo --B 0.11 --dataset communities --name final_eo_communities --split 4

### Split 3 ###

python main.py --num_cores 8 --T_inner 500 --T 10 --eta 0.15 --eta_inner 0.5 --gamma_2 0.05 --constraint eo --B 0.11 --dataset adult --name final_eo_adult --split 3

python main.py --num_cores 8 --T_inner 500 --T 10 --eta 0.15 --eta_inner 0.5 --gamma_2 0.05 --constraint eo --B 0.11 --dataset lawschool --name final_eo_lawschool --split 3

python main.py --num_cores 8 --T_inner 500 --T 10 --eta 0.15 --eta_inner 0.5 --gamma_2 0.05 --constraint eo --B 0.11 --dataset communities --name final_eo_communities --split 3

### Split 2 ###

python main.py --num_cores 8 --T_inner 500 --T 10 --eta 0.15 --eta_inner 0.5 --gamma_2 0.05 --constraint eo --B 0.11 --dataset adult --name final_eo_adult --split 2

python main.py --num_cores 8 --T_inner 500 --T 10 --eta 0.15 --eta_inner 0.5 --gamma_2 0.05 --constraint eo --B 0.11 --dataset lawschool --name final_eo_lawschool --split 2

python main.py --num_cores 8 --T_inner 500 --T 10 --eta 0.15 --eta_inner 0.5 --gamma_2 0.05 --constraint eo --B 0.11 --dataset communities --name final_eo_communities --split 2

### Split 1 ###

python main.py --num_cores 8 --T_inner 500 --T 10 --eta 0.15 --eta_inner 0.5 --gamma_2 0.05 --constraint eo --B 0.11 --dataset adult --name final_eo_adult

python main.py --num_cores 8 --T_inner 500 --T 10 --eta 0.15 --eta_inner 0.5 --gamma_2 0.05 --constraint eo --B 0.11 --dataset lawschool --name final_eo_lawschool

python main.py --num_cores 8 --T_inner 500 --T 10 --eta 0.15 --eta_inner 0.5 --gamma_2 0.05 --constraint eo --B 0.11 --dataset communities --name final_eo_communities