#!/bin/sh

### Split 1 ###

python main.py --T_inner 1000 --T 25 --eta 0.15 --eta_inner 0.65 --gamma_2 0.05 --constraint dp --B 0.25 --dataset adult --name ensemble_adult1000

python main.py --T_inner 1000 --T 25 --eta 0.15 --eta_inner 0.6 --gamma_2 0.05 --constraint dp --B 0.225 --dataset lawschool --name ensemble_lawschool1000

python main.py --T_inner 1000 --T 25 --eta 0.15 --eta_inner 0.55 --gamma_2 0.05 --constraint dp --B 0.275 --dataset communities --name ensemble_communities1000

### Split 2 ###

python main.py --T_inner 1000 --T 25 --eta 0.15 --eta_inner 0.65 --gamma_2 0.05 --constraint dp --B 0.25 --dataset adult --name ensemble_adult1000 --split 2

python main.py --T_inner 1000 --T 25 --eta 0.15 --eta_inner 0.6 --gamma_2 0.05 --constraint dp --B 0.225 --dataset lawschool --name ensemble_lawschool1000 --split 2

python main.py --T_inner 1000 --T 25 --eta 0.15 --eta_inner 0.55 --gamma_2 0.05 --constraint dp --B 0.275 --dataset communities --name ensemble_communities1000 --split 2

### Split 3 ###

python main.py --T_inner 1000 --T 25 --eta 0.15 --eta_inner 0.65 --gamma_2 0.05 --constraint dp --B 0.25 --dataset adult --name ensemble_adult1000 --split 3

python main.py --T_inner 1000 --T 25 --eta 0.15 --eta_inner 0.6 --gamma_2 0.05 --constraint dp --B 0.225 --dataset lawschool --name ensemble_lawschool1000 --split 3

python main.py --T_inner 1000 --T 25 --eta 0.15 --eta_inner 0.55 --gamma_2 0.05 --constraint dp --B 0.275 --dataset communities --name ensemble_communities1000 --split 3