#!/bin/sh

### Split 1 ###

python main.py --T_inner 500 --T 25 --eta 0.2 --eta_inner 0.55 --gamma_2 0.05 --constraint dp --B 0.15 --dataset adult --name ensemble_adult500

python main.py --T_inner 500 --T 25 --eta 0.2 --eta_inner 0.55 --gamma_2 0.05 --constraint dp --B 0.15 --dataset lawschool --name ensemble_lawschool500

python main.py --T_inner 500 --T 25 --eta 0.2 --eta_inner 0.5 --gamma_2 0.05 --constraint dp --B 0.165 --dataset communities --name ensemble_communities500

### Split 2 ###

python main.py --T_inner 500 --T 25 --eta 0.2 --eta_inner 0.55 --gamma_2 0.05 --constraint dp --B 0.15 --dataset adult --name ensemble_adult500 --split 2

python main.py --T_inner 500 --T 25 --eta 0.2 --eta_inner 0.55 --gamma_2 0.05 --constraint dp --B 0.15 --dataset lawschool --name ensemble_lawschool500 --split 2

python main.py --T_inner 500 --T 25 --eta 0.2 --eta_inner 0.5 --gamma_2 0.05 --constraint dp --B 0.165 --dataset communities --name ensemble_communities500 --split 2

### Split 3 ###

python main.py --T_inner 500 --T 25 --eta 0.2 --eta_inner 0.55 --gamma_2 0.05 --constraint dp --B 0.15 --dataset adult --name ensemble_adult500 --split 3

python main.py --T_inner 500 --T 25 --eta 0.2 --eta_inner 0.55 --gamma_2 0.05 --constraint dp --B 0.15 --dataset lawschool --name ensemble_lawschool500 --split 3

python main.py --T_inner 500 --T 25 --eta 0.2 --eta_inner 0.5 --gamma_2 0.05 --constraint dp --B 0.165 --dataset communities --name ensemble_communities500 --split 3