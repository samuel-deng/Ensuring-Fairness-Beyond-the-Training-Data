#!/bin/sh

### Split 1 ###

python main.py --num_cores 8 --T_inner 1000 --T 5 --eta 0.15 --eta_inner 0.6 --gamma_2 0.05 --constraint dp --B 0.05 --dataset compas --name final_dp_compas --split 1

### Split 2 ###

python main.py --num_cores 8 --T_inner 1000 --T 5 --eta 0.15 --eta_inner 0.6 --gamma_2 0.05 --constraint dp --B 0.05 --dataset compas --name final_dp_compas --split 2

### Split 3 ###

python main.py --num_cores 8 --T_inner 1000 --T 5 --eta 0.15 --eta_inner 0.6 --gamma_2 0.05 --constraint dp --B 0.05 --dataset compas --name final_dp_compas --split 3

### Split 4 ###

python main.py --num_cores 8 --T_inner 1000 --T 5 --eta 0.15 --eta_inner 0.6 --gamma_2 0.05 --constraint dp --B 0.05 --dataset compas --name final_dp_compas --split 4

### Split 5 ###

python main.py --num_cores 8 --T_inner 1000 --T 5 --eta 0.15 --eta_inner 0.6 --gamma_2 0.05 --constraint dp --B 0.05 --dataset compas --name final_dp_compas --split 5