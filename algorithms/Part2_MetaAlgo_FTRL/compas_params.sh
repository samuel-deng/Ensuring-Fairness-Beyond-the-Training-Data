#!/bin/sh


python main.py --num_cores 8 --no_output y --T_inner 1000 --T 1 --eta 0.15 --eta_inner 0.6 --gamma_2 0.05 --constraint dp --B 0.0125 --dataset compas --onlyh0 y --verbose y

python main.py --num_cores 8 --no_output y --T_inner 1000 --T 1 --eta 0.15 --eta_inner 0.6 --gamma_2 0.05 --constraint dp --B 0.025 --dataset compas --onlyh0 y --verbose y

python main.py --num_cores 8 --no_output y --T_inner 1000 --T 1 --eta 0.15 --eta_inner 0.6 --gamma_2 0.05 --constraint dp --B 0.0375 --dataset compas --onlyh0 y --verbose y

python main.py --num_cores 8 --no_output y --T_inner 1000 --T 1 --eta 0.15 --eta_inner 0.6 --gamma_2 0.05 --constraint dp --B 0.05 --dataset compas --onlyh0 y --verbose y

python main.py --num_cores 8 --no_output y --T_inner 1000 --T 1 --eta 0.15 --eta_inner 0.6 --gamma_2 0.05 --constraint dp --B 0.075 --dataset compas --onlyh0 y --verbose y

python main.py --num_cores 8 --no_output y --T_inner 1000 --T 1 --eta 0.15 --eta_inner 0.6 --gamma_2 0.05 --constraint dp --B 0.1 --dataset compas --onlyh0 y --verbose y

python main.py --num_cores 8 --no_output y --T_inner 1000 --T 1 --eta 0.15 --eta_inner 0.6 --gamma_2 0.05 --constraint dp --B 0.125 --dataset compas --onlyh0 y --verbose y

python main.py --num_cores 8 --no_output y --T_inner 1000 --T 1 --eta 0.15 --eta_inner 0.6 --gamma_2 0.05 --constraint dp --B 0.15 --dataset compas --onlyh0 y --verbose y

python main.py --num_cores 8 --no_output y --T_inner 1000 --T 1 --eta 0.15 --eta_inner 0.6 --gamma_2 0.05 --constraint dp --B 0.175 --dataset compas --onlyh0 y --verbose y


