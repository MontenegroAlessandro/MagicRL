#!/bin/bash
# First experiment with off_pg
#python3 run.py --alg off_pg --env pendulum --n_workers 10 --var 0.1 --horizon 200 --ite 800 --window_length 8 --n_trials 5 --clip 0 --dir results/

# Second experiment with pg (using same parameters)
#python3 run.py --alg pg --env pendulum --n_workers 10 --var 0.1 --horizon 200 --ite 800 --n_trials 5 --clip 0 --dir results/

# third experiment with off_pg
#python3 run.py --alg off_pg --env pendulum --n_workers 10 --var 0.1 --horizon 200 --ite 800 --window_length 16 --n_trials 5 --clip 0 --dir results/

python3 run.py --alg off_pg --env pendulum --n_workers 10 --var 0.1 --horizon 200 --ite 800 --window_length 32 --n_trials 5 --clip 0 --batch 100 --dir results/