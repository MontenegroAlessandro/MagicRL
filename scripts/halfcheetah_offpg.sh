#!/bin/bash
# First experiment with off_pg with window length 8
python3 run.py --alg off_pg --env half_cheetah --n_workers 10 --var 0.1 --horizon 100 --ite 2000 --window_length 8 --n_trials 5 --clip 0 --dir results/cheetah/

# Second experiment with pg (using same parameters)
python3 run.py --alg pg --env half_cheetah --n_workers 10 --var 0.1 --horizon 100 --ite 2000 --n_trials 5 --clip 0 --dir results/cheetah/

# third experiment with off_pg with window length 16
python3 run.py --alg off_pg --env half_cheetah --n_workers 10 --var 0.1 --horizon 100 --ite 2000 --window_length 16 --n_trials 5 --clip 0 --dir results/cheetah/

# fourth experiment with off_pg with window length 32
python3 run.py --alg off_pg --env half_cheetah --n_workers 10 --var 0.1 --horizon 100 --ite 2000 --window_length 32 --n_trials 5 --clip 0 --dir results/cheetah/

# fifth experiment with off_pg with batch size 50
python3 run.py --alg off_pg --env half_cheetah --n_workers 10 --var 0.1 --horizon 100 --ite 4000 --window_length 16 --n_trials 5 --clip 0 --batch 50 --dir results/cheetah/

# sixth experiment with off_pg with batch size 25
python3 run.py --alg off_pg --env half_cheetah --n_workers 10 --var 0.1 --horizon 100 --ite 8000 --window_length 16 --n_trials 5 --clip 0 --batch 25 --dir results/cheetah/

# seventh experiment with off_pg with batch size 10
python3 run.py --alg off_pg --env half_cheetah --n_workers 10 --var 0.1 --horizon 100 --ite 20000 --window_length 16 --n_trials 5 --clip 0 --batch 10 --dir results/cheetah/