#!/bin/bash

python3 run.py --alg off_pg --pol "nn" --lr 0.0001 --env half_cheetah --n_workers 10 --var 0.1 --horizon 100 --ite 2500 --window_length 8 --weight_type "MIS" --n_trials 10 --clip 0 --batch 40 --dir results/MIS/

python3 run.py --alg off_pg --pol "nn" --lr 0.001 --env swimmer --n_workers 10 --var 0.1 --horizon 100 --ite 5000 --window_length 4 --weight_type "MIS" --n_trials 5 --clip 0 --batch 20 --dir results/MIS/