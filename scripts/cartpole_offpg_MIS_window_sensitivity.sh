python3 run.py --alg off_pg --env cartpole --n_workers 10 --var 0.3 --horizon 200 --ite 1000 --window_length 2 --n_trials 10 --clip 0 --batch 5 --weight_type 'MIS' --lr 0.01 --dir results/cartpole/window_sensitivity/

python3 run.py --alg off_pg --env cartpole --n_workers 10 --var 0.3 --horizon 200 --ite 1000 --window_length 4 --n_trials 10 --clip 0 --batch 5 --weight_type 'MIS' --lr 0.01 --dir results/cartpole/window_sensitivity/

python3 run.py --alg off_pg --env cartpole --n_workers 10 --var 0.3 --horizon 200 --ite 1000 --window_length 8 --n_trials 10 --clip 0 --batch 5 --weight_type 'MIS' --lr 0.01 --dir results/cartpole/window_sensitivity/

python3 run.py --alg off_pg --env cartpole --n_workers 10 --var 0.3 --horizon 200 --ite 1000 --window_length 16 --n_trials 10 --clip 0 --batch 5 --weight_type 'MIS' --lr 0.01 --dir results/cartpole/window_sensitivity/

python3 run.py --alg off_pg --env cartpole --n_workers 10 --var 0.3 --horizon 200 --ite 1000 --window_length 32 --n_trials 10 --clip 0 --batch 5 --weight_type 'MIS' --lr 0.01 --dir results/cartpole/window_sensitivity/