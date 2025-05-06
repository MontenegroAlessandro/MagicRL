python3 run.py --alg off_pg --env cartpole --n_workers 10 --var 0.3 --horizon 200 --ite 1000 --window_length 8 --n_trials 5 --clip 0 --batch 10 --weight_type 'MIS' --lr 0.01 --dir results/cartpole/baselines/

python3 run.py --alg pg --env cartpole --n_workers 10 --var 0.3 --horizon 200 --ite 1000 --n_trials 5 --clip 0 --batch 10 --lr 0.01 --dir results/cartpole/baselines/