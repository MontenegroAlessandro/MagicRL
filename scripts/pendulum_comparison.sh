python3 run.py --alg pg --env pendulum --n_workers 10 --var 0.1 --horizon 200 --ite 2000 --n_trials 5 --clip 0 --batch 5 --dir results/pendulum_comparison_5batch/

python3 run.py --alg off_pg --env pendulum --n_workers 10 --var 0.1 --horizon 200 --ite 2000 --window_length 16 --n_trials 5 --clip 0 --batch 5 --weight_type 'BH' --dir results/pendulum_comparison_5batch/

python3 run.py --alg off_pg --env pendulum --n_workers 10 --var 0.1 --horizon 200 --ite 2000 --window_length 16 --n_trials 5 --clip 0 --batch 5 --weight_type 'MIS' --dir results/pendulum_comparison_5batch/