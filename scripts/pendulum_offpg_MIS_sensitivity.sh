python3 run.py --alg off_pg --env pendulum --n_workers 10 --var 0.3 --horizon 200 --ite 2500 --window_length 4 --n_trials 5 --clip 0 --batch 5 --weight_type 'MIS' --lr 0.01 --dir results/pendulum_MIS/sensitivity_trajectory2/

python3 run.py --alg pg --env pendulum --n_workers 10 --var 0.3 --horizon 200 --ite 2500 --n_trials 5 --clip 0 --batch 20 --lr 0.01 --dir results/pendulum_MIS/sensitivity_trajectory2/

python3 run.py --alg off_pg --env pendulum --n_workers 10 --var 0.3 --horizon 200 --ite 1250 --window_length 4 --n_trials 5 --clip 0 --batch 10 --weight_type 'MIS' --lr 0.01 --dir results/pendulum_MIS/sensitivity_trajectory2/

python3 run.py --alg pg --env pendulum --n_workers 10 --var 0.3 --horizon 200 --ite 1250 --n_trials 5 --clip 0 --batch 40 --lr 0.01 --dir results/pendulum_MIS/sensitivity_trajectory2/

python3 run.py --alg off_pg --env pendulum --n_workers 10 --var 0.3 --horizon 200 --ite 500 --window_length 4 --n_trials 5 --clip 0 --batch 25 --weight_type 'MIS' --lr 0.01 --dir results/pendulum_MIS/sensitivity_trajectory2/

python3 run.py --alg pg --env pendulum --n_workers 10 --var 0.3 --horizon 200 --ite 500 --n_trials 5 --clip 0 --batch 100 --lr 0.01 --dir results/pendulum_MIS/sensitivity_trajectory2/

