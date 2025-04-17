#python3 run.py --alg pg --env pendulum --n_workers 10 --var 0.3 --horizon 200 --ite 1000 --n_trials 5 --clip 0 --lr 0.003 --dir results/pendulum_MIS/window_comparison/

#python3 run.py --alg off_pg --env pendulum --n_workers 10 --var 0.3 --horizon 200 --ite 1000 --window_length 8 --n_trials 5 --clip 0 --lr 0.003 --weight_type 'MIS' --dir results/pendulum_MIS/window_comparison/

#python3 run.py --alg off_pg --env pendulum --n_workers 10 --var 0.3 --horizon 200 --ite 1000 --window_length 4 --n_trials 5 --clip 0 --lr 0.003 --weight_type 'MIS' --dir results/pendulum_MIS/window_comparison/

#python3 run.py --alg off_pg --env pendulum --n_workers 10 --var 0.3 --horizon 200 --ite 1000 --window_length 16 --n_trials 5 --clip 0 --lr 0.003 --weight_type 'MIS' --dir results/pendulum_MIS/window_comparison/



# fourth experiment with off_pg with batch size 50
#python3 run.py --alg off_pg --env pendulum --n_workers 10 --var 0.3 --horizon 200 --ite 2000 --window_length 8 --n_trials 5 --clip 0 --batch 50 --weight_type 'MIS' --lr 0.003 --dir results/pendulum_MIS/batch_comparison/

# fifth experiment with off_pg with batch size 25
#python3 run.py --alg off_pg --env pendulum --n_workers 10 --var 0.3 --horizon 200 --ite 4000 --window_length 8 --n_trials 5 --clip 0 --batch 25 --weight_type 'MIS' --lr 0.003 --dir results/pendulum_MIS/batch_comparison/

# sixth experiment with off_pg with batch size 10
#python3 run.py --alg off_pg --env pendulum --n_workers 10 --var 0.3 --horizon 200 --ite 10000 --window_length 8 --n_trials 5 --clip 0 --batch 10 --weight_type 'MIS' --lr 0.003 --dir results/pendulum_MIS/batch_comparison/

# sixth experiment with off_pg with batch size 5
#python3 run.py --alg off_pg --env pendulum --n_workers 10 --var 0.3 --horizon 200 --ite 20000 --window_length 8 --n_trials 5 --clip 0 --batch 5 --weight_type 'MIS' --lr 0.003 --dir results/pendulum_MIS/batch_comparison/


#python3 run.py --alg off_pg --env pendulum --n_workers 10 --var 0.3 --horizon 200 --ite 2000 --window_length 4 --n_trials 5 --clip 0 --batch 5 --weight_type 'MIS' --lr 0.003 --dir results/pendulum_MIS/window_5_batch/

#python3 run.py --alg off_pg --env pendulum --n_workers 10 --var 0.3 --horizon 200 --ite 2000 --window_length 8 --n_trials 5 --clip 0 --batch 5 --weight_type 'MIS' --lr 0.003 --dir results/pendulum_MIS/window_5_batch/

#python3 run.py --alg off_pg --env pendulum --n_workers 10 --var 0.3 --horizon 200 --ite 2000 --window_length 16 --n_trials 5 --clip 0 --batch 5 --weight_type 'MIS' --lr 0.003 --dir results/pendulum_MIS/window_5_batch/

python3 run.py --alg pg --env pendulum --n_workers 10 --var 0.3 --horizon 200 --ite 2000 --n_trials 5 --clip 0 --batch 5 --lr 0.003 --dir results/pendulum_MIS/window_5_batch/
