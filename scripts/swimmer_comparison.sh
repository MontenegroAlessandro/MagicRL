python3 run.py --alg off_pg --env swimmer --n_workers 10 --var 0.1 --horizon 200 --ite 2000 --window_length 16 --n_trials 10 --clip 0 --batch 10 --weight_type 'MIS' --dir results/swimmer_comparison_5batch/

python3 run.py --alg off_pg --env swimmer --n_workers 10 --var 0.1 --horizon 200 --ite 2000 --window_length 16 --n_trials 10 --clip 0 --batch 10 --weight_type 'BH' --dir results/swimmer_comparison_5batch/

python3 run.py --alg pg --env swimmer --n_workers 10 --var 0.1 --horizon 200 --ite 2000 --n_trials 10 --clip 0 --batch 10 --dir results/swimmer_comparison_5batch/