python3 run.py --alg off_pg --env swimmer --n_workers 10 --var 0.3 --horizon 200 --ite 25000 --window_length 8 --n_trials 5 --clip 0 --lr 0.001 --batch 20 --pol 'nn' --weight_type 'MIS' --dir results/swimmer_MIS_nn/batch_comparison/

python3 run.py --alg pg --env swimmer --n_workers 10 --var 0.3 --horizon 200 --ite 5000 --n_trials 5 --clip 0 --lr 0.001 --batch 100 --pol 'nn' --dir results/swimmer_MIS_nn/batch_comparison/

python3 run.py --alg off_pg --env swimmer --n_workers 10 --var 0.3 --horizon 200 --ite 10000 --window_length 8 --n_trials 5 --clip 0 --lr 0.001 --batch 50 --pol 'nn' --weight_type 'MIS' --dir results/swimmer_MIS_nn/batch_comparison/