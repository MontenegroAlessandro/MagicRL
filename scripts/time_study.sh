cd ..

# python3 run.py \
#     --dir /Users/ale/code/PyProjects/results/time_res/ \
#     --ite 500 \
#     --alg pg \
#     --pol linear \
#     --env cartpole \
#     --horizon 200 \
#     --lr 0.01 \
#     --lr_strategy adam \
#     --n_workers 8 \
#     --batch 128 \
#     --clip 0 \
#     --var 0.3 \
#     --n_trials 10

# python3 run.py \
#     --dir /Users/ale/code/PyProjects/results/time_res/ \
#     --ite 1000 \
#     --alg off_pg \
#     --weight_type RPG \
#     --window_length 2 \
#     --pol linear \
#     --env cartpole \
#     --horizon 200 \
#     --lr 0.01 \
#     --lr_strategy adam \
#     --n_workers 8 \
#     --batch 64 \
#     --clip 0 \
#     --var 0.3 \
#     --n_trials 10

# python3 run.py \
#     --dir /Users/ale/code/PyProjects/results/time_res/ \
#     --ite 2000 \
#     --alg off_pg \
#     --weight_type RPG \
#     --window_length 4 \
#     --pol linear \
#     --env cartpole \
#     --horizon 200 \
#     --lr 0.01 \
#     --lr_strategy adam \
#     --n_workers 8 \
#     --batch 32 \
#     --clip 0 \
#     --var 0.3 \
#     --n_trials 10

# python3 run.py \
#     --dir /Users/ale/code/PyProjects/results/time_res/ \
#     --ite 1000 \
#     --alg off_pg \
#     --weight_type RPG \
#     --window_length 8 \
#     --pol linear \
#     --env cartpole \
#     --horizon 200 \
#     --lr 0.01 \
#     --lr_strategy adam \
#     --n_workers 8 \
#     --batch 10 \
#     --clip 0 \
#     --var 0.3 \
#     --n_trials 10

# python3 run.py \
#     --dir /Users/ale/code/PyProjects/results/time_res/ \
#     --ite 1000 \
#     --alg off_pg \
#     --weight_type BH \
#     --window_length 8 \
#     --pol linear \
#     --env cartpole \
#     --horizon 200 \
#     --lr 0.01 \
#     --lr_strategy adam \
#     --n_workers 8 \
#     --batch 10 \
#     --clip 0 \
#     --var 0.3 \
#     --n_trials 10

# python3 run.py \
#     --dir /Users/ale/code/PyProjects/results/time_res/ \
#     --ite 5000 \
#     --alg off_pg \
#     --weight_type BH \
#     --window_length 4 \
#     --pol nn \
#     --env swimmer \
#     --horizon 200 \
#     --lr 0.001 \
#     --lr_strategy adam \
#     --n_workers 8 \
#     --batch 20 \
#     --clip 0 \
#     --var 0.3 \
#     --n_trials 5

# python3 run.py \
#     --dir /Users/ale/code/PyProjects/results/time_res/ \
#     --ite 5000 \
#     --alg off_pg \
#     --weight_type RPG \
#     --window_length 4 \
#     --pol nn \
#     --env swimmer \
#     --horizon 200 \
#     --lr 0.001 \
#     --lr_strategy adam \
#     --n_workers 8 \
#     --batch 20 \
#     --clip 0 \
#     --var 0.3 \
#     --n_trials 5

python3 run.py \
    --dir /Users/ale/code/PyProjects/results/time_res/ \
    --ite 10000 \
    --alg off_pg \
    --weight_type BH \
    --window_length 8 \
    --pol nn \
    --env half_cheetah \
    --horizon 100 \
    --lr 0.0001 \
    --lr_strategy adam \
    --n_workers 8 \
    --batch 40 \
    --clip 0 \
    --var 0.1 \
    --n_trials 10