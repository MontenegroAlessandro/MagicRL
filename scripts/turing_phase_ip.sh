cd ..

# taskset -ca 0-35 python run_phases.py \
#     --learn 1 \
#     --sigma_lr 0.01 \
#     --sigma_lr_strategy adam \
#     --phases 5000 \
#     --sigma_exponent 1 \
#     --sigma_init 1 \
#     --sigma_param exp \
#     --dir /data2/alessandro/urgent/swimmer/ \
#     --ite 1 \
#     --alg pg \
#     --var 1.0 \
#     --pol linear \
#     --env swimmer \
#     --horizon 200 \
#     --gamma 1.0 \
#     --lr 0.01 \
#     --lr_strategy adam \
#     --n_workers 36 \
#     --batch 100 \
#     --clip 0 \
#     --n_trials 5 

# taskset -ca 0-35 python run_phases.py \
#     --learn 0 \
#     --sigma_lr 0.01 \
#     --sigma_lr_strategy adam \
#     --phases 10 \
#     --sigma_exponent 1 \
#     --sigma_init 1 \
#     --sigma_param exp \
#     --dir /data2/alessandro/urgent/swimmer/ \
#     --ite 500 \
#     --alg pg \
#     --var 1.0 \
#     --pol linear \
#     --env swimmer \
#     --horizon 200 \
#     --gamma 1.0 \
#     --lr 0.01 \
#     --lr_strategy adam \
#     --n_workers 36 \
#     --batch 100 \
#     --clip 0 \
#     --n_trials 5

taskset -ca 0-35 python run_phases.py \
    --learn 0 \
    --sigma_lr 0.01 \
    --sigma_lr_strategy adam \
    --phases 25 \
    --sigma_exponent 1 \
    --sigma_init 1 \
    --sigma_param exp \
    --dir /data2/alessandro/urgent/swimmer/ \
    --ite 200 \
    --alg pg \
    --var 1.0 \
    --pol linear \
    --env swimmer \
    --horizon 200 \
    --gamma 1.0 \
    --lr 0.01 \
    --lr_strategy adam \
    --n_workers 36 \
    --batch 100 \
    --clip 0 \
    --n_trials 5

taskset -ca 0-35 python run_phases.py \
    --learn 0 \
    --sigma_lr 0.01 \
    --sigma_lr_strategy adam \
    --phases 5000 \
    --sigma_exponent 0.5 \
    --sigma_init 1 \
    --sigma_param exp \
    --dir /data2/alessandro/urgent/swimmer/ \
    --ite 1 \
    --alg pg \
    --var 1.0 \
    --pol linear \
    --env swimmer \
    --horizon 200 \
    --gamma 1.0 \
    --lr 0.01 \
    --lr_strategy adam \
    --n_workers 36 \
    --batch 100 \
    --clip 0 \
    --n_trials 5

# taskset -ca 0-35 python run.py \
#     --dir /data2/alessandro/urgent/swimmer/ \
#     --ite 5000 \
#     --alg pg \
#     --var 0.014 \
#     --pol linear \
#     --env swimmer \
#     --horizon 200 \
#     --gamma 1.0 \
#     --lr 0.01 \
#     --lr_strategy adam \
#     --n_workers 36 \
#     --batch 100 \
#     --clip 0 \
#     --n_trials 5

# taskset -ca 0-35 python run.py \
#     --dir /data2/alessandro/urgent/swimmer/ \
#     --ite 5000 \
#     --alg pg \
#     --var 1 \
#     --pol linear \
#     --env swimmer \
#     --horizon 200 \
#     --gamma 1.0 \
#     --lr 0.01 \
#     --lr_strategy adam \
#     --n_workers 36 \
#     --batch 100 \
#     --clip 0 \
#     --n_trials 5

# taskset -ca 0-35 python run.py \
#     --dir /data2/alessandro/urgent/swimmer/ \
#     --ite 5000 \
#     --alg pg \
#     --var 0.5 \
#     --pol linear \
#     --env swimmer \
#     --horizon 200 \
#     --gamma 1.0 \
#     --lr 0.01 \
#     --lr_strategy adam \
#     --n_workers 36 \
#     --batch 100 \
#     --clip 0 \
#     --n_trials 5
