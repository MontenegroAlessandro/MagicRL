# too much variance
python3 run_cost.py --dir /Users/leonardo/Desktop/Thesis/Data/ --ite 3000 --alg cpg --risk tc --risk_param 0 --reg 0.001 --var 0.05 --pol linear --env hopper --horizon 100 --gamma 1 --lr 0.01 0.1 0.1 --lr_strategy adam --n_workers 8 --batch 100 --n_trials 1 --env_param 2 --c_bounds 25 --l_init 0 --eta_init 0

# many too high bound
python3 run_cost.py --dir /Users/leonardo/Desktop/Thesis/Data/ --ite 3000 --alg cpg --risk tc --risk_param 0 --reg 0.001 --var 0.01 --pol linear --env hopper --horizon 100 --gamma 1 --lr 0.01 0.1 0.1 --lr_strategy adam --n_workers 8 --batch 100 --n_trials 1 --env_param 2 --c_bounds 25 --l_init 0 --eta_init 0

# the bound is not the problem, decrease the learning rates
python3 run_cost.py --dir /Users/leonardo/Desktop/Thesis/Data/ --ite 3000 --alg cpg --risk tc --risk_param 0 --reg 0.001 --var 0.01 --pol linear --env hopper --horizon 100 --gamma 1 --lr 0.01 0.1 0.1 --lr_strategy adam --n_workers 8 --batch 100 --n_trials 1 --env_param 2 --c_bounds 20 --l_init 0 --eta_init 0

# the bound may be too high.
python3 run_cost.py --dir /Users/leonardo/Desktop/Thesis/Data/ --ite 3000 --alg cpg --risk tc --risk_param 0 --reg 0.001 --var 0.01 --pol linear --env hopper --horizon 100 --gamma 1 --lr 0.001 0.01 0.1 --lr_strategy adam --n_workers 10 --batch 100 --n_trials 1 --env_param 2 --c_bounds 25 --l_init 0 --eta_init 0

# good configuration
python3 run_cost.py --dir /Users/leonardo/Desktop/Thesis/Data/ --ite 3000 --alg cpg --risk tc --risk_param 0 --reg 0.001 --var 0.01 --pol linear --env hopper --horizon 100 --gamma 1 --lr 0.001 0.01 0.1 --lr_strategy adam --n_workers 12 --batch 100 --n_trials 1 --env_param 2 --c_bounds 20 --l_init 0 --eta_init 0

# good cofiguration
python3 run_cost.py --dir /Users/leonardo/Desktop/Thesis/Data/ --ite 3000 --alg cpg --risk tc --risk_param 0 --reg 0.0001 --var 1 --pol linear --env hopper --horizon 100 --gamma 1 --lr 0.001 0.01 0.1 --lr_strategy adam --n_workers 12 --batch 100 --n_trials 1 --env_param 2 --c_bounds 100 --l_init 0 --eta_init 0

python3 run_cost.py --dir /Users/leonardo/Desktop/Thesis/Data/ --ite 3000 --alg cpg --risk tc --risk_param 0 --reg 0.0001 --var 1 --pol linear --env hopper --horizon 100 --gamma 1 --lr 0.001 0.01 0.1 --lr_strategy adam --n_workers 12 --batch 100 --n_trials 1 --env_param 2 --c_bounds 75 --l_init 0 --eta_init 0


# Parametrization paper with different variances

taskset -ca 0-11 python3 run_cost.py --dir /data1/montenegro_cesani/ --ite 3000 --alg cpg --risk tc --risk_param 0 --reg 0.0001 --var 0.0001 --pol linear --env swimmer --horizon 100 --gamma 1 --lr 0.001 0.01 0.1 --lr_strategy adam --n_workers 12 --batch 100 --n_trials 1 --env_param 2 --c_bounds 50 --l_init 0 --eta_init 0
taskset -ca 0-11 python3 run_cost.py --dir /data1/montenegro_cesani/ --ite 3000 --alg cpg --risk tc --risk_param 0 --reg 0.0001 --var 0.0005 --pol linear --env swimmer --horizon 100 --gamma 1 --lr 0.001 0.01 0.1 --lr_strategy adam --n_workers 12 --batch 100 --n_trials 1 --env_param 2 --c_bounds 50 --l_init 0 --eta_init 0
taskset -ca 0-11 python3 run_cost.py --dir /data1/montenegro_cesani/ --ite 3000 --alg cpg --risk tc --risk_param 0 --reg 0.0001 --var 0.00075 --pol linear --env swimmer --horizon 100 --gamma 1 --lr 0.001 0.01 0.1 --lr_strategy adam --n_workers 12 --batch 100 --n_trials 1 --env_param 2 --c_bounds 50 --l_init 0 --eta_init 0
taskset -ca 0-11 python3 run_cost.py --dir /data1/montenegro_cesani/ --ite 3000 --alg cpg --risk tc --risk_param 0 --reg 0.0001 --var 0.001 --pol linear --env swimmer --horizon 100 --gamma 1 --lr 0.001 0.01 0.1 --lr_strategy adam --n_workers 12 --batch 100 --n_trials 1 --env_param 2 --c_bounds 50 --l_init 0 --eta_init 0
taskset -ca 0-11 python3 run_cost.py --dir /data1/montenegro_cesani/ --ite 3000 --alg cpg --risk tc --risk_param 0 --reg 0.0001 --var 0.005 --pol linear --env swimmer --horizon 100 --gamma 1 --lr 0.001 0.01 0.1 --lr_strategy adam --n_workers 12 --batch 100 --n_trials 1 --env_param 2 --c_bounds 50 --l_init 0 --eta_init 0
taskset -ca 0-11 python3 run_cost.py --dir /data1/montenegro_cesani/ --ite 3000 --alg cpg --risk tc --risk_param 0 --reg 0.0001 --var 0.0075 --pol linear --env swimmer --horizon 100 --gamma 1 --lr 0.001 0.01 0.1 --lr_strategy adam --n_workers 12 --batch 100 --n_trials 1 --env_param 2 --c_bounds 50 --l_init 0 --eta_init 0
taskset -ca 0-11 python3 run_cost.py --dir /data1/montenegro_cesani/ --ite 3000 --alg cpg --risk tc --risk_param 0 --reg 0.0001 --var 0.01 --pol linear --env swimmer --horizon 100 --gamma 1 --lr 0.001 0.01 0.1 --lr_strategy adam --n_workers 12 --batch 100 --n_trials 1 --env_param 2 --c_bounds 0 --l_init 0 --eta_init 0
taskset -ca 0-11 python3 run_cost.py --dir /data1/montenegro_cesani/ --ite 3000 --alg cpg --risk tc --risk_param 0 --reg 0.0001 --var 0.05 --pol linear --env swimmer --horizon 100 --gamma 1 --lr 0.001 0.01 0.1 --lr_strategy adam --n_workers 12 --batch 100 --n_trials 1 --env_param 2 --c_bounds 0 --l_init 0 --eta_init 0
taskset -ca 0-11 python3 run_cost.py --dir /data1/montenegro_cesani/ --ite 3000 --alg cpg --risk tc --risk_param 0 --reg 0.0001 --var 0.075 --pol linear --env swimmer --horizon 100 --gamma 1 --lr 0.001 0.01 0.1 --lr_strategy adam --n_workers 12 --batch 100 --n_trials 1 --env_param 2 --c_bounds 00 --l_init 0 --eta_init 0

