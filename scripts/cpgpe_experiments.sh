python3 run_cost.py --dir /data1/montenegro_cesani/--ite 3000 --alg cpgpe --risk tc --risk_param 0 --reg 0.001 --var 0.01 --pol linear --env hopper --horizon 100 --gamma 1 --lr 0.01 0.1 0.1 --lr_strategy adam --n_workers 8 --batch 100 --n_trials 5 --env_param 2 --c_bounds 25 --l_init 0 --eta_init 0
python3 run_cost.py --dir /data1/montenegro_cesani/ --ite 3000 --alg cpgpe --risk tc --risk_param 0 --reg 0.001 --var 0.1 --pol linear --env hopper --horizon 100 --gamma 1 --lr 0.01 0.1 0.1 --lr_strategy adam --n_workers 8 --batch 100 --n_trials 5 --env_param 2 --c_bounds 25 --l_init 0 --eta_init 0
python3 run_cost.py --dir /data1/montenegro_cesani/ --ite 3000 --alg cpgpe --risk tc --risk_param 0 --reg 0.001 --var 1.0 --pol linear --env hopper --horizon 100 --gamma 1 --lr 0.01 0.1 0.1 --lr_strategy adam --n_workers 8 --batch 100 --n_trials 5 --env_param 2 --c_bounds 25 --l_init 0 --eta_init 0
python3 run_cost.py --dir /data1/montenegro_cesani/ --ite 3000 --alg cpgpe --risk tc --risk_param 0 --reg 0.001 --var 10.0 --pol linear --env hopper --horizon 100 --gamma 1 --lr 0.01 0.1 0.1 --lr_strategy adam --n_workers 8 --batch 100 --n_trials 5 --env_param 2 --c_bounds 25 --l_init 0 --eta_init 0
python3 run_cost.py --dir /data1/montenegro_cesani/ --ite 3000 --alg cpgpe --risk tc --risk_param 0 --reg 0.001 --var 100.0 --pol linear --env hopper --horizon 100 --gamma 1 --lr 0.01 0.1 0.1 --lr_strategy adam --n_workers 8 --batch 100 --n_trials 5 --env_param 2 --c_bounds 25 --l_init 0 --eta_init 0


# Commands for the server:
taskset -ca 6-17 python3 run_cost.py --dir /data1/montenegro_cesani/ --ite 3000 --alg cpgpe --risk tc --risk_param 0 --reg 0.0001 --var 0.001 --pol linear --env swimmer --horizon 100 --gamma 1 --lr 0.001 0.01 0.1 --lr_strategy adam --n_workers 12 --batch 100 --n_trials 5 --env_param 2 --c_bounds 5 --l_init 0 --eta_init 0

taskset -ca 5-16 python3 run_cost.py --dir /data1/montenegro_cesani/ --ite 3000 --alg cpgpe --risk tc --risk_param 0 --reg 0.0001 --var 0.05 --pol linear --env swimmer --horizon 100 --gamma 1 --lr 0.001 0.01 0.1 --lr_strategy adam --n_workers 12 --batch 100 --n_trials 5 --env_param 2 --c_bounds 50 --l_init 0 --eta_init 0

taskset -ca 5-16 python3 run_cost.py --dir /data1/montenegro_cesani/ --ite 3000 --alg cpgpe --risk tc --risk_param 0 --reg 0.0001 --var 0.075 --pol linear --env swimmer --horizon 100 --gamma 1 --lr 0.001 0.01 0.1 --lr_strategy adam --n_workers 12 --batch 100 --n_trials 5 --env_param 2 --c_bounds 50 --l_init 0 --eta_init 0

taskset -ca 0-11 python3 run_cost.py --dir /data1/montenegro_cesani/ --ite 3000 --alg cpgpe --risk tc --risk_param 0 --reg 0.0001 --var 0.0001 --pol linear --env swimmer --horizon 100 --gamma 1 --lr 0.001 0.01 0.1 --lr_strategy adam --n_workers 12 --batch 100 --n_trials 5 --env_param 2 --c_bounds 50 --l_init 0 --eta_init 0

taskset -ca 0-11 python3 run_cost.py --dir /data1/montenegro_cesani/ --ite 3000 --alg cpgpe --risk tc --risk_param 0 --reg 0.0001 --var 0.0005 --pol linear --env swimmer --horizon 100 --gamma 1 --lr 0.001 0.01 0.1 --lr_strategy adam --n_workers 12 --batch 100 --n_trials 5 --env_param 2 --c_bounds 50 --l_init 0 --eta_init 0

taskset -ca 0-11 python3 run_cost.py --dir /data1/montenegro_cesani/ --ite 3000 --alg cpgpe --risk tc --risk_param 0 --reg 0.0001 --var 0.00075 --pol linear --env swimmer --horizon 100 --gamma 1 --lr 0.001 0.01 0.1 --lr_strategy adam --n_workers 12 --batch 100 --n_trials 5 --env_param 2 --c_bounds 50 --l_init 0 --eta_init 0


# robot world
taskset -ca 12-23 python3 run_cost.py --dir /data1/montenegro_cesani/ding/ --ite 10000 --alg cpgpe --risk tc --risk_param 0 --reg 0.0001 --var 0.000001 --pol linear --env robot_world --horizon 400 --gamma 0.99 --lr 0.000005 0.05 0.01 --lr_strategy adam --n_workers 12 --batch 100 --n_trials 5 --env_param 2 --c_bounds 1000 --l_init 0 --eta_init 0 --clip 0 --deterministic 1

# Gridwold
taskset -ca 0-17 python3 run_cost.py --dir /data1/montenegro_cesani/ --ite 12000 --alg cpgpe --risk tc --risk_param 0 --reg 0.0001 --var 0.025 --pol gw_pol --env gw_c --horizon 100  --gamma 1 --lr 0.0025 0.25 0.01 --lr_strategy adam --n_workers 18 --batch 100 --n_trials 1 --env_param 2 --c_bounds 3 --l_init 0 --eta_init 0.0 --clip 0 --deterministic 1 --l_init 0.0


