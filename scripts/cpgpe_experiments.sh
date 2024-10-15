python3 run_cost.py --dir /data1/montenegro_cesani/--ite 3000 --alg cpgpe --risk tc --risk_param 0 --reg 0.001 --var 0.01 --pol linear --env hopper --horizon 100 --gamma 1 --lr 0.01 0.1 0.1 --lr_strategy adam --n_workers 8 --batch 100 --n_trials 5 --env_param 2 --c_bounds 25 --l_init 0 --eta_init 0
python3 run_cost.py --dir /data1/montenegro_cesani/ --ite 3000 --alg cpgpe --risk tc --risk_param 0 --reg 0.001 --var 0.1 --pol linear --env hopper --horizon 100 --gamma 1 --lr 0.01 0.1 0.1 --lr_strategy adam --n_workers 8 --batch 100 --n_trials 5 --env_param 2 --c_bounds 25 --l_init 0 --eta_init 0
python3 run_cost.py --dir /data1/montenegro_cesani/ --ite 3000 --alg cpgpe --risk tc --risk_param 0 --reg 0.001 --var 1.0 --pol linear --env hopper --horizon 100 --gamma 1 --lr 0.01 0.1 0.1 --lr_strategy adam --n_workers 8 --batch 100 --n_trials 5 --env_param 2 --c_bounds 25 --l_init 0 --eta_init 0
python3 run_cost.py --dir /data1/montenegro_cesani/ --ite 3000 --alg cpgpe --risk tc --risk_param 0 --reg 0.001 --var 10.0 --pol linear --env hopper --horizon 100 --gamma 1 --lr 0.01 0.1 0.1 --lr_strategy adam --n_workers 8 --batch 100 --n_trials 5 --env_param 2 --c_bounds 25 --l_init 0 --eta_init 0
python3 run_cost.py --dir /data1/montenegro_cesani/ --ite 3000 --alg cpgpe --risk tc --risk_param 0 --reg 0.001 --var 100.0 --pol linear --env hopper --horizon 100 --gamma 1 --lr 0.01 0.1 0.1 --lr_strategy adam --n_workers 8 --batch 100 --n_trials 5 --env_param 2 --c_bounds 25 --l_init 0 --eta_init 0


# Commands for the server:
taskset -ca 5-16 python3 run_cost.py --dir /data1/montenegro_cesani/ --ite 3000 --alg cpgpe --risk tc --risk_param 0 --reg 0.001 --var 0.01 --pol linear --env swimmer --horizon 100 --gamma 1 --lr 0.001 0.01 0.1 --lr_strategy adam --n_workers 12 --batch 100 --n_trials 5 --env_param 2 --c_bounds 50 --l_init 0 --eta_init 0

taskset -ca 5-16 python3 run_cost.py --dir /data1/montenegro_cesani/ --ite 3000 --alg cpgpe --risk tc --risk_param 0 --reg 0.001 --var 0.05 --pol linear --env swimmer --horizon 100 --gamma 1 --lr 0.001 0.01 0.1 --lr_strategy adam --n_workers 12 --batch 100 --n_trials 5 --env_param 2 --c_bounds 50 --l_init 0 --eta_init 0

taskset -ca 5-16 python3 run_cost.py --dir /data1/montenegro_cesani/ --ite 3000 --alg cpgpe --risk tc --risk_param 0 --reg 0.001 --var 0.075 --pol linear --env swimmer --horizon 100 --gamma 1 --lr 0.001 0.01 0.1 --lr_strategy adam --n_workers 12 --batch 100 --n_trials 5 --env_param 2 --c_bounds 50 --l_init 0 --eta_init 0

taskset -ca 5-16 python3 run_cost.py --dir /data1/montenegro_cesani/ --ite 3000 --alg cpgpe --risk tc --risk_param 0 --reg 0.001 --var 0.001 --pol linear --env swimmer --horizon 100 --gamma 1 --lr 0.001 0.01 0.1 --lr_strategy adam --n_workers 12 --batch 100 --n_trials 5 --env_param 2 --c_bounds 50 --l_init 0 --eta_init 0

taskset -ca 5-16 python3 run_cost.py --dir /data1/montenegro_cesani/ --ite 3000 --alg cpgpe --risk tc --risk_param 0 --reg 0.001 --var 0.005 --pol linear --env swimmer --horizon 100 --gamma 1 --lr 0.001 0.01 0.1 --lr_strategy adam --n_workers 12 --batch 100 --n_trials 5 --env_param 2 --c_bounds 50 --l_init 0 --eta_init 0

taskset -ca 5-16 python3 run_cost.py --dir /data1/montenegro_cesani/ --ite 3000 --alg cpgpe --risk tc --risk_param 0 --reg 0.001 --var 0.0075 --pol linear --env swimmer --horizon 100 --gamma 1 --lr 0.001 0.01 0.1 --lr_strategy adam --n_workers 12 --batch 100 --n_trials 5 --env_param 2 --c_bounds 50 --l_init 0 --eta_init 0