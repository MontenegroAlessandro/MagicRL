cd ..
# CPGPE
# python3 run_cost.py --dir /Users/ale/PyProjects/results/cpgpe/ --ite 6000 --alg cpgpe --risk tc --risk_param 0 --reg 0.0001 --var 0.001 --pol linear --env lqr --horizon 50 --gamma 1 --lr 0.001 0.01 0.001 --lr_strategy adam --n_workers 8 --batch 100 --n_trials 1 --env_param 2 --c_bounds 0.2 --l_init 0 --eta_init 0
# python3 run_cost.py --dir /Users/ale/PyProjects/results/cpgpe/ --ite 6000 --alg cpgpe --risk cvar --risk_param 0.95 --reg 0.0001 --var 0.001 --pol linear --env lqr --horizon 50 --gamma 1 --lr 0.001 0.01 0.001 --lr_strategy adam --n_workers 8 --batch 100 --n_trials 1 --env_param 2 --c_bounds 3 --l_init 0 --eta_init 0
# python3 run_cost.py --dir /Users/ale/PyProjects/results/cpgpe/ --ite 6000 --alg cpgpe --risk mv --risk_param 0.95 --reg 0.0001 --var 0.001 --pol linear --env lqr --horizon 50 --gamma 1 --lr 0.001 0.01 0.001 --lr_strategy adam --n_workers 8 --batch 100 --n_trials 1 --env_param 2 --c_bounds 0.2 --l_init 0 --eta_init 0
# python3 run_cost.py --dir /Users/ale/PyProjects/results/cpgpe/ --ite 6000 --alg cpgpe --risk chance --risk_param 0.3 --reg 0.0001 --var 0.001 --pol linear --env lqr --horizon 50 --gamma 1 --lr 0.001 0.01 0.001 --lr_strategy adam --n_workers 8 --batch 100 --n_trials 1 --env_param 2 --c_bounds 0.2 --l_init 0 --eta_init 0

# CPG
# python3 run_cost.py --dir /Users/ale/PyProjects/results/cpg/ --ite 6000 --alg cpg --risk tc --risk_param 0 --reg 0.0001 --var 0.001 --pol linear --env lqr --horizon 50 --gamma 1 --lr 0.001 0.01 0.001 --lr_strategy adam --n_workers 8 --batch 100 --n_trials 1 --env_param 2 --c_bounds 0.2 --l_init 0 --eta_init 0
# python3 run_cost.py --dir /Users/ale/PyProjects/results/cpg/ --ite 6000 --alg cpg --risk cvar --risk_param 0.95 --reg 0.0001 --var 0.001 --pol linear --env lqr --horizon 50 --gamma 1 --lr 0.001 0.01 0.001 --lr_strategy adam --n_workers 8 --batch 100 --n_trials 1 --env_param 2 --c_bounds 3 --l_init 0 --eta_init 0
# python3 run_cost.py --dir /Users/ale/PyProjects/results/cpg/ --ite 6000 --alg cpg --risk mv --risk_param 0.1 --reg 0.0001 --var 0.001 --pol linear --env lqr --horizon 50 --gamma 1 --lr 0.001 0.01 0.001 --lr_strategy adam --n_workers 8 --batch 100 --n_trials 1 --env_param 2 --c_bounds 0.2 --l_init 0 --eta_init 0

# NPGPD
python3 run_cost.py --dir /Users/ale/PyProjects/results/npgpd/ --ite 200 --alg npgpd --risk tc --pol softmax --env gw_d --horizon 100 --gamma 0.99 --lr 0.001 0.01 --lr_strategy adam --n_workers 8 --batch 10 --n_trials 1 --c_bounds 0.2
# python3 run_cost.py --dir /Users/ale/PyProjects/results/cpg/ --ite 3000 --alg cpg --risk tc --risk_param 0 --reg 0.0001 --var 0 --pol softmax --env gw_d --horizon 100 --gamma 0.99 --lr 0.01 0.1 0.01 --lr_strategy adam --n_workers 8 --batch 100 --n_trials 1 --env_param 0 --c_bounds 0.2 --l_init 0 --eta_init 0