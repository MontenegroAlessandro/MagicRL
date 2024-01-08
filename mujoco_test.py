# Libraries
import copy

from envs import *
from policies import LinearPolicy, NeuralNetworkPolicy, LinearGaussianPolicy, DeepGaussian
from algorithms import PGPE, PolicyGradient
from data_processors import IdentityDataProcessor
from art import *
import torch
import torch.nn as nn
import json

"""Global Vars"""
# general
MODE = "learn_test"

env_selection = ["half_cheetah", "swimmer"]
ENV = env_selection[1]

pol_selection = ["nn_policy", "linear", "gaussian"]
POL = pol_selection[0]

alg_selection = ["pg", "pgpe"]
ALG = alg_selection[0]

# environment
horizon = 200
gamma = 0.995
RENDER = False

# algorithm
DEBUG = False
NATURAL = False
ITE = 500
BATCH = 100
N_JOBS_PARAM = 8
LR_STRATEGY = "adam"
if ALG == "pgpe":
    dir = f"/Users/ale/results/pgpe/pgpe_test_{ITE}_"
    LEARN_STD = False
    EPISODES = 1
    N_JOBS_TRAJ = 1
    STD_DECAY = 0   # 1e-5
    STD_MIN = 1e-4
else:
    dir = f"/Users/ale/results/pg/pg_test_{ITE}_"
    ESTIMATOR = "GPOMDP"

if LR_STRATEGY == "adam" and ALG == "pgpe":
    INIT_LR = 1e-1
    dir += "adam_01_"
elif LR_STRATEGY == "adam" and ALG == "pg":
    INIT_LR = 1e-2
    dir += "adam_001_"
else:
    INIT_LR = 1e-3
    dir += "clr_0001_"

# test
test_ite = horizon
num_test = 10

"""Environment"""
if ENV == "half_cheetah":
    env_class = HalfCheetah
    env = HalfCheetah(horizon=horizon, gamma=gamma, render=RENDER)
    dir += f"half_cheetah_{horizon}_"
    MULTI_LINEAR = True
elif ENV == "swimmer":
    env_class = Swimmer
    env = Swimmer(horizon=horizon, gamma=gamma, render=RENDER)
    dir += f"swimmer_{horizon}_"
    MULTI_LINEAR = True
else:
    raise NotImplementedError
s_dim = env.state_dim
a_dim = env.action_dim

"""Data Processor"""
dp = IdentityDataProcessor()

"""Policy"""
if POL == "nn_policy":
    net = nn.Sequential(
        nn.Linear(s_dim, 5, bias=False),
        nn.Linear(5, a_dim, bias=False)
    )
    model_desc = dict(
        layers_shape=[(s_dim, 5), (5, a_dim)]
    )
    if ALG == "pgpe":
        pol = NeuralNetworkPolicy(
            parameters=None,
            input_size=s_dim,
            output_size=a_dim,
            model=copy.deepcopy(net),
            model_desc=copy.deepcopy(model_desc)
        )
        tot_params = pol.tot_params
    else:
        pol = DeepGaussian(
            parameters=None,
            input_size=s_dim,
            output_size=a_dim,
            model=copy.deepcopy(net),
            model_desc=copy.deepcopy(model_desc),
            std_dev=np.sqrt(0.01),
            std_decay=0,
            std_min=1e-6
        )
        tot_params = pol.tot_params
    dir += f"nn_policy_{tot_params}"
    dir += "_var_001"
elif POL == "linear":
    pol = LinearPolicy(
        parameters=np.ones((s_dim * a_dim)),
        dim_state=s_dim,
        dim_action=a_dim,
        multi_linear=MULTI_LINEAR
    )
    tot_params = s_dim * a_dim
    dir += f"linear_policy_{tot_params}"
elif POL == "gaussian":
    tot_params = s_dim * a_dim
    pol = LinearGaussianPolicy(
        parameters=np.ones(tot_params),
        dim_state=s_dim,
        dim_action=a_dim,
        std_dev=np.sqrt(1),
        std_decay=0,
        std_min=1e-6,
        multi_linear=MULTI_LINEAR
    )
    dir += f"lingauss_policy_{tot_params}_var_1"
else:
    raise NotImplementedError

"""Algorithms"""
if ALG == "pgpe":
    hp = np.zeros((2, tot_params))
    hp[0] = [0.5] * tot_params
    # hp[1] = [0.001] * tot_params                    # var = 1
    # dir += "_var_1"
    # hp[1] = [1.151292546497023] * tot_params      # var = 10
    # dir += "_var_10"
    # hp[1] = [2.302585092994046] * tot_params      # var = 100
    # dir += "_var_100"
    # hp[1] = [-1.1512925464970227] * tot_params    # var = 0.1
    # dir += "_var_01"
    hp[1] = [-2.3025850929940455] * tot_params    # var = 0.01
    dir += "_var_001"
    alg_parameters = dict(
        lr=[INIT_LR],
        initial_rho=hp,
        ite=ITE,
        batch_size=BATCH,
        episodes_per_theta=EPISODES,
        env=env,
        policy=pol,
        data_processor=dp,
        directory=dir,
        verbose=DEBUG,
        natural=NATURAL,
        checkpoint_freq=100,
        lr_strategy=LR_STRATEGY,
        learn_std=LEARN_STD,
        std_decay=STD_DECAY,
        std_min=STD_MIN,
        n_jobs_param=N_JOBS_PARAM,
        n_jobs_traj=N_JOBS_TRAJ
    )
    alg = PGPE(**alg_parameters)
else:
    alg_parameters = dict(
        lr=[INIT_LR],
        lr_strategy=LR_STRATEGY,
        estimator_type=ESTIMATOR,
        initial_theta=[0.5] * tot_params,
        ite=ITE,
        batch_size=BATCH,
        env=env,
        policy=pol,
        data_processor=dp,
        directory=dir,
        verbose=DEBUG,
        natural=NATURAL,
        checkpoint_freq=100,
        n_jobs=N_JOBS_PARAM
    )
    alg = PolicyGradient(**alg_parameters)

if __name__ == "__main__":
    # Learn phase
    print(text2art("== MUJOCO TEST =="))
    if MODE in ["learn", "learn_test"]:
        print(text2art("Learn Start"))
        alg.learn()
        alg.save_results()
        print(alg.performance_idx)

    # Test phase
    # todo aggiusta il fatto dello 0 e vedi di mettere il std decay
    # todo to clip or not to clip
    if MODE in ["test", "learn_test"]:
        print(text2art("** TEST **"))
        env = env_class(horizon=horizon, gamma=gamma, render=True)
        if ALG == "pgpe":
            pol.set_parameters(thetas=alg.best_theta[0])
        else:
            pol.set_parameters(thetas=alg.best_theta)
            pol.std_dev = 0
        for _ in range(num_test):
            env.reset()
            state = env.state
            r = 0
            for i in range(test_ite):
                state, rew, _, _ = env.step(action=pol.draw_action(state))
                r += (gamma ** i) * rew
            print(f"PERFORMANCE: {r}")
